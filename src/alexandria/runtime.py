"""Stage runtime: registry, run context, resumable map/reduce helpers, manifests and dashboard panels.

A stage is a function registered with @stage. It runs on its own (`alexandria ws1 <stage>`), reads its inputs from
files earlier stages wrote, and writes everything under   <out_root>/<profile>/<ws>/<stage>/ :
    parts/                one output per input shard (map stages; resumable, atomic)
    manifest.json         what ran: config fingerprint, versions, host, SLURM job, timing, status
    dashboard.json        panels the dashboard renders (KPIs, tables, charts)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config
    from .data import Shard

STAGES: dict[tuple[str, str], StageSpec] = {}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # strict JSON: browsers' JSON.parse rejects NaN/Infinity (np.float64 is a float, so _json_default never sees it)
    tmp.write_text(json.dumps(_finite(obj), indent=1, default=_json_default, allow_nan=False))
    os.replace(tmp, path)


def _finite(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def atomic_write_parquet(path: Path, table) -> None:
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, dt.date)):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


@dataclass
class StageSpec:
    ws: str
    name: str
    fn: Callable[[Context], Any]
    help: str
    kind: str = "reduce"          # map = one output per input shard (SLURM array-able); reduce = one job
    needs: tuple[str, ...] = ()   # stages whose outputs this one reads (for `submit --with-deps` and checks)
    resources: str = "cpu"        # key of site.slurm (cpu | gpu)
    env: str = "main"             # key of site.envs (main = Alex, llm = Alex-vllm)
    config_keys: tuple[str, ...] = ()


def stage(ws: str, name: str, *, help: str, kind: str = "reduce", needs: tuple[str, ...] = (),
          resources: str = "cpu", env: str = "main", config_keys: tuple[str, ...] = ()):
    def deco(fn):
        STAGES[(ws, name)] = StageSpec(ws, name, fn, help, kind, needs, resources, env, config_keys)
        return fn
    return deco


@dataclass
class Context:
    cfg: Config
    spec: StageSpec
    shard_index: int = 0
    shard_count: int = 1
    force: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def out(self) -> Path:
        return self.cfg.stage_dir(self.spec.ws, self.spec.name)

    def input_dir(self, stage_name: str, ws: str | None = None) -> Path:
        d = self.cfg.stage_dir(ws or self.spec.ws, stage_name)
        if not d.exists():
            raise FileNotFoundError(f"stage {stage_name!r} has no output at {d}; run it first")
        return d

    def section(self, dotted: str) -> dict[str, Any]:
        cur: Any = self.cfg.raw
        for part in dotted.split("."):
            cur = (cur or {}).get(part, {})
        return cur or {}

    # ---------------------------------------------------------------- map helper
    @property
    def workers(self) -> int:
        """Processes for map stages: --workers, else the CPUs SLURM gave this job, else 1."""
        return int(self.extra.get("workers") or os.environ.get("SLURM_CPUS_PER_TASK") or 1)

    def map_shards(self, fn: Callable[[Config, Shard], Any], shards: list[Shard] | None = None) -> dict[str, int]:
        """Run fn(cfg, shard) -> pyarrow.Table for this task's shards in `workers` processes.

        fn must be a module-level function (it is sent to worker processes). One output per shard in parts/;
        shards whose output exists are skipped, so a re-submitted job resumes where the last one stopped."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from .data import input_shards, partition
        allshards = shards if shards is not None else input_shards(self.cfg)
        mine = partition(allshards, self.shard_index, self.shard_count)
        parts = self.out / "parts"
        parts.mkdir(parents=True, exist_ok=True)

        def dst(sh: Shard) -> Path:
            return parts / (sh.rel.replace("/", "__").removesuffix(".parquet") + ".parquet")

        # parts made under a different config are stale: recompute them instead of silently mixing settings
        fp = self.cfg.fingerprint(*self.spec.config_keys, "data")
        fp_file = parts / "_config_fingerprint.json"
        stale = False
        if fp_file.exists():
            old = json.loads(fp_file.read_text()).get("fingerprint")
            stale = old != fp
            if stale:
                log(f"{self.spec.name}: config changed ({old} -> {fp}); recomputing existing parts")
        elif any(parts.glob("*.parquet")):
            stale = True
            log(f"{self.spec.name}: parts without a config fingerprint; recomputing")
        if not fp_file.exists() or stale:
            atomic_write_json(fp_file, {"fingerprint": fp, "config": {k: self.section(k) for k in self.spec.config_keys}})
        todo = [sh for sh in mine if self.force or stale or not dst(sh).exists()]
        stats = {"shards_total": len(allshards), "shards_mine": len(mine), "shards_skipped": len(mine) - len(todo),
                 "shards_done": 0, "rows": 0, "workers": self.workers}
        t0 = time.time()

        def finish(sh: Shard, tbl) -> None:
            atomic_write_parquet(dst(sh), tbl)
            stats["shards_done"] += 1
            stats["rows"] += tbl.num_rows
            n = stats["shards_done"]
            if n % 25 == 0 or n == len(todo):
                rate = stats["rows"] / max(time.time() - t0, 1e-9)
                log(f"{self.spec.name}: {n}/{len(todo)} shards ({stats['rows']} rows, {rate:.0f} rows/s)")

        if self.workers <= 1 or len(todo) <= 1:
            for sh in todo:
                finish(sh, fn(self.cfg, sh))
        else:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(todo))) as pool:
                futures = {pool.submit(fn, self.cfg, sh): sh for sh in todo}
                for fut in as_completed(futures):
                    finish(futures[fut], fut.result())
        return stats

    def parts(self, stage_name: str | None = None, ws: str | None = None) -> list[Path]:
        d = (self.out if stage_name is None else self.input_dir(stage_name, ws)) / "parts"
        return sorted(d.glob("*.parquet"))

    # ---------------------------------------------------------------- outputs
    def publish(self, title: str, panels: list[dict[str, Any]], summary: str = "") -> None:
        """Panels for the dashboard (see dashboard/README in the static app for the panel types)."""
        atomic_write_json(self.out / "dashboard.json", {
            "ws": self.spec.ws, "stage": self.spec.name, "title": title, "summary": summary,
            "profile": self.cfg.profile, "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "panels": panels})

    def skip(self, title: str, reason: str) -> dict:
        """Mark this run as deliberately skipped (status `skipped`, not `done`) and say why on the dashboard."""
        self.publish(title, [{"type": "markdown", "text": f"**Skipped.** {reason}"}], summary="skipped")
        return {"skipped": reason}

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.out / name
        atomic_write_json(p, obj)
        return p


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pyarrow", "pandas", "duckdb", "pydantic", "rapidfuzz", "datasketch", "torch",
                "transformers", "vllm", "fasttext"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", "?")
    return out


def run_stage(ctx: Context) -> int:
    """Run one stage with a manifest; returns a process exit code."""
    ctx.out.mkdir(parents=True, exist_ok=True)
    task = f".task{ctx.shard_index}" if ctx.shard_count > 1 else ""
    mpath = ctx.out / f"manifest{task}.json"
    from . import __version__
    manifest: dict[str, Any] = {
        "ws": ctx.spec.ws, "stage": ctx.spec.name, "profile": ctx.cfg.profile, "kind": ctx.spec.kind,
        "shard": [ctx.shard_index, ctx.shard_count], "status": "running",
        "started": dt.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "code_version": __version__, "config_fingerprint": ctx.cfg.fingerprint(*ctx.spec.config_keys, "data"),
        "config": {k: ctx.section(k) for k in ctx.spec.config_keys}, "argv": sys.argv,
        "dataset": ctx.cfg.dataset.name, "llm_overrides": ctx.section("site.llm") or None,
    }
    atomic_write_json(mpath, manifest)
    t0 = time.time()
    log(f"stage {ctx.spec.ws}/{ctx.spec.name} [{ctx.cfg.profile}] shard {ctx.shard_index + 1}/{ctx.shard_count} -> {ctx.out}")
    try:
        result = ctx.spec.fn(ctx)
        skipped = isinstance(result, dict) and "skipped" in result
        manifest.update(status="skipped" if skipped else "done", result=result if isinstance(result, dict) else None)
        code = 0
    except Exception as e:                               # noqa: BLE001  (recorded in the manifest, then re-raised)
        manifest.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        code = 1
        traceback.print_exc()
    manifest.update(finished=dt.datetime.now().isoformat(timespec="seconds"),
                    seconds=round(time.time() - t0, 1), versions=_versions())
    atomic_write_json(mpath, manifest)
    log(f"stage {ctx.spec.name}: {manifest['status']} in {manifest['seconds']} s")
    return code
