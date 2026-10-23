"""`alexandria submit <ws> <stage>`: submit a stage (optionally with the stages it needs) as SLURM jobs.

Resources come from configs/site.yaml (slurm.cpu / slurm.gpu; the debug profile uses slurm.debug for CPU stages and a
1-hour slurm.gpu job for GPU stages), the Python env from the stage's `env` (main = Alex, llm = Alex-vllm). Stages
needed by the requested one are submitted first and chained with --dependency=afterok, so a failed stage stops the
chain. Already-finished stages (manifest status done) are skipped unless --force.

    alexandria submit ws1 quality --profile full --with-deps     # features -> quality
    alexandria submit ws1 features --profile full --array 2      # 2 jobs, each with all CPUs of its node slice
    alexandria submit ws1 all --profile debug                    # every WS1 stage in dependency order
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .config import PROFILES, REPO_ROOT, load_config
from .runtime import STAGES, StageSpec


def _done(cfg, spec: StageSpec) -> bool:
    """True when the stage has manifests and every one of them (all array tasks) says done."""
    manifests = list(cfg.stage_dir(spec.ws, spec.name).glob("manifest*.json"))
    try:
        return bool(manifests) and all(json.loads(m.read_text()).get("status") == "done" for m in manifests)
    except json.JSONDecodeError:
        return False


def _order(ws: str, targets: list[str], with_deps: bool) -> list[StageSpec]:
    seen: list[str] = []

    def visit(name: str) -> None:
        if name in seen:
            return
        if with_deps:
            for dep in STAGES[(ws, name)].needs:
                visit(dep)
        seen.append(name)

    for t in targets:
        visit(t)
    return [STAGES[(ws, n)] for n in seen]


def sbatch_cmd(cfg, spec: StageSpec, profile: str, overrides: list[str], deps: list[str], array: int,
               force: bool) -> list[str]:
    s = cfg.site.slurm
    res = s.gpu if spec.resources == "gpu" else (s.debug if profile == "debug" else s.cpu)
    time = res.debug_time if profile == "debug" else res.time
    cmd = ["sbatch", "--parsable", f"--account={s.account}", f"--partition={res.partition}",
           f"--cpus-per-task={res.cpus}", f"--mem={res.mem}", f"--time={time}",
           f"--job-name=alex_{spec.name}", f"--output={cfg.site.logs}/alexandria_{spec.name}_%j.out"]
    if res.gres:
        cmd.append(f"--gres={res.gres}")
    if deps:
        cmd.append("--dependency=afterok:" + ":".join(deps))
    if spec.kind == "map" and array > 1:
        cmd.append(f"--array=0-{array - 1}")
        cmd[cmd.index(next(c for c in cmd if c.startswith("--output=")))] = \
            f"--output={cfg.site.logs}/alexandria_{spec.name}_%A_%a.out"
    alex = [spec.ws, spec.name, "--profile", profile, "--dataset", cfg.dataset.name] + \
        [x for o in overrides for x in ("--set", o)]
    if force:
        alex.append("--force")
    return cmd + [str(REPO_ROOT / "slurm" / "stage.sbatch"), f"ENV={spec.env}", "--", *alex]


def main(argv: list[str]) -> int:
    from .cli import _load_stages
    _load_stages()
    p = argparse.ArgumentParser(prog="alexandria submit", description=__doc__.split("\n\n")[0])
    p.add_argument("ws")
    p.add_argument("stages", nargs="+", help="stage names, or 'all'")
    p.add_argument("--profile", choices=PROFILES, default="debug")
    p.add_argument("--dataset", default=None, metavar="NAME", help="configs/datasets/<NAME>.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--with-deps", action="store_true", help="also submit the stages it needs (skips finished ones)")
    p.add_argument("--array", type=int, default=1, help="map stages: split the shards over N array jobs")
    p.add_argument("--force", action="store_true", help="re-run even if finished")
    p.add_argument("--after", action="append", default=[], metavar="JOBID",
                   help="also wait for these SLURM jobs (afterok), e.g. a pipeline job submitted earlier")
    p.add_argument("--one-job", action="store_true",
                   help="run all listed stages in ONE allocation, in order (grete:interactive allows 2 jobs/user)")
    p.add_argument("--dry-run", action="store_true", help="print the sbatch commands only")
    a = p.parse_args(argv)
    names = [n for (w, n) in STAGES if w == a.ws] if a.stages == ["all"] else a.stages
    cfg = load_config(a.ws, a.profile, a.overrides, dataset=a.dataset)
    Path(cfg.site.logs).mkdir(parents=True, exist_ok=True)
    if a.one_job:
        specs = _order(a.ws, names, a.with_deps or a.stages == ["all"])
        if not a.force:
            specs = [sp for sp in specs if sp.name in names or not _done(cfg, sp)]
        envs = {sp.env for sp in specs}
        if len(envs) > 1:
            print(f"--one-job needs one Python env; these stages use {sorted(envs)} (submit GPU stages separately)")
            return 2
        gpu = any(sp.resources == "gpu" for sp in specs)
        first = specs[0]
        cmd = sbatch_cmd(cfg, first if not gpu else next(sp for sp in specs if sp.resources == "gpu"),
                         a.profile, a.overrides, list(a.after), 1, a.force)
        cmd = [f"--job-name=alex_{a.ws}_pipeline" if c.startswith("--job-name=") else
               f"--output={cfg.site.logs}/alexandria_{a.ws}_pipeline_%j.out" if c.startswith("--output=") else c
               for c in cmd]
        cut = cmd.index("--")
        cmd = cmd[:cut + 1] + ["pipeline", a.ws, *[sp.name for sp in specs], "--profile", a.profile,
                               "--dataset", cfg.dataset.name] + \
            [x for o in a.overrides for x in ("--set", o)] + (["--force"] if a.force else [])
        if a.dry_run:
            print(shlex.join(cmd))
            return 0
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"sbatch failed: {out.stderr.strip()}")
            return 1
        print(f"pipeline job {out.stdout.strip()}: {' -> '.join(sp.name for sp in specs)}")
        return 0
    jobs: dict[str, str] = {}
    for spec in _order(a.ws, names, a.with_deps or a.stages == ["all"]):
        if not a.force and spec.name not in names and _done(cfg, spec):
            print(f"skip {spec.name:18} (done)")
            continue
        deps = [jobs[d] for d in spec.needs if d in jobs] + list(a.after)
        cmd = sbatch_cmd(cfg, spec, a.profile, a.overrides, deps, a.array, a.force)
        if a.dry_run:
            print(shlex.join(cmd))
            jobs[spec.name] = f"<{spec.name}>"
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"sbatch failed for {spec.name}: {out.stderr.strip()}")
            return 1
        jobs[spec.name] = out.stdout.strip().split(";")[0]
        print(f"{spec.name:18} job {jobs[spec.name]}  ({spec.resources}, env={spec.env})"
              + (f"  after {','.join(deps)}" if deps else ""))
    return 0
