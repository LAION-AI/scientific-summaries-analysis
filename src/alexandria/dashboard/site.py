"""Build and serve the static dashboard.

build(cfg) collects every stage's dashboard.json and manifest*.json under <out_root>/<profile>/ into
<out_root>/<profile>/dashboard/ (data.json + the static app). serve(root, host, port) serves that folder on
localhost; with `cfg` and `rebuild_every` it also rebuilds in the background so the page follows running jobs.
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import shutil
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..runtime import STAGES, atomic_write_json, log

if TYPE_CHECKING:
    from ..config import Config

STATIC = Path(__file__).with_name("static")
CDN_CHART = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log(f"dashboard: skipping unreadable {p}: {e}")
        return None


def _run_summary(m: dict, path: Path) -> dict:
    err = (m.get("error") or "").strip().splitlines()
    return {"ws": m.get("ws"), "stage": m.get("stage"), "status": m.get("status"), "shard": m.get("shard"),
            "started": m.get("started"), "finished": m.get("finished"), "seconds": m.get("seconds"),
            "host": m.get("host"), "slurm_job": m.get("slurm_job"), "slurm_array_task": m.get("slurm_array_task"),
            "error": err[0] if err else None, "traceback": m.get("traceback"),
            "config_fingerprint": m.get("config_fingerprint"), "result": m.get("result"), "file": path.name}


def _stage_status(runs: list[dict]) -> str:
    """Combine the manifests of one stage (one per array task) into one status."""
    if not runs:
        return "not run"
    states = {r.get("status") for r in runs}
    if "failed" in states:
        return "failed"
    if "running" in states:
        return "running"
    if states == {"skipped"}:
        return "skipped"
    return "done" if states == {"done"} else "partial"


def _copy_images(payload: dict, stage_dir: Path, root: Path, ws: str, stage: str) -> None:
    for panel in payload.get("panels") or []:
        if panel.get("type") != "image" or not panel.get("src"):
            continue
        src = stage_dir / panel["src"]
        if not src.exists():
            panel["missing"] = True
            continue
        dst = root / "assets" / ws / stage / panel["src"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        panel["src"] = f"assets/{ws}/{stage}/{panel['src']}"


def _copy_static(root: Path) -> None:
    for p in STATIC.rglob("*"):
        if p.is_file():
            dst = root / p.relative_to(STATIC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp")
            shutil.copy2(p, tmp)
            tmp.replace(dst)


def _copy_reports(base: Path, root: Path) -> list[dict]:
    """Copy every workstream's PDF report (<ws>/report/*.pdf) next to the dashboard so the page can link it."""
    out = []
    for pdf in sorted(base.glob("*/report/*.pdf")):
        dst = root / "reports" / pdf.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_mtime < pdf.stat().st_mtime:
            shutil.copy2(pdf, dst)
        out.append({"ws": pdf.parent.parent.name, "name": pdf.name, "href": f"reports/{pdf.name}",
                    "updated": dt.datetime.fromtimestamp(pdf.stat().st_mtime).isoformat(timespec="seconds")})
    return out


def build(cfg: Config) -> Path:
    base = cfg.run_root()
    root = base / "dashboard"
    root.mkdir(parents=True, exist_ok=True)
    _copy_static(root)

    # stages in registration (pipeline) order, then anything on disk that is not registered (older code)
    registry = [{"ws": s.ws, "name": s.name, "help": s.help, "needs": list(s.needs), "kind": s.kind,
                 "resources": s.resources, "env": s.env} for s in STAGES.values()]
    order = [(r["ws"], r["name"]) for r in registry]
    on_disk = sorted({(p.parent.parent.name, p.parent.name)
                      for pat in ("*/*/dashboard.json", "*/*/manifest*.json") for p in base.glob(pat)
                      if p.parent.parent.name not in ("dashboard", "_sample")})
    for key in on_disk:
        if key not in order and key[0] != "reports":
            order.append(key)
            registry.append({"ws": key[0], "name": key[1], "help": "", "needs": [], "kind": "?",
                             "resources": "?", "env": "?", "unregistered": True})

    runs: list[dict] = []
    workstreams: dict[str, list[dict]] = {}
    for ws, stage in order:
        sdir = base / ws / stage
        stage_runs = []
        for mp in sorted(sdir.glob("manifest*.json")):
            m = _read_json(mp)
            if isinstance(m, dict):
                stage_runs.append(_run_summary(m, mp))
        runs += stage_runs
        payload = _read_json(sdir / "dashboard.json") if (sdir / "dashboard.json").exists() else None
        if isinstance(payload, dict):
            _copy_images(payload, sdir, root, ws, stage)
        entry = {"ws": ws, "stage": stage, "status": _stage_status(stage_runs),
                 "title": (payload or {}).get("title") or stage, "summary": (payload or {}).get("summary") or "",
                 "updated": (payload or {}).get("updated"), "panels": (payload or {}).get("panels") or []}
        workstreams.setdefault(ws, []).append(entry)

    status = {f"{e['ws']}/{e['stage']}": e["status"] for es in workstreams.values() for e in es}
    for r in registry:
        r["status"] = status.get(f"{r['ws']}/{r['name']}", "not run")
    runs.sort(key=lambda r: r.get("started") or "", reverse=True)
    atomic_write_json(root / "data.json", {
        "profile": cfg.profile, "built": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": {"name": cfg.dataset.name, "title": cfg.dataset.title or cfg.dataset.name,
                    "description": cfg.dataset.description, "homepage": cfg.dataset.homepage,
                    "groups": [s.group for s in cfg.data.sources]},
        "reports": _copy_reports(base, root),
        "out_root": str(base), "registry": registry, "workstreams": workstreams, "runs": runs,
        "chart_cdn": CDN_CHART})
    return root


class _NoCache(SimpleHTTPRequestHandler):
    def log_message(self, *a: Any) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def make_server(root: Path, host: str = "127.0.0.1", port: int = 8770) -> ThreadingHTTPServer:
    handler = functools.partial(_NoCache, directory=str(Path(root).resolve()))
    for p in range(port, port + 21):
        try:
            srv = ThreadingHTTPServer((host, p), handler)
            srv.daemon_threads = True
            return srv
        except OSError:
            continue
    raise RuntimeError(f"no free port in {port}..{port + 20}")


def serve(root: Path, host: str = "127.0.0.1", port: int = 8770, cfg: Config | None = None,
          rebuild_every: int = 60) -> None:
    """Serve `root` until Ctrl-C. With cfg, rebuild data.json every `rebuild_every` s (0 = never)."""
    srv = make_server(root, host, port)
    p = srv.server_address[1]
    print(f"dashboard: http://localhost:{p}/   (Ctrl-C to stop)")
    print(f"  VS Code (Remote-SSH) forwards port {p} automatically; otherwise on your laptop:\n"
          f"  ssh -L {p}:localhost:{p} <login node>   and open http://localhost:{p}/", flush=True)
    stop = threading.Event()
    if cfg is not None and rebuild_every > 0:
        def loop() -> None:
            while not stop.wait(rebuild_every):
                try:
                    build(cfg)
                except Exception as e:                    # noqa: BLE001  (keep serving the last good build)
                    log(f"dashboard rebuild failed: {e}")
        threading.Thread(target=loop, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()


def serve_forever_in_thread(root: Path, host: str = "127.0.0.1", port: int = 8770) -> tuple[ThreadingHTTPServer, str]:
    """For tests / notebooks: start in a daemon thread -> (server, url). Call server.shutdown() to stop."""
    srv = make_server(root, host, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.05)
    return srv, f"http://{host}:{srv.server_address[1]}/"
