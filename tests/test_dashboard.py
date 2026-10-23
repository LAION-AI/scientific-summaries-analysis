"""Dashboard build + serve on a temporary results folder (no cluster data needed)."""
import json
import urllib.request

import pytest

from alexandria.config import load_config
from alexandria.dashboard import build, serve_forever_in_thread


@pytest.fixture()
def cfg(tmp_path):
    return load_config("ws1", "debug", overrides=[f"site.out_root={tmp_path}"])


def _stage(base, ws, name, payload=None, manifests=()):
    d = base / ws / name
    d.mkdir(parents=True)
    if payload is not None:
        (d / "dashboard.json").write_text(json.dumps(payload))
    for fname, m in manifests:
        (d / fname).write_text(json.dumps(m))
    return d


def test_build_collects_panels_manifests_and_images(cfg, tmp_path):
    base = tmp_path / "debug"
    d = _stage(base, "ws1", "alpha",
               {"title": "Alpha", "summary": "s", "panels": [
                   {"type": "kpis", "items": [{"label": "n", "value": 3, "fmt": "int"}]},
                   {"type": "bar", "title": "b", "labels": ["a", "b"],
                    "series": [{"name": "arxiv/new", "data": [0.1, 0.2], "lo": [0.05, 0.1], "hi": [0.2, 0.3]}]},
                   {"type": "image", "title": "fig", "src": "figures/f.png"},
                   {"type": "image", "title": "gone", "src": "figures/missing.png"}]},
               [("manifest.json", {"ws": "ws1", "stage": "alpha", "status": "done", "started": "2026-10-21T10:00:00",
                                   "seconds": 1.5, "shard": [0, 1]})])
    (d / "figures").mkdir()
    (d / "figures" / "f.png").write_bytes(b"\x89PNG fake")
    _stage(base, "ws1", "beta", None,
           [("manifest.task0.json", {"ws": "ws1", "stage": "beta", "status": "done", "shard": [0, 2]}),
            ("manifest.task1.json", {"ws": "ws1", "stage": "beta", "status": "failed", "shard": [1, 2],
                                     "error": "ValueError: boom\nmore", "traceback": "Traceback ..."})])

    root = build(cfg)
    data = json.loads((root / "data.json").read_text())
    assert data["profile"] == "debug"
    stages = {s["stage"]: s for s in data["workstreams"]["ws1"]}
    assert stages["alpha"]["status"] == "done" and stages["alpha"]["title"] == "Alpha"
    assert stages["beta"]["status"] == "failed" and stages["beta"]["panels"] == []
    imgs = [p for p in stages["alpha"]["panels"] if p["type"] == "image"]
    assert imgs[0]["src"] == "assets/ws1/alpha/figures/f.png" and (root / imgs[0]["src"]).exists()
    assert imgs[1].get("missing") is True
    failed = [r for r in data["runs"] if r["status"] == "failed"]
    assert failed[0]["error"] == "ValueError: boom"
    reg = {(r["ws"], r["name"]): r for r in data["registry"]}
    assert reg[("ws1", "beta")]["status"] == "failed"
    for f in ("index.html", "app.js", "style.css", "vendor/chart.umd.min.js"):
        assert (root / f).exists(), f


def test_build_empty_and_serve(cfg):
    root = build(cfg)
    data = json.loads((root / "data.json").read_text())
    assert data["runs"] == []
    srv, url = serve_forever_in_thread(root, port=18770)
    try:
        html = urllib.request.urlopen(url, timeout=5).read().decode()
        assert "Alexandria ICML 2027" in html
        resp = urllib.request.urlopen(url + "data.json", timeout=5)
        assert resp.headers["Cache-Control"] == "no-store"
        assert json.loads(resp.read())["profile"] == "debug"
    finally:
        srv.shutdown()
        srv.server_close()
