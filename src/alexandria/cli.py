"""`alexandria <command>`: one entry point for every workstream stage.

    alexandria list                                   # all stages with what they need and where they run
    alexandria datasets                               # the datasets configured in configs/datasets/
    alexandria sample                                 # make the tiny debug subset (once)
    alexandria ws1 features --profile debug           # run one stage
    alexandria ws1 features --profile full --shard 3/16    # one worker of a map stage (SLURM arrays set this)
    alexandria ws1 describe --set ws1.describe.top_fields=60
    alexandria dashboard build --profile debug        # collect every stage's panels into the dashboard
    alexandria dashboard serve --profile debug        # http://localhost:8770 (VS Code forwards the port)
    alexandria fetch-models                           # login node: download models used by GPU stages
    alexandria fetch-benchmarks                       # login node: PubMedQA, LitQA2, SciFact (+ gated with HF_TOKEN)
    alexandria skill mcp --profile debug              # the Alexandria skill as an MCP server (stdio)
    alexandria submit ws1 quality --profile full --with-deps   # SLURM jobs, chained by what each stage needs

Each stage reads only files earlier stages wrote, so any stage can be re-run on its own.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from .config import PROFILES, available_datasets, load_config
from .runtime import STAGES, Context, run_stage

WORKSTREAMS = ("ws1", "ws2", "ws3")


def _load_stages() -> None:
    for ws in WORKSTREAMS:
        importlib.import_module(f"alexandria.{ws}")


def _shard(text: str | None) -> tuple[int, int]:
    if text:
        i, n = text.split("/")
        return int(i) - 1, int(n)                 # --shard 3/16 is the 3rd of 16 (1-based on the command line)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        lo = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        return int(os.environ["SLURM_ARRAY_TASK_ID"]) - lo, int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return 0, 1


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", choices=PROFILES, default=os.environ.get("ALEXANDRIA_PROFILE", "debug"))
    p.add_argument("--dataset", default=None, metavar="NAME",
                   help=f"configs/datasets/<NAME>.yaml (available: {', '.join(available_datasets())}; "
                        "default: site.default_dataset)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. ws1.checks.min_title_similarity=70 (repeatable)")


def cmd_list() -> int:
    _load_stages()
    for (ws, name), s in STAGES.items():
        needs = ", ".join(s.needs) or "-"
        print(f"{ws:4} {name:18} {s.kind:6} {s.resources:3} env={s.env:4} needs: {needs:32} {s.help}")
    return 0


def cmd_stage(ws: str, argv: list[str]) -> int:
    _load_stages()
    names = [n for (w, n) in STAGES if w == ws]
    p = argparse.ArgumentParser(prog=f"alexandria {ws}", description=f"run one {ws} stage")
    p.add_argument("stage", choices=names)
    _common(p)
    p.add_argument("--shard", help="i/n: run worker i of n (map stages; default from SLURM_ARRAY_TASK_ID)")
    p.add_argument("--workers", type=int, help="processes for map stages (default: SLURM_CPUS_PER_TASK, else 1)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    a = p.parse_args(argv)
    spec = STAGES[(ws, a.stage)]
    cfg = load_config(ws, a.profile, a.overrides, dataset=a.dataset)
    idx, cnt = _shard(a.shard) if spec.kind == "map" else (0, 1)
    os.environ.setdefault("HF_HOME", str(cfg.site.hf_home))
    return run_stage(Context(cfg, spec, idx, cnt, a.force, extra={"workers": a.workers}))


def cmd_pipeline(ws: str, argv: list[str]) -> int:
    """Run several stages of one workstream in dependency order in this process (one SLURM allocation)."""
    from .slurm import _order
    _load_stages()
    p = argparse.ArgumentParser(prog=f"alexandria pipeline {ws}")
    p.add_argument("stages", nargs="+")
    _common(p)
    p.add_argument("--workers", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config(ws, a.profile, a.overrides, dataset=a.dataset)
    os.environ.setdefault("HF_HOME", str(cfg.site.hf_home))
    failed = []
    for spec in _order(ws, a.stages, with_deps=False):
        if any(d in failed for d in spec.needs):
            print(f"skip {spec.name}: a stage it needs failed")
            failed.append(spec.name)
            continue
        if run_stage(Context(cfg, spec, 0, 1, a.force, extra={"workers": a.workers})) != 0:
            failed.append(spec.name)
    print("pipeline:", "all done" if not failed else f"failed: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_sample(argv: list[str]) -> int:
    from .data import make_sample
    p = argparse.ArgumentParser(prog="alexandria sample", description="copy a tiny stratified debug subset")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--dataset", default=None, metavar="NAME")
    a = p.parse_args(argv)
    cfg = load_config("ws1", "debug", a.overrides, dataset=a.dataset)
    s = cfg.profile_settings
    out = make_sample(cfg, int(s.get("files_per_group", 2)), int(s.get("rows_per_file", 60)), int(s.get("seed", 0)))
    print(f"debug sample: {out}")
    return 0


def cmd_dashboard(argv: list[str]) -> int:
    from .dashboard import build, serve
    p = argparse.ArgumentParser(prog="alexandria dashboard")
    p.add_argument("action", choices=["build", "serve"])
    _common(p)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--rebuild-every", type=int, default=60, help="serve: rebuild data.json every N s (0 = never)")
    a = p.parse_args(argv)
    _load_stages()
    cfg = load_config("ws1", a.profile, a.overrides, dataset=a.dataset)
    root = build(cfg)
    if a.action == "serve":
        serve(root, a.host, a.port, cfg=cfg, rebuild_every=a.rebuild_every)
    else:
        print(f"dashboard: {root / 'index.html'}")
    return 0


def cmd_fetch_models(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="alexandria fetch-models",
                                description="download the models GPU stages use into site.hf_home (login node)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args(argv)
    cfg = load_config("ws1", "debug", a.overrides)
    os.environ["HF_HOME"] = str(cfg.site.hf_home)
    from huggingface_hub import snapshot_download
    for key in ("grounding", "judge"):
        section = cfg.section("ws1b").get(key, {})
        model = section.get("model")
        if model:
            print(f"== {model} @ {section.get('revision') or 'main'} -> {cfg.site.hf_home}", flush=True)
            snapshot_download(model, revision=section.get("revision"), allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py",
                                                     "tokenizer*", "*.tiktoken"])
    return 0


def cmd_skill_mcp(argv: list[str]) -> int:
    """Serve the Alexandria skill tools over MCP (stdio by default) from the WS3 index of a profile."""
    p = argparse.ArgumentParser(prog="alexandria skill mcp")
    _common(p)
    p.add_argument("--view", default="summary", choices=["summary", "fulltext"])
    p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    p.add_argument("--no-dense", action="store_true", help="BM25 only (no GPU / encoder needed)")
    a = p.parse_args(argv)
    cfg = load_config("ws3", a.profile, a.overrides, dataset=a.dataset)
    os.environ.setdefault("HF_HOME", str(cfg.site.hf_home))
    from .llm import embedder_spec
    from .skill.backend import Backend
    from .skill.mcp_server import build_server
    be = Backend(cfg.stage_dir("ws3", "index"), embedder_spec(cfg, cfg.section("ws3").get("index", {}).get("embedder", "bge-m3")),
                 cfg.stage_dir("ws3", "index") / "mcp_cache.sqlite", dense=not a.no_dense)
    build_server(be, a.view).run(a.transport)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        return cmd_list()
    if cmd == "datasets":
        for name in available_datasets():
            c = load_config("ws1", "debug", dataset=name)
            print(f"{name:28} {c.dataset.title or '-':40} {len(c.data.sources)} source group(s) -> {c.run_root().parent}")
        return 0
    if cmd == "sample":
        return cmd_sample(rest)
    if cmd == "dashboard":
        return cmd_dashboard(rest)
    if cmd == "fetch-models":
        return cmd_fetch_models(rest)
    if cmd == "fetch-benchmarks":
        from .ws3.benchmarks import fetch
        cfg = load_config("ws3", "debug")
        os.environ.setdefault("HF_HOME", str(cfg.site.hf_home))
        res = fetch(cfg, rest or None)
        return 0 if any(v == "ok" for v in res.values()) else 1
    if cmd == "skill" and rest[:1] == ["mcp"]:
        return cmd_skill_mcp(rest[1:])
    if cmd == "pipeline" and rest and rest[0] in WORKSTREAMS:
        return cmd_pipeline(rest[0], rest[1:])
    if cmd == "submit":
        from .slurm import main as submit_main
        return submit_main(rest)
    if cmd in WORKSTREAMS:
        return cmd_stage(cmd, rest)
    print(f"unknown command {cmd!r}\n\n{__doc__.strip()}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
