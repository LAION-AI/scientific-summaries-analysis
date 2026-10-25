"""Stage `index` of X2: incremental update of the WS3 retrieval index with the new papers.

A copy of the WS3 index gets the new papers' units appended (KU fields + claims as `summary` units, full-text chunks
as `fulltext` units; ws3.index.paper_units). Dense vectors are computed for the NEW units only and appended to each
kind's matrix (the backend reads them in unit order within a kind, so [old; new] stays aligned); BM25 is rebuilt per
kind (cheap: term statistics change). A new version hash invalidates tool-response caches.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd

from ..llm import embedder_spec
from ..runtime import Context, log, stage
from ..skill.retrieval import BM25, Encoder
from ..ws3.index import paper_units
from .common import stage_table


def new_units(papers: pd.DataFrame, kus: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    by = dict(zip(kus["paper_key"], kus["ku"], strict=True))
    rows = []
    for p in papers.itertuples():
        ku = json.loads(by.get(p.paper_key) or "{}") or {}
        rec = {k: (json.dumps(v) if k == "claims" and not isinstance(v, str) else v) for k, v in ku.items()}
        rec["text_sanitized"] = p.full                              # source_text() reads the full text from here
        for u in paper_units(rec, cfg):
            rows.append({**u, "paper_key": p.paper_key, "file": None, "row": None, "paper_id": p.arxiv_id,
                         "title": p.title[:200], "has_text": True, "gold": False, "added_by": "x2"})
    return pd.DataFrame(rows)


@stage("x2", "index", needs=("summarize",), resources="gpu", config_keys=("x2.index",),
       help="incremental update: the WS3 index + the new papers' KU units and full-text chunks (dense: new units only)")
def run_index(ctx: Context) -> dict:
    cfg = ctx.section("x2.index")
    src = ctx.cfg.stage_dir("ws3", "index")
    if not (src / "index.json").exists():
        raise FileNotFoundError("the WS3 index is missing: run ws3 index first")
    out = ctx.out
    for f in src.iterdir():                                       # start from the WS3 index as it is
        if f.suffix in (".npy", ".npz", ".json", ".parquet") and f.name not in ("manifest.json", "dashboard.json"):
            shutil.copy2(f, out / f.name)
    old = pd.read_parquet(out / "units.parquet")
    add = new_units(stage_table(ctx, "papers", "papers.parquet"), stage_table(ctx, "summarize", "kus.parquet"), cfg)
    add.insert(0, "uid", range(int(old["uid"].max()) + 1, int(old["uid"].max()) + 1 + len(add)))
    units = pd.concat([old, add], ignore_index=True)
    units.to_parquet(out / "units.parquet", index=False)
    enc = Encoder(embedder_spec(ctx.cfg, cfg.get("embedder", "bge-m3")))
    sizes, t_emb = {}, 0.0
    for kind, g in units.groupby("kind", sort=False):
        BM25.build(g["text"].tolist()).save(out / f"bm25_{kind}")
        new = add[add["kind"] == kind]["text"].tolist()
        t0 = time.time()
        vec = enc.encode(new, 512).astype(np.float16) if new else np.zeros((0, 1024), np.float16)
        t_emb += time.time() - t0
        old_emb = np.load(out / f"emb_{kind}.npy") if (out / f"emb_{kind}.npy").exists() else vec[:0]
        np.save(out / f"emb_{kind}.npy", np.concatenate([old_emb, vec]) if len(old_emb) else vec)
        sizes[kind] = len(g)
        log(f"index: {kind}: {len(g) - len(new)} old + {len(new)} new units")
    meta = json.loads((src / "index.json").read_text())
    version = hashlib.sha256(json.dumps([meta["version"], sizes, len(add)]).encode()).hexdigest()[:12]
    n_new = int(add["paper_key"].nunique())
    ctx.write_json("index.json", {**meta, "version": version, "papers": meta["papers"] + n_new, "units": sizes,
                                  "base_version": meta["version"], "added_papers": n_new})
    ctx.publish("X2 incremental index update", [{"type": "kpis", "items": [
        {"label": "papers added", "value": n_new, "fmt": "int"},
        {"label": "units added", "value": len(add), "fmt": "int"},
        {"label": "papers in the index", "value": meta["papers"] + n_new, "fmt": "int"},
        {"label": "embedding time for the update", "value": t_emb, "fmt": "float", "hint": "seconds, new units only"},
        {"label": "index version", "value": version, "hint": f"base WS3 {meta['version']}"}]}],
        summary=f"+{n_new} papers ({len(add)} units) on top of the WS3 index")
    return {"added_papers": n_new, "added_units": len(add), "version": version}
