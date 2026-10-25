"""Stage `storage` (X6, CPU): what one paper costs to store, ship and index, per representation.

Per WS2 paper: UTF-8 bytes and gzip bytes and tokens (answer-model tokenizer) of the KU as released (JSON record of
the 18 summary fields), the clean KU rendering (what the answerer reads), and the full source text. Index cost per
paper from the built indices: WS3 (BM25 + bge-m3 per unit kind: summary vs fulltext) and X1 (per representation:
fields / claims / whole KU vs full-text chunks; flat fp32 embeddings and the released IVF-PQ48 file), with X1's
measured query latency per representation.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pandas as pd

from ..data import fetch_records
from ..runtime import Context, log, stage
from ..ws1.faith.common import source_text
from .common import stage_dir


def text_cost(tok, text: str) -> dict:
    b = (text or "").encode("utf-8")
    return {"bytes": len(b), "gzip_bytes": len(gzip.compress(b, 6)) if b else 0,
            "tokens": len(tok(text or "", add_special_tokens=False)["input_ids"])}


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def ws3_index_cost(d: Path) -> list[dict]:
    if not (d / "units.parquet").exists():
        return []
    u = pd.read_parquet(d / "units.parquet", columns=["kind", "paper_key"])
    rows = []
    for kind, g in u.groupby("kind"):
        files = [d / f"bm25_{kind}.npz", d / f"bm25_{kind}.vocab.json", d / f"emb_{kind}.npy"]
        n = g["paper_key"].nunique()
        rows.append({"index": "ws3", "rep": kind, "units": len(g), "papers": n,
                     "bytes": sum(_size(f) for f in files), "bytes_per_paper": sum(_size(f) for f in files) / max(n, 1),
                     "units_per_paper": len(g) / max(n, 1)})
    return rows


def x1_index_cost(d: Path, corpus: Path) -> list[dict]:
    """Per X1 representation: index bytes per paper that has units of it (flat fp32 + BM25, and PQ48 + BM25)."""
    if not (d / "index.json").exists():
        return []
    reps = {r["rep"]: r for r in json.loads((d / "index.json").read_text()).get("reps", [])}
    cp = pd.read_parquet(corpus / "papers.parquet") if (corpus / "papers.parquet").exists() else pd.DataFrame()
    rows = []
    for rep, info in reps.items():
        n = int((cp[f"n_{rep}"] > 0).sum()) if f"n_{rep}" in cp else 0
        bm = _size(d / f"bm25_{rep}")
        flat, pq = _size(d / f"emb_{rep}.npy") + bm, _size(d / f"pq48_{rep}.faiss") + bm
        rows.append({"index": "x1", "rep": rep, "units": info.get("units"), "papers": n, "bytes": flat,
                     "bytes_per_paper": flat / n if n else None, "pq_bytes_per_paper": pq / n if n else None,
                     "units_per_paper": info.get("units", 0) / n if n else None,
                     "encode_s_per_paper": info.get("encode_s", 0) / n if n else None})
    return rows


def x1_latency(d: Path) -> list[dict]:
    p = d / "dashboard.json"
    if not p.exists():
        return []
    for panel in json.loads(p.read_text()).get("panels", []):
        if panel.get("type") == "table" and str(panel.get("title", "")).startswith("Latency"):
            return panel.get("rows", [])
    return []


@stage("x6", "storage", config_keys=("x6.models",),
       help="bytes / gzip bytes / tokens per paper (KU JSON, clean KU, full text) and index size + query latency")
def run(ctx: Context) -> dict:
    from transformers import AutoTokenizer

    from ..llm import model_spec
    os.environ.setdefault("HF_HOME", str(ctx.cfg.site.hf_home))
    spec = model_spec(ctx.cfg, ctx.section("x6").get("models", ["qwen2.5-7b"])[0])
    tok = AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision"))
    papers = pd.read_parquet(stage_dir(ctx, "ws2", "papers") / "papers.parquet")
    kus = pd.read_parquet(stage_dir(ctx, "ws2", "ku") / "kus.parquet").set_index("paper_key")
    cx = pd.read_parquet(stage_dir(ctx, "ws2", "contexts") / "contexts.parquet")
    clean = cx[cx["arm"] == "ku_clean"].set_index("paper_key")["context"]
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)), ["paper_id", "text_raw",
                                                                                           "text_sanitized"])
    rows = []
    for p in papers.itertuples():
        if p.paper_key not in kus.index:
            continue
        src = source_text(recs.get((p.file, int(p.row))) or {})
        reps = {"ku_json": kus.at[p.paper_key, "ku_raw"], "ku_clean": clean.get(p.paper_key, ""), "full_text": src}
        for rep, text in reps.items():
            rows.append({"paper_key": p.paper_key, "rep": rep, **text_cost(tok, str(text or ""))})
    per = pd.DataFrame(rows)
    per.to_parquet(ctx.out / "per_paper.parquet", index=False)
    has_src = set(per[(per["rep"] == "full_text") & (per["bytes"] > 0)]["paper_key"])
    sub = per[per["paper_key"].isin(has_src)]
    tbl = sub.groupby("rep").agg(papers=("paper_key", "nunique"), bytes=("bytes", "median"),
                                 gzip_bytes=("gzip_bytes", "median"), tokens=("tokens", "median"),
                                 mean_tokens=("tokens", "mean")).reset_index()
    ft = sub[sub["rep"] == "full_text"].set_index("paper_key")
    ratios = []
    for rep in ("ku_json", "ku_clean"):
        r = sub[sub["rep"] == rep].set_index("paper_key")
        ratios.append({"rep": rep, "token_ratio_vs_full": float((r["tokens"] / ft["tokens"].reindex(r.index)).median()),
                       "byte_ratio_vs_full": float((r["bytes"] / ft["bytes"].reindex(r.index)).median())})
    idx = ws3_index_cost(stage_dir(ctx, "ws3", "index")) + \
        x1_index_cost(stage_dir(ctx, "x1", "index"), stage_dir(ctx, "x1", "corpus"))
    lat = x1_latency(stage_dir(ctx, "x1", "results"))
    tbl.to_csv(ctx.out / "per_paper_summary.csv", index=False)
    pd.DataFrame(idx).to_csv(ctx.out / "index_cost.csv", index=False)
    ctx.write_json("storage.json", {"per_paper": tbl.to_dict("records"), "ratios": ratios, "index": idx,
                                    "x1_latency": lat, "tokenizer": spec["name"], "papers_with_source": len(has_src)})
    log(f"storage: {len(has_src)} papers with source text")
    ctx.publish("X6 storage and index cost", [
        {"type": "table", "title": "Per paper (median; papers with source text)",
         "columns": [{"key": "rep", "label": "representation"}, {"key": "papers", "label": "papers", "fmt": "int"},
                     {"key": "bytes", "label": "bytes", "fmt": "int"}, {"key": "gzip_bytes", "label": "gzip bytes", "fmt": "int"},
                     {"key": "tokens", "label": "tokens", "fmt": "int"}], "rows": tbl.to_dict("records")},
        {"type": "table", "title": "Index size per paper",
         "columns": [{"key": "index", "label": "index"}, {"key": "rep", "label": "units"},
                     {"key": "units", "label": "units", "fmt": "int"},
                     {"key": "bytes_per_paper", "label": "bytes / paper (flat + BM25)", "fmt": "int"},
                     {"key": "pq_bytes_per_paper", "label": "bytes / paper (PQ48 + BM25)", "fmt": "int"}], "rows": idx}],
        summary=f"{len(has_src)} papers")
    return {"papers": len(has_src), "ratios": ratios}
