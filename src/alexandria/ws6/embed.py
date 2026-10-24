"""Stage `embed` (WS6, GPU): full-precision gte-modernbert embeddings (fp32, L2-normalised, CLS pooling).

The dataset's FAISS indices store PQ48 codes (lossy product quantisation) and no raw vectors exist locally
(search/faiss holds only metadata), so the KUs are re-embedded with the same embedder at full precision. Sets:
  projector   KU prose of the projector train / dev papers
  ws2         KU prose of the WS2 evaluation papers (the extreme-compression point: embedding instead of text)
  ws3corpus   per paper of the WS3 index corpus: its KU summary units joined (retrieval + knowledge tokens)
  ws3query    the qabench-open questions (retrieval queries)
Each set: <set>.npy (n x 768, float32) + <set>_keys.parquet (row order).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..runtime import Context, log, stage
from ..ws4.common import clean_ku, render
from .common import load_encoder, stage_table


def texts_ws2(ctx: Context) -> pd.DataFrame:
    kus = pd.read_parquet(ctx.cfg.stage_dir("ws2", "ku") / "kus.parquet", columns=["paper_key", "ku_fixed"])
    return pd.DataFrame({"key": kus["paper_key"], "text": [render(clean_ku(json.loads(k)), "prose") for k in kus["ku_fixed"]]})


def texts_ws3corpus(ctx: Context) -> pd.DataFrame:
    u = pd.read_parquet(ctx.cfg.stage_dir("ws3", "index") / "units.parquet", columns=["paper_key", "kind", "field", "title", "text"])
    u = u[u["kind"] == "summary"]
    rows = [{"key": pk, "text": f"# {g['title'].iloc[0]}\n\n" + "\n\n".join(f"## {f}\n{t}" for f, t in zip(g["field"], g["text"], strict=True))}
            for pk, g in u.groupby("paper_key", sort=True)]
    return pd.DataFrame(rows)


def texts_ws3query(ctx: Context) -> pd.DataFrame:
    it = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet", columns=["bench", "item_id", "question"])
    it = it[it["bench"] == "qabench-open"]
    return pd.DataFrame({"key": it["item_id"], "text": it["question"]})


@stage("ws6", "embed", needs=("papers",), resources="gpu", config_keys=("ws6.embed",),
       help="full-precision gte-modernbert embeddings: projector papers, WS2 eval papers, WS3 corpus, WS3 queries")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws6.embed")
    import torch
    enc = load_encoder(ctx, "cuda" if torch.cuda.is_available() else "cpu")
    papers = stage_table(ctx, "papers", "papers.parquet")
    sets = {"projector": pd.DataFrame({"key": papers["paper_key"],
                                       "text": [render(json.loads(k), "prose") for k in papers["ku"]]}),
            "ws2": texts_ws2(ctx), "ws3corpus": texts_ws3corpus(ctx), "ws3query": texts_ws3query(ctx)}
    stats = []
    for name, df in sets.items():
        log(f"embed: {name}: {len(df)} texts")
        v = enc.encode(df["text"].tolist(), int(cfg.get("max_length", 8192)))
        np.save(ctx.out / f"{name}.npy", v.astype(np.float32))
        df[["key"]].to_parquet(ctx.out / f"{name}_keys.parquet", index=False)
        ntok = [len(x) for x in enc.tok(df["text"].tolist(), add_special_tokens=False)["input_ids"]]
        stats.append({"set": name, "texts": len(df), "dim": int(v.shape[1]), "median_tokens": float(np.median(ntok)),
                      "truncated": int(sum(n > int(cfg.get("max_length", 8192)) for n in ntok)),
                      "norm_mean": float(np.linalg.norm(v, axis=1).mean())})
    ctx.publish("WS6 full-precision KU embeddings", [
        {"type": "kpis", "items": [{"label": f"{s['set']} vectors", "value": s["texts"], "fmt": "int"} for s in stats]
         + [{"label": "dimension", "value": stats[0]["dim"], "fmt": "int", "hint": "gte-modernbert, CLS pooling, fp32"}]},
        {"type": "markdown", "text": "The dataset's FAISS indices store PQ48 codes (lossy) and no raw vectors exist "
                                     "locally (search/faiss holds only metadata), so the KUs are re-embedded here at "
                                     "full precision with the same embedder (Alibaba-NLP/gte-modernbert-base)."},
        {"type": "table", "title": "Embedded sets", "rows": stats, "columns": [
            {"key": "set", "label": "set"}, {"key": "texts", "label": "texts", "fmt": "int"},
            {"key": "median_tokens", "label": "median tokens", "fmt": "int"},
            {"key": "truncated", "label": "truncated (> max length)", "fmt": "int"}]},
    ], summary=", ".join(f"{s['set']} {s['texts']}" for s in stats))
    return {s["set"]: s["texts"] for s in stats}
