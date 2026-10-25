"""Stage `ws3-diagnosis` (X1, GPU for the bge-m3 query encoder): why did WS3 agents find the gold paper less often
than plain RAG (debug: ~67% vs 98%)?

On WS3's own index (alexandria.skill: BM25 + bge-m3, RRF; views `summary` and `fulltext`), for every QA-bench item:
  * the gold paper's rank for the QUESTION itself (what arm R searches with, at k = 20 passages);
  * the gold paper's rank for every search QUERY the agent actually issued (from the episode traces), against the `k`
    it asked for -> was the gold paper reachable with the agent's own query and budget?
Rows: item x arm x source (question / agent query i) with the view, k, and gold rank (None = not in the top 100).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..runtime import Context, log, stage
from ..skill.retrieval import BM25
from .common import load_encoder, rrf, to_papers

VIEW = {"S": "summary", "S-OA": "summary", "F": "fulltext", "R": "both"}


def _agent_queries(trace) -> list[dict]:
    try:
        steps = json.loads(trace) if isinstance(trace, str) else list(trace or [])
    except json.JSONDecodeError:
        return []
    return [{"query": s["arguments"].get("query", ""), "k": int(s["arguments"].get("k", 5) or 5)}
            for s in steps if isinstance(s, dict) and s.get("action") in ("search", "search_fulltext")
            and isinstance(s.get("arguments"), dict) and s["arguments"].get("query")]


@stage("x1", "ws3-diagnosis", resources="gpu", config_keys=("x1.ws3_diagnosis",),
       help="WS3 failure analysis: gold-paper rank for the question vs the agents' own queries and k, on WS3's index")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x1.ws3_diagnosis")
    idx = ctx.cfg.stage_dir("ws3", "index")
    items_p = ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet"
    if not (idx / "units.parquet").exists() or not items_p.exists():
        return ctx.skip("X1 WS3 diagnosis", "needs the WS3 index and benchmarks")
    units = pd.read_parquet(idx / "units.parquet", columns=["kind", "paper_key", "has_text"])
    kinds = {k: np.flatnonzero(units["kind"].to_numpy() == k) for k in ("summary", "fulltext")}
    bm = {k: BM25.load(idx / f"bm25_{k}") for k in kinds}
    emb = {k: np.load(idx / f"emb_{k}.npy").astype(np.float32) for k in kinds}
    paper = {k: units["paper_key"].to_numpy()[ix] for k, ix in kinds.items()}
    enc = load_encoder(ctx, cfg.get("embedder", "bge-m3"))
    items = pd.read_parquet(items_p)
    items = items[(items["bench"] == "qabench-open") & items["gold_paper_key"].notna()]
    gold = dict(zip(items["item_id"], items["gold_paper_key"], strict=True))
    q = items["question"].tolist()
    ag = []
    for p in sorted((ctx.cfg.stage_dir("ws3", "agent") / "parts").glob("*.parquet")):
        a = pd.read_parquet(p, columns=["bench", "item_id", "arm", "model", "trace", "gold_retrieved"])
        ag.append(a[(a["bench"] == "qabench-open") & a["arm"].isin(cfg.get("arms", ["S", "F", "S-OA", "R"]))])
    ag = pd.concat(ag, ignore_index=True) if ag else pd.DataFrame()
    searches = [{"item_id": r.item_id, "arm": r.arm, "model": r.model, "i": i, **s}
                for r in ag.itertuples() for i, s in enumerate(_agent_queries(r.trace))]
    texts = q + [s["query"] for s in searches]
    log(f"ws3-diagnosis: {len(q)} questions, {len(searches)} agent queries")
    vecs = enc.encode(texts, 256).astype(np.float32)

    def ranked(text: str, v: np.ndarray, kind: str) -> list[str]:
        out = []
        for k in (("summary", "fulltext") if kind == "both" else (kind,)):
            s_bm = bm[k].scores(text)
            top_b = np.argsort(-s_bm)[:1000]
            s_d = emb[k] @ v
            top_d = np.argsort(-s_d)[:1000]
            out += [to_papers(top_b, s_bm[top_b], paper[k], 100)[0], to_papers(top_d, s_d[top_d], paper[k], 100)[0]]
        return rrf(out, 60, 100)

    def rank_of(r: list[str], g: str) -> int | None:
        return r.index(g) + 1 if g in r else None

    rows = []
    for i, it in enumerate(items.itertuples()):
        for kind in ("summary", "fulltext", "both"):
            rows.append({"item_id": it.item_id, "arm": "-", "model": "-", "source": "question", "view": kind, "k": None,
                         "gold_rank": rank_of(ranked(it.question, vecs[i], kind), it.gold_paper_key)})
    for j, s in enumerate(searches):
        kind = VIEW.get(s["arm"], "summary")
        rows.append({"item_id": s["item_id"], "arm": s["arm"], "model": s["model"], "source": f"agent query {s['i'] + 1}",
                     "view": kind, "k": s["k"],
                     "gold_rank": rank_of(ranked(s["query"], vecs[len(q) + j], kind), gold.get(s["item_id"]))})
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.out / "diagnosis.parquet", index=False)
    ag_k = pd.Series([s["k"] for s in searches])
    if len(ag):
        ag["searched"] = ag["trace"].map(lambda t: bool(_agent_queries(t)))
        ag["n_queries"] = ag["trace"].map(lambda t: len(_agent_queries(t)))
        per_arm = ag.groupby("arm").agg(episodes=("searched", "size"), searched=("searched", "mean"),
                                        queries_per_episode=("n_queries", "mean"),
                                        gold_retrieved=("gold_retrieved", "mean")).reset_index()
    else:
        per_arm = pd.DataFrame()
    per_arm.to_csv(ctx.out / "agent_search_rate.csv", index=False)
    ctx.write_json("agent_k.json", {"k_counts": ag_k.value_counts().sort_index().to_dict() if len(ag_k) else {},
                                    "episodes": len(ag), "episodes_with_search": int(ag["searched"].sum()) if len(ag) else 0})
    return {"rows": len(df), "agent_queries": len(searches)}
