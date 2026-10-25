"""Stage `results` (X1): gold-paper retrieval quality, PQ48 loss, latency, and the WS3 diagnosis.

Per benchmark x representation x retriever: Recall@{1,5,10,20,100} (share of the gold papers in the top k), MRR,
nDCG@10, each with a 95% bootstrap CI over queries; evaluated on queries whose gold paper is in the release only.
PQ48: the drop vs exact flat search in gold-paper Recall@10 / MRR, and the unit-level agreement with the exact top-k.
Timing: query encoding and per-retriever search latency (p50 / p95, single query) and batch QPS.
WS3 diagnosis: share of items whose gold paper is in the top k for the question vs for the agents' own queries at
the k they asked for, per arm.
"""
from __future__ import annotations

import json

import pandas as pd

from ..figures import plt, save, style
from ..runtime import Context, stage
from .common import bootstrap_ci, mrr, ndcg_at, recall_at, stage_table


def col(key: str, label: str, fmt: str | None = "pct1") -> dict:
    return {"key": key, "label": label, "fmt": fmt}


@stage("x1", "results", needs=("search",), config_keys=("x1.results",),
       help="Recall@k / MRR / nDCG@10 with CIs per benchmark x representation x retriever; PQ48 loss; latency; WS3 diagnosis")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x1.results")
    ks, nb = [int(k) for k in cfg.get("ks", [1, 5, 10, 20, 100])], int(cfg.get("n_boot", 2000))
    r = stage_table(ctx, "search", "rankings.parquet")
    q = stage_table(ctx, "benchmarks", "queries.parquet")[["qid", "gold_keys"]]
    gold = {qid: set(g) for qid, g in zip(q["qid"], q["gold_keys"], strict=True)}
    it = ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet"
    if it.exists():
        w = pd.read_parquet(it, columns=["item_id", "gold_paper_key"]).dropna()
        gold.update({i: {g} for i, g in zip(w["item_id"], w["gold_paper_key"], strict=True)})
    per = []
    for row in r.itertuples():
        g = gold.get(row.qid, set())
        ranked = list(row.ranked)
        per.append({"bench": row.bench, "rep": row.rep, "retriever": row.retriever, "qid": row.qid,
                    **{f"R@{k}": recall_at(ranked, g, k) for k in ks}, "MRR": mrr(ranked, g), "nDCG@10": ndcg_at(ranked, g, 10)})
    per = pd.DataFrame(per)
    per.to_parquet(ctx.out / "per_query.parquet", index=False)
    rows = []
    for (b, rep, ret), g in per.groupby(["bench", "rep", "retriever"]):
        out = {"bench": b, "rep": rep, "retriever": ret, "queries": len(g)}
        for m in [f"R@{k}" for k in ks] + ["MRR", "nDCG@10"]:
            mean, lo, hi = bootstrap_ci(g[m].to_numpy(), nb)
            out[m], out[f"{m}_lo"], out[f"{m}_hi"] = mean, lo, hi
        rows.append(out)
    tab = pd.DataFrame(rows)
    tab.to_csv(ctx.out / "retrieval.csv", index=False)
    flat = tab[tab["retriever"] == "dense"].set_index(["bench", "rep"])
    pq = tab[tab["retriever"] == "dense_pq48"].set_index(["bench", "rep"])
    loss = (pq[["R@10", "MRR"]] - flat[["R@10", "MRR"]]).reset_index().rename(columns={"R@10": "dR@10", "MRR": "dMRR"})
    agree = pd.read_csv(ctx.cfg.stage_dir("x1", "search") / "pq_vs_flat.csv")
    agree = agree.pivot_table(index="rep", columns="k", values="unit_recall_vs_flat").reset_index()
    agree.columns = ["rep"] + [f"unit_recall@{c}" for c in agree.columns[1:]]
    loss = loss.merge(agree, on="rep", how="left")
    loss.to_csv(ctx.out / "pq48_loss.csv", index=False)
    timing = pd.read_csv(ctx.cfg.stage_dir("x1", "search") / "timing.csv")
    hw = json.loads((ctx.cfg.stage_dir("x1", "search") / "hardware.json").read_text())

    # WS3 diagnosis
    diag_p = ctx.cfg.stage_dir("x1", "ws3-diagnosis") / "diagnosis.parquet"
    diag_rows = []
    if diag_p.exists():
        d = pd.read_parquet(diag_p)
        qd = d[d["source"] == "question"]
        for view, g in qd.groupby("view"):
            for k in (3, 5, 10, 20):
                diag_rows.append({"who": "question", "view": view, "k": k,
                                  "gold_in_top_k": float((g["gold_rank"].fillna(1e9) <= k).mean()), "n": len(g)})
        ad = d[d["source"] != "question"].copy()
        if len(ad):
            ad["hit"] = ad["gold_rank"].fillna(1e9) <= ad["k"]
            first = ad[ad["source"] == "agent query 1"]
            for arm, g in first.groupby("arm"):
                diag_rows.append({"who": f"agent {arm}, 1st query", "view": g["view"].iloc[0], "k": float(g["k"].median()),
                                  "gold_in_top_k": float(g["hit"].mean()), "n": len(g)})
            anyq = ad.groupby(["arm", "model", "item_id"])["hit"].any().reset_index()
            for arm, g in anyq.groupby("arm"):
                diag_rows.append({"who": f"agent {arm}, any query", "view": "-", "k": None,
                                  "gold_in_top_k": float(g["hit"].mean()), "n": len(g)})
            for arm, g in ad[ad["source"] == "agent query 1"].groupby("arm"):
                diag_rows.append({"who": f"agent {arm}, 1st query at k=20", "view": g["view"].iloc[0], "k": 20,
                                  "gold_in_top_k": float((g["gold_rank"].fillna(1e9) <= 20).mean()), "n": len(g)})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(ctx.out / "ws3_diagnosis.csv", index=False)
    kd = ctx.cfg.stage_dir("x1", "ws3-diagnosis") / "agent_k.json"
    agent_k = json.loads(kd.read_text()) if kd.exists() else {}

    style()
    main = tab[tab["bench"] != "ws3-qabench"]
    benches = sorted(main["bench"].unique())
    fig, axes = plt.subplots(1, max(len(benches), 1), figsize=(4.2 * max(len(benches), 1), 3.2), squeeze=False)
    for ax, b in zip(axes[0], benches, strict=False):
        t = main[main["bench"] == b].pivot_table(index="rep", columns="retriever", values="R@10")
        t.plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f"{b}: gold Recall@10", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    figs = save(fig, ctx.out / "figures", "x1_recall10")

    best = main.sort_values("R@10", ascending=False).groupby("bench").head(1)
    kp = [{"label": f"{row['bench']}: best Recall@10", "value": float(row["R@10"]), "fmt": "pct1",
           "hint": f"{row['retriever']} over {row['rep']}"} for _, row in best.iterrows()]
    if len(loss):
        kp.append({"label": "PQ48 vs exact: mean Recall@10 change", "value": float(loss["dR@10"].mean()), "fmt": "pct1"})
    qd = diag[(diag["who"] == "question") & (diag["view"] == "summary") & (diag["k"] == 3)] if len(diag) else diag
    ad = diag[diag["who"] == "agent S, 1st query"] if len(diag) else diag
    if len(qd) and len(ad):
        kp.append({"label": "WS3: gold in top-k, question vs agent query", "value":
                   f"{qd['gold_in_top_k'].iloc[0]:.0%} vs {ad['gold_in_top_k'].iloc[0]:.0%}", "hint": "summary view, k = 3"})
    panels = [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "image", "title": "Gold-paper Recall@10 per representation and retriever", "src": figs["png"], "alt": "recall@10 bars"},
        {"type": "table", "title": "Gold-paper retrieval (queries with a linked gold paper; 95% bootstrap CI)",
         "columns": [col("bench", "benchmark", None), col("rep", "representation", None), col("retriever", "retriever", None),
                     col("queries", "queries", "int"), col("R@1", "R@1"), col("R@5", "R@5"), col("R@10", "R@10"),
                     col("R@10_lo", "lo"), col("R@10_hi", "hi"), col("R@100", "R@100"), col("MRR", "MRR", "float3"),
                     col("nDCG@10", "nDCG@10", "float3")],
         "rows": tab.sort_values(["bench", "rep", "retriever"]).to_dict("records")},
        {"type": "table", "title": "IVF-PQ48 (as released) vs exact flat search",
         "hint": "dR@10 / dMRR: PQ48 minus flat on gold-paper metrics; unit recall: share of the exact top-k units PQ48 returns",
         "columns": [col("bench", "benchmark", None), col("rep", "representation", None), col("dR@10", "Δ R@10"),
                     col("dMRR", "Δ MRR", "float3"), col("unit_recall@10", "unit recall@10"), col("unit_recall@100", "unit recall@100")],
         "rows": loss.to_dict("records")},
        {"type": "table", "title": f"Latency and throughput (CPU search: {hw.get('cpus')} cores; encoder GPU: {hw.get('gpu')})",
         "columns": [col("rep", "representation", None), col("retriever", "retriever / step", None), col("units", "units", "int"),
                     col("p50_ms", "p50 ms", "float"), col("p95_ms", "p95 ms", "float"), col("qps_batch", "QPS (batch)", "float")],
         "rows": timing.to_dict("records")},
    ]
    rate_p = ctx.cfg.stage_dir("x1", "ws3-diagnosis") / "agent_search_rate.csv"
    if rate_p.exists() and rate_p.stat().st_size > 1:
        rate = pd.read_csv(rate_p)
        panels.append({"type": "table", "title": "WS3 agents: how often did they search at all?",
                       "hint": "episodes on the QA bench; gold retrieved = WS3's gold-paper-retrieved rate",
                       "columns": [col("arm", "arm", None), col("episodes", "episodes", "int"), col("searched", "searched"),
                                   col("queries_per_episode", "queries / episode", "float"), col("gold_retrieved", "gold retrieved")],
                       "rows": rate.to_dict("records")})
    if len(diag):
        panels.append({"type": "table", "title": "WS3 diagnosis: is the gold paper in the top k? (WS3's own BM25 + bge-m3 index)",
                       "hint": "question = what plain RAG searches with; agent rows use each agent's own query and the k it asked "
                               f"for; k requested by agents: {agent_k.get('k_counts', {})}",
                       "columns": [col("who", "query", None), col("view", "view", None), col("k", "k", "float"),
                                   col("gold_in_top_k", "gold in top k"), col("n", "n", "int")],
                       "rows": diag.to_dict("records")})
    ctx.publish("X1 results: how well do the indices find the gold paper?", panels,
                summary="; ".join(f"{r['bench']} R@10 {r['R@10']:.0%} ({r['retriever']}/{r['rep']})" for _, r in best.iterrows()))
    return {"rows": len(tab)}
