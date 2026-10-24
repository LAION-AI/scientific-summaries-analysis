"""Stage `results` (WS6): the extreme-compression point.

WS2 set: accuracy (SimpleQA judge, CORRECT / all) per arm with 95% paper-clustered bootstrap CIs (same resample for
every arm, so differences are paired), against context tokens per paper: closed book (0), k knowledge tokens (k),
the KU as text (hundreds-thousands). The WS2 arms of the WS2 run for the same LLM family are drawn on the same axis
when present (ws2/results/accuracy.csv). WS3 set: arm EMB on qabench-open with the gold-paper-retrieved rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..figures import plt, save, style
from ..runtime import Context, stage
from ..stats import cluster_bootstrap, paired_difference


def arm_table(g: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, dict]:
    """Per (set, arm): accuracy with paper-clustered bootstrap CI, attempted share, median context tokens."""
    tbl, boot = cluster_bootstrap(g, "correct", "cluster", ["set", "arm"], n_boot=n_boot)
    extra = g.groupby(["set", "arm"]).agg(tokens=("ctx_tokens", "median"), attempted=("attempted", "mean"),
                                          em=("em", "mean"), gold_retrieved=("gold_retrieved", "mean")).reset_index()
    return tbl.merge(extra, on=["set", "arm"]), boot


@stage("ws6", "results", needs=("grade",), config_keys=("ws6.results",),
       help="accuracy vs context tokens: closed book, k knowledge tokens, KU text (WS2); arm EMB (WS3)")
def run(ctx: Context) -> dict:
    n_boot = int(ctx.section("ws6.results").get("n_boot", 2000))
    g = pd.read_parquet(ctx.cfg.stage_dir("ws6", "grade") / "graded.parquet")
    g["gold_retrieved"] = pd.to_numeric(g["gold_retrieved"], errors="coerce")
    acc, boot = arm_table(g, n_boot)
    cmp_rows = []
    for s in acc["set"].unique():
        pt = {(r.set, r.arm): r.mean for r in acc[acc["set"] == s].itertuples()}
        for ref in ("closed_book", "ku_text"):
            for (ss, a) in pt:
                if a != ref and (s, ref) in pt:
                    cmp_rows.append({"set": s, "arm": a, "vs": ref,
                                     **paired_difference(boot, (s, a), (s, ref), pt[(s, a)], pt[(s, ref)])})
    cmp = pd.DataFrame(cmp_rows)
    acc.to_csv(ctx.out / "accuracy.csv", index=False)
    cmp.to_csv(ctx.out / "comparisons.csv", index=False)

    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    w2 = acc[acc["set"] == "ws2"].sort_values("tokens")
    x = np.maximum(w2["tokens"].to_numpy(dtype=float), 0.5)                     # closed book at 0.5 on the log axis
    ax.errorbar(x, w2["mean"], yerr=[w2["mean"] - w2["lo"], w2["hi"] - w2["mean"]], fmt="o", capsize=3, label="WS6 (frozen LLM)")
    for xi, (_, r) in zip(x, w2.iterrows(), strict=True):
        ax.annotate(r["arm"], (xi, r["mean"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ws2p = ctx.cfg.stage_dir("ws2", "results") / "accuracy.csv"
    if ws2p.exists():
        ref = pd.read_csv(ws2p)
        ref = ref.groupby("arm").agg(mean=("mean", "mean"), tokens=("median_tokens", "median")).reset_index()
        ax.scatter(np.maximum(ref["tokens"], 0.5), ref["mean"], marker="s", alpha=.5, label="WS2 arms (7-8B answerers, mean)")
    ax.set_xscale("log")
    ax.set_xlabel("context tokens per paper (log; closed book at 0.5)")
    ax.set_ylabel("accuracy (judge CORRECT)")
    ax.legend(fontsize=7)
    figs = save(fig, ctx.out / "figures", "ws6_accuracy_vs_tokens")

    def pick(s, a):
        r = acc[(acc["set"] == s) & (acc["arm"] == a)]
        return float(r["mean"].iloc[0]) if len(r) else None
    emb = acc[(acc["set"] == "ws2") & acc["arm"].str.startswith("emb_")].sort_values("tokens")
    best = emb.sort_values("mean", ascending=False).head(1)
    kp = [{"label": "closed book", "value": pick("ws2", "closed_book"), "fmt": "pct1"},
          {"label": f"best embedding arm ({best['arm'].iloc[0]})" if len(best) else "embedding arm",
           "value": float(best["mean"].iloc[0]) if len(best) else None, "fmt": "pct1",
           "hint": f"{int(best['tokens'].iloc[0])} tokens per paper" if len(best) else ""},
          {"label": "KU as text", "value": pick("ws2", "ku_text"), "fmt": "pct1",
           "hint": f"{int(acc.loc[(acc['set'] == 'ws2') & (acc['arm'] == 'ku_text'), 'tokens'].iloc[0])} tokens per paper"
           if pick("ws2", "ku_text") is not None else ""}]
    w3 = acc[acc["set"] == "ws3"]
    if len(w3):
        kp.append({"label": "WS3 arm EMB: gold paper retrieved", "value": float(w3["gold_retrieved"].mean()), "fmt": "pct1"})
    cols = [{"key": "set", "label": "set"}, {"key": "arm", "label": "arm"}, {"key": "tokens", "label": "tokens / paper", "fmt": "int"},
            {"key": "mean", "label": "accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
            {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "attempted", "label": "attempted", "fmt": "pct1"},
            {"key": "em", "label": "exact match", "fmt": "pct1"}, {"key": "gold_retrieved", "label": "gold retrieved", "fmt": "pct1"},
            {"key": "n", "label": "questions", "fmt": "int"}]
    ctx.publish("WS6 results: embeddings as input", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "image", "title": "Accuracy vs context tokens per paper (WS2 questions, gold paper given)",
         "src": figs["png"], "alt": "accuracy vs tokens with the knowledge-token points"},
        {"type": "table", "title": "Accuracy per arm (95% paper-clustered bootstrap CI)", "rows": acc.to_dict("records"), "columns": cols},
        {"type": "table", "title": "Paired differences vs closed book and vs KU text", "rows": cmp.to_dict("records"),
         "columns": [{"key": "set", "label": "set"}, {"key": "arm", "label": "arm"}, {"key": "vs", "label": "vs"},
                     {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "p", "label": "p", "fmt": "float3"}]},
    ], summary="; ".join(f"{r.set}/{r.arm} {r.mean:.1%}" for r in acc.itertuples()))
    return {"arms": len(acc)}
