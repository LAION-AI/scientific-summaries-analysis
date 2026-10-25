"""Stage `results` (X4): what does field-level / cross-paper access buy?

Accuracy (SimpleQA judge) per model x arm x item type, with 95% bootstrap CIs over items (paired across arms);
paired differences of every arm vs full-text RAG (reference_arm) and vs closed book with Holm correction; the
gold-paper-retrieved rate (did the answering paper reach the model?), cross-paper group recall, retrieved tokens and
GPU-seconds per item; and how often the field router picked the item's field.
"""
from __future__ import annotations

import pandas as pd

from ..figures import plt, save, style
from ..runtime import Context, stage
from ..stats import cluster_bootstrap, holm, paired_difference
from .common import TO_SKILL_FIELD


@stage("x4", "results", needs=("grade",), config_keys=("x4.results",),
       help="accuracy per arm x item type with paired CIs (vs full-text RAG and closed book), gold-paper retrieval, cost")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x4.results")
    ref, n_boot = cfg.get("reference_arm", "R-fulltext"), int(cfg.get("n_boot", 2000))
    g = pd.read_parquet(ctx.cfg.stage_dir("x4", "grade") / "graded.parquet")
    arms = [a for a in ctx.section("x4.answer").get("arms", []) if a in set(g["arm"])]
    acc_rows, cmp_rows = [], []
    a0 = g[g["arm"] == "A0"].groupby("item_id")["correct"].max()             # paper-required: no model answers closed book
    req = g[g["item_id"].isin(a0.index[a0 == 0])] if len(a0) else g
    for typ, d in [("all", g), *g.groupby("type"), ("paper-required", req)]:
        tbl, boot = cluster_bootstrap(d, "correct", "item_id", ["model", "arm"], n_boot=n_boot)
        cost = d.groupby(["model", "arm"]).agg(gold=("gold_retrieved", "mean"), group=("group_recall", "mean"),
                                               tokens=("retrieved_tokens", "mean"), calls=("tool_calls", "mean"),
                                               gpu_s=("gpu_seconds", "mean")).reset_index()
        tbl = tbl.merge(cost, on=["model", "arm"]).assign(type=typ)
        acc_rows.append(tbl)
        pt = {(r.model, r.arm): r.mean for r in tbl.itertuples()}
        for (m, a) in pt:
            for base in (ref, "A0"):
                if a != base and (m, base) in pt:
                    cmp_rows.append({"type": typ, "model": m, "arm": a, "vs": base,
                                     **paired_difference(boot, (m, a), (m, base), pt[(m, a)], pt[(m, base)])})
    acc = pd.concat(acc_rows, ignore_index=True)
    cmp = pd.DataFrame(cmp_rows)
    if len(cmp):
        cmp["p_holm"] = holm(cmp["p"].tolist())
    acc.to_csv(ctx.out / "accuracy.csv", index=False)
    cmp.to_csv(ctx.out / "comparisons.csv", index=False)
    rf = g[g["arm"] == "R-field"].drop_duplicates("item_id")
    route_ok = float((rf["routed_field"] == rf["field"].map(TO_SKILL_FIELD)).mean()) if len(rf) else None

    style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, typ in zip(axes, ("single", "cross"), strict=True):
        t = acc[acc["type"] == typ].groupby("arm")["mean"].mean().reindex(arms)
        ax.bar(range(len(t)), t.values * 100, color="#4C72B0")
        ax.set_xticks(range(len(t)), t.index, rotation=45, fontsize=7)
        ax.set_title({"single": "field-targeted (one paper)", "cross": "cross-paper"}[typ], fontsize=9)
    axes[0].set_ylabel("accuracy (%), mean over models")
    figs = save(fig, ctx.out / "figures", "x4_accuracy_by_arm")

    allm = acc[acc["type"] == "all"].groupby("arm")[["mean", "gold"]].mean()
    best = allm["mean"].idxmax() if len(allm) else None
    d_field = cmp[(cmp["type"] == "cross") & (cmp["arm"] == "R-field") & (cmp["vs"] == ref)]["diff"].mean() if len(cmp) else None
    kp = [{"label": "items", "value": int(g["item_id"].nunique()), "fmt": "int"},
          {"label": "paper-required items", "value": int(req["item_id"].nunique()), "fmt": "int",
           "hint": "no model answers them closed book"},
          {"label": "best arm (all items)", "value": best, "hint": f"{allm.loc[best, 'mean']:.1%}" if best else ""},
          {"label": "R-field vs full-text RAG, cross-paper", "value": d_field, "fmt": "pct1",
           "hint": "Δ accuracy, mean over models"},
          {"label": "field router picks the item's field", "value": route_ok, "fmt": "pct1"}]
    cols = [{"key": "type", "label": "items"}, {"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
            {"key": "mean", "label": "accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
            {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "gold", "label": "gold paper retrieved", "fmt": "pct1"},
            {"key": "group", "label": "group recall", "fmt": "pct1"}, {"key": "tokens", "label": "retrieved tokens", "fmt": "int"},
            {"key": "calls", "label": "tool calls", "fmt": "float"}, {"key": "gpu_s", "label": "GPU s / item", "fmt": "float3"}]
    ctx.publish("X4 results: field-level and cross-paper questions", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "image", "title": "Accuracy by arm and item type", "src": figs["png"], "alt": "accuracy per arm"},
        {"type": "table", "title": "Accuracy, retrieval and cost per item type, model and arm", "columns": cols,
         "rows": acc.sort_values(["type", "model", "arm"]).to_dict("records")},
        {"type": "table", "title": f"Paired differences vs {ref} and vs closed book (Holm over all rows)",
         "columns": [{"key": "type", "label": "items"}, {"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "vs", "label": "vs"}, {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"},
                     {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"},
                     {"key": "p_holm", "label": "p (Holm)", "fmt": "float3"}],
         "rows": cmp.to_dict("records") if len(cmp) else []},
    ], summary=f"{g['item_id'].nunique()} items; best arm {best}")
    return {"items": int(g["item_id"].nunique()), "best_arm": best}
