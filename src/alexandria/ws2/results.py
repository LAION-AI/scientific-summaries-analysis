"""Stage `results` (WS2): accuracy of every representation arm, with uncertainty, strata and the compression view.

Accuracy = judge CORRECT / all questions (SimpleQA protocol), averaged over seeds per question first. Every interval
is a 95% bootstrap interval that resamples whole papers (questions of one paper are not independent), with the same
resample for every arm and model, so arm differences are paired (stats.cluster_bootstrap / paired_difference).
Differences vs closed_book and vs full_text get Holm-adjusted p-values over all comparisons of the table.

Strata reported:
  * context-dependent: questions the model does NOT answer correctly closed-book (where the representation matters);
  * before/after cleaning: ku_clean - ku_raw on papers WS1 flagged vs not flagged;
  * token-matched view: accuracy against median context tokens per arm (compression), incl. full_text_matched;
  * help / harm flips vs closed_book; question type; domain;
  * lexical bias: accuracy by question-context 5-gram overlap tercile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..figures import plt, save, style
from ..runtime import Context, stage
from ..stats import cluster_bootstrap, holm, paired_difference
from .common import ARMS, load_stage_table


def per_question(g: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds: one row per (model, arm, qid)."""
    keys = ["model", "arm", "qid", "paper_key"]
    return g.groupby(keys, as_index=False).agg(correct=("correct", "mean"), attempted=("attempted", "mean"),
                                               em=("em", "mean"), f1=("f1", "mean"))


def accuracy_table(pq: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, dict]:
    tbl, boot = cluster_bootstrap(pq, "correct", "paper_key", ["model", "arm"], n_boot=n_boot)
    extra = pq.groupby(["model", "arm"]).agg(attempted=("attempted", "mean"), em=("em", "mean"), f1=("f1", "mean"))
    return tbl.merge(extra.reset_index(), on=["model", "arm"]), boot


def comparisons(tbl: pd.DataFrame, boot: dict, refs: list[str]) -> pd.DataFrame:
    point = {(r.model, r.arm): r.mean for r in tbl.itertuples()}
    rows = []
    for (model, arm) in point:
        for ref in refs:
            if arm == ref or (model, ref) not in point:
                continue
            d = paired_difference(boot, (model, arm), (model, ref), point[(model, arm)], point[(model, ref)])
            rows.append({"model": model, "arm": arm, "vs": ref, **d})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["p"].tolist())
    return df


def flips(pq: pd.DataFrame) -> pd.DataFrame:
    cb = pq[pq["arm"] == "closed_book"].set_index(["model", "qid"])["correct"]
    rows = []
    for (model, arm), g in pq[pq["arm"] != "closed_book"].groupby(["model", "arm"]):
        base = cb.reindex(list(zip(g["model"], g["qid"], strict=True))).to_numpy()
        cur = g["correct"].to_numpy()
        rows.append({"model": model, "arm": arm, "helped": float(((base < .5) & (cur >= .5)).mean()),
                     "harmed": float(((base >= .5) & (cur < .5)).mean()), "n": len(g)})
    return pd.DataFrame(rows)


@stage("ws2", "results", needs=("grade", "contexts", "questions", "papers"), config_keys=("ws2.results",),
       help="accuracy per arm and model with paper-clustered bootstrap CIs, paired tests, strata, compression view")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.results")
    n_boot = int(cfg.get("n_boot", 2000))
    g = load_stage_table(ctx, "grade", "graded.parquet")
    qs = load_stage_table(ctx, "questions", "questions.parquet")[["qid", "paper_key", "qtype", "domain", "ws1_flagged"]]
    cx = load_stage_table(ctx, "contexts", "contexts.parquet")[["paper_key", "arm", "tokens"]]
    g = g.merge(qs, on="qid", how="left")
    pq = per_question(g).merge(qs.drop(columns=["paper_key"]), on="qid", how="left")
    arms = [a for a in ARMS if a in set(pq["arm"])]
    models = sorted(pq["model"].unique())

    tbl, boot = accuracy_table(pq, n_boot)
    tok = cx.groupby("arm")["tokens"].median()
    tbl["median_tokens"] = tbl["arm"].map(tok)
    tbl["arm"] = pd.Categorical(tbl["arm"], arms, ordered=True)
    tbl = tbl.sort_values(["model", "arm"])
    tbl["arm"] = tbl["arm"].astype(str)
    comp = comparisons(tbl, boot, ["closed_book", "full_text"])

    # context-dependent stratum: questions this model gets wrong closed-book
    cb_ok = pq[pq["arm"] == "closed_book"].set_index(["model", "qid"])["correct"] >= .5
    pq["closed_book_correct"] = [bool(cb_ok.get((m, q), False)) for m, q in zip(pq["model"], pq["qid"], strict=True)]
    dep = pq[~pq["closed_book_correct"]]
    dep_tbl, dep_boot = cluster_bootstrap(dep, "correct", "paper_key", ["model", "arm"], n_boot=n_boot)
    dep_comp = comparisons(dep_tbl, dep_boot, ["full_text"])

    # before / after cleaning: clean - raw on WS1-flagged vs unflagged papers
    clean_rows = []
    for flagged, sub in pq.groupby("ws1_flagged"):
        t, b = cluster_bootstrap(sub, "correct", "paper_key", ["model", "arm"], n_boot=n_boot)
        pt = {(r.model, r.arm): r.mean for r in t.itertuples()}
        for m in models:
            if (m, "ku_clean") in pt and (m, "ku_raw") in pt:
                d = paired_difference(b, (m, "ku_clean"), (m, "ku_raw"), pt[(m, "ku_clean")], pt[(m, "ku_raw")])
                clean_rows.append({"model": m, "papers": "WS1-flagged" if flagged else "not flagged",
                                   "questions": int((sub["model"] == m).sum() / max(len(arms), 1)), **d})
    fl = flips(pq)

    # lexical-bias covariate: full_text accuracy by question/context overlap tercile
    try:
        ov = load_stage_table(ctx, "contexts", "question_overlap.parquet")
        ft = pq[pq["arm"] == "full_text"].merge(ov[ov["arm"] == "full_text"][["qid", "overlap5"]], on="qid")
        ft["overlap_tercile"] = pd.qcut(ft["overlap5"].rank(method="first"), 3, labels=["low", "mid", "high"])
        bias = ft.groupby(["model", "overlap_tercile"], observed=True)["correct"].mean().reset_index()
    except (FileNotFoundError, ValueError):
        bias = pd.DataFrame()

    by_type = pq.groupby(["arm", "qtype"])["correct"].mean().unstack().reindex(arms)
    tbl.to_csv(ctx.out / "accuracy.csv", index=False)
    comp.to_csv(ctx.out / "comparisons.csv", index=False)
    dep_tbl.to_csv(ctx.out / "accuracy_context_dependent.csv", index=False)
    pd.DataFrame(clean_rows).to_csv(ctx.out / "cleaning_effect.csv", index=False)
    fl.to_csv(ctx.out / "flips.csv", index=False)
    tex = tbl.pivot(index="arm", columns="model", values="mean").reindex(arms)
    (ctx.out / "accuracy_table.tex").write_text(
        tex.map(lambda v: f"{v:.1%}" if pd.notna(v) else "–").to_latex(escape=True))

    # paper figure: accuracy vs context tokens (compression view)
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        t = tbl[(tbl["model"] == m) & (tbl["arm"] != "closed_book")].sort_values("median_tokens")
        ax.errorbar(t["median_tokens"], t["mean"], yerr=[t["mean"] - t["lo"], t["hi"] - t["mean"]], marker="o",
                    capsize=2, lw=1, label=m)
        for r in t.itertuples():
            ax.annotate(r.arm, (r.median_tokens, r.mean), fontsize=6, xytext=(3, 3), textcoords="offset points")
        cbm = tbl[(tbl["model"] == m) & (tbl["arm"] == "closed_book")]["mean"]
        if len(cbm):
            ax.axhline(float(cbm.iloc[0]), ls=":", lw=.8, color=ax.lines[-1].get_color())
    ax.set_xscale("log")
    ax.set_xlabel("median context tokens (log)")
    ax.set_ylabel("accuracy (judge CORRECT)")
    ax.legend(fontsize=7)
    fig_paths = save(fig, ctx.out / "figures", "ws2_accuracy_vs_tokens")

    def acc_series(t: pd.DataFrame) -> list[dict]:
        out = []
        for m in models:
            tt = t[t["model"] == m].set_index("arm").reindex(arms)
            out.append({"name": m, "data": tt["mean"].tolist(), "lo": tt["lo"].tolist(), "hi": tt["hi"].tolist()})
        return out

    best = tbl[tbl["arm"] != "closed_book"].sort_values("mean", ascending=False).head(1)
    cmp_cols = [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"}, {"key": "vs", "label": "vs"},
                {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "p_holm", "label": "p (Holm)", "fmt": "float3"}]
    ctx.publish("WS2 results: representation comparison", [
        {"type": "kpis", "headline": True, "items": [
            {"label": "questions", "value": int(pq["qid"].nunique()), "fmt": "int"},
            {"label": "papers", "value": int(pq["paper_key"].nunique()), "fmt": "int"},
            {"label": "answerer models", "value": len(models), "fmt": "int", "hint": ", ".join(models)},
            {"label": "context-dependent questions", "value": float((~pq[pq["arm"] == "closed_book"]["closed_book_correct"]).mean()),
             "fmt": "pct1", "hint": "not answered correctly closed-book (mean over models)"},
            *([{"label": f"best arm ({best.iloc[0]['model']})", "value": best.iloc[0]["arm"]}] if len(best) else [])]},
        {"type": "bar", "title": "Accuracy per arm and model", "fmt": "pct1", "hint": "95% paper-clustered bootstrap CI",
         "labels": arms, "series": acc_series(tbl)},
        {"type": "image", "title": "Accuracy vs context tokens (compression view)", "src": fig_paths["png"],
         "alt": "accuracy against median context tokens per arm"},
        {"type": "table", "title": "Accuracy per model and arm",
         "columns": [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "mean", "label": "accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "attempted", "label": "attempted", "fmt": "pct1"},
                     {"key": "f1", "label": "token F1", "fmt": "float3"},
                     {"key": "median_tokens", "label": "median tokens", "fmt": "int"}],
         "rows": tbl.to_dict("records"), "limit": 100},
        {"type": "table", "title": "Paired differences (vs closed book and vs full text)",
         "hint": "paired paper-clustered bootstrap; Holm over all rows", "columns": cmp_cols,
         "rows": comp.to_dict("records"), "limit": 100},
        {"type": "bar", "title": "Context-dependent stratum: accuracy per arm",
         "hint": "only questions the model does not answer correctly closed-book", "fmt": "pct1",
         "labels": [a for a in arms if a != "closed_book"],
         "series": [{**s, "data": s["data"][1:], "lo": s["lo"][1:], "hi": s["hi"][1:]} for s in acc_series(dep_tbl)]
         if arms and arms[0] == "closed_book" else acc_series(dep_tbl)},
        {"type": "table", "title": "Context-dependent stratum: difference vs full text", "columns": cmp_cols,
         "rows": dep_comp.to_dict("records"), "limit": 100},
        {"type": "table", "title": "Before / after cleaning: ku_clean − ku_raw",
         "hint": "on papers WS1 flagged vs not flagged; a cleaning effect should show on the flagged ones",
         "columns": [{"key": "model", "label": "model"}, {"key": "papers", "label": "papers"},
                     {"key": "questions", "label": "questions", "fmt": "int"},
                     {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "p", "label": "p", "fmt": "float3"}],
         "rows": clean_rows},
        {"type": "table", "title": "Help / harm flips vs closed book",
         "columns": [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "helped", "label": "helped", "fmt": "pct1"}, {"key": "harmed", "label": "harmed", "fmt": "pct1"},
                     {"key": "n", "label": "questions", "fmt": "int"}], "rows": fl.to_dict("records"), "limit": 100},
        {"type": "bar", "title": "Accuracy by question type (all models)", "fmt": "pct1", "horizontal": True,
         "labels": [str(c) for c in by_type.columns],
         "series": [{"name": a, "data": [None if np.isnan(v) else float(v) for v in by_type.loc[a].tolist()]}
                    for a in arms]},
        *([{"type": "table", "title": "Lexical bias check: full-text accuracy by question/context overlap",
            "hint": "if accuracy rises with overlap, question wording favours the full-text arm",
            "columns": [{"key": "model", "label": "model"}, {"key": "overlap_tercile", "label": "overlap"},
                        {"key": "correct", "label": "accuracy", "fmt": "pct1"}],
            "rows": bias.astype({"overlap_tercile": str}).to_dict("records")}] if len(bias) else []),
    ], summary=f"{pq['qid'].nunique()} questions x {len(arms)} arms x {len(models)} models")
    return {"questions": int(pq["qid"].nunique()), "models": models,
            "accuracy": {f"{r.model}/{r.arm}": round(r.mean, 4) for r in tbl.itertuples()}}
