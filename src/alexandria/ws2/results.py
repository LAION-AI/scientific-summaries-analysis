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

# arms on the accuracy-vs-tokens figure (the field ablations have their own panel)
FIGURE_ARMS = ["full_text", "full_text_matched", "ku_raw", "ku_clean", "ku_local", "prose_local"]


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
    abl = ablation_table(pq, tbl, boot, cx, models, n_boot)
    if len(abl):
        abl.to_csv(ctx.out / "ablation.csv", index=False)

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
        t = tbl[(tbl["model"] == m) & tbl["arm"].isin(FIGURE_ARMS)].sort_values("median_tokens")
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
        *(ablation_panels(abl) if len(abl) else []),
        *([{"type": "table", "title": "Lexical bias check: full-text accuracy by question/context overlap",
            "hint": "if accuracy rises with overlap, question wording favours the full-text arm",
            "columns": [{"key": "model", "label": "model"}, {"key": "overlap_tercile", "label": "overlap"},
                        {"key": "correct", "label": "accuracy", "fmt": "pct1"}],
            "rows": bias.astype({"overlap_tercile": str}).to_dict("records")}] if len(bias) else []),
    ], summary=f"{pq['qid'].nunique()} questions x {len(arms)} arms x {len(models)} models")
    return {"questions": int(pq["qid"].nunique()), "models": models,
            "accuracy": {f"{r.model}/{r.arm}": round(r.mean, 4) for r in tbl.itertuples()}}


# ---------------------------------------------------------------- structure ablation
ABLATION_COMPARISONS = [   # (arm, reference, what it tests)
    ("ku_local", "prose_local", "structure: same model, same source, same length (KU vs prose)"),
    ("ku_clean", "ku_flat", "structure: same content, headings and field order removed"),
    ("ku_minus_context", "ku_clean", "drop context fields"),
    ("ku_minus_methods", "ku_clean", "drop methods fields"),
    ("ku_minus_results", "ku_clean", "drop results fields"),
    ("ku_minus_claims", "ku_clean", "drop claims"),
    ("ku_minus_limitations", "ku_clean", "drop limitations"),
    ("ku_minus_takeaways", "ku_clean", "drop take-aways"),
    ("ku_results_claims", "ku_clean", "keep only results + claims"),
]


def ablation_table(pq: pd.DataFrame, tbl: pd.DataFrame, boot: dict, cx: pd.DataFrame, models: list[str],
                   n_boot: int) -> pd.DataFrame:
    """Paired differences for the structure ablation, per model and pooled over models (questions averaged over
    models first), with paper-clustered bootstrap CIs and Holm over every row; median tokens of both arms."""
    have = set(pq["arm"])
    comps = [c for c in ABLATION_COMPARISONS if c[0] in have and c[1] in have]
    if not comps:
        return pd.DataFrame()
    tok = cx.groupby("arm")["tokens"].median()
    point = {(r.model, r.arm): r.mean for r in tbl.itertuples()}
    rows = []
    for a, ref, what in comps:
        for m in models:
            if (m, a) in point and (m, ref) in point:
                rows.append({"model": m, "arm": a, "vs": ref, "tests": what,
                             **paired_difference(boot, (m, a), (m, ref), point[(m, a)], point[(m, ref)])})
    pooled = pq.groupby(["arm", "qid", "paper_key"], as_index=False)["correct"].mean()
    ptbl, pboot = cluster_bootstrap(pooled, "correct", "paper_key", ["arm"], n_boot=n_boot)
    pp = {r.arm: r.mean for r in ptbl.itertuples()}
    for a, ref, what in comps:
        rows.append({"model": "all (pooled)", "arm": a, "vs": ref, "tests": what,
                     **paired_difference(pboot, (a,), (ref,), pp[a], pp[ref])})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].tolist())
    df["acc_arm"] = [pp[a] if m == "all (pooled)" else point[(m, a)] for m, a in zip(df["model"], df["arm"], strict=True)]
    df["acc_ref"] = [pp[r] if m == "all (pooled)" else point[(m, r)] for m, r in zip(df["model"], df["vs"], strict=True)]
    df["tokens_arm"] = df["arm"].map(tok)
    df["tokens_ref"] = df["vs"].map(tok)
    return df


def ablation_panels(abl: pd.DataFrame) -> list[dict]:
    pooled = abl[abl["model"] == "all (pooled)"]
    s = pooled[pooled["vs"] == "prose_local"]
    kp = []
    if len(s):
        r = s.iloc[0]
        kp.append({"label": "KU − prose (same model, same length)", "value": float(r["diff"]), "fmt": "pct1",
                   "hint": f"95% CI {r['lo']:+.1%} to {r['hi']:+.1%}; Holm p = {r['p_holm']:.3g}"})
    f = pooled[pooled["arm"] == "ku_clean"]
    if len(f):
        r = f.iloc[0]
        kp.append({"label": "headings + field order (ku_clean − ku_flat)", "value": float(r["diff"]), "fmt": "pct1",
                   "hint": f"95% CI {r['lo']:+.1%} to {r['hi']:+.1%}"})
    drops = pooled[pooled["vs"] == "ku_clean"].sort_values("diff")
    if len(drops):
        kp.append({"label": "most important field group", "value": drops.iloc[0]["arm"].replace("ku_minus_", ""),
                   "hint": f"dropping it costs {-drops.iloc[0]['diff']:.1%}"})
    cols = [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"}, {"key": "vs", "label": "vs"},
            {"key": "tests", "label": "tests"}, {"key": "acc_arm", "label": "acc. arm", "fmt": "pct1"},
            {"key": "acc_ref", "label": "acc. ref", "fmt": "pct1"}, {"key": "diff", "label": "Δ", "fmt": "pct1"},
            {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"},
            {"key": "p_holm", "label": "p (Holm)", "fmt": "float3"},
            {"key": "tokens_arm", "label": "tokens arm", "fmt": "int"}, {"key": "tokens_ref", "label": "tokens ref", "fmt": "int"}]
    return [
        {"type": "kpis", "items": kp},
        {"type": "bar", "title": "Structure ablation: Δ accuracy (pooled over models)", "fmt": "pct1", "horizontal": True,
         "hint": "paired, paper-clustered bootstrap 95% CI; first two rows test structure, the rest drop field groups",
         "labels": [f"{r.arm} − {r.vs}" for r in pooled.itertuples()],
         "series": [{"name": "Δ accuracy", "data": pooled["diff"].tolist(), "lo": pooled["lo"].tolist(),
                     "hi": pooled["hi"].tolist()}]},
        {"type": "table", "title": "Structure ablation: paired differences (per model and pooled)", "columns": cols,
         "rows": abl.to_dict("records"), "limit": 120},
    ]
