"""Stage `results` (WS4): is the KU a good training signal?

Per model x method x condition (mean over seeds):
  * pushing models: closed-book science score (mean over the science sets that ran) and every set on its own;
  * forgetting: general sets (MMLU, GSM8K, IFEval);
  * knowledge injection: probe accuracy (SimpleQA judge) on trained vs held-out papers; the injection effect is
    (trained - held-out) gain over the base model, a difference in differences, with a 95% bootstrap CI that
    resamples papers (held-out papers control for any general improvement).
Intervals: with >= 2 seeds a t-interval over seeds; with one seed the lm-eval standard error (x 1.96). Differences
vs the untrained base model and vs the `none` control (replay only, same steps) of the same model and method.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..figures import plt, save, style
from ..runtime import Context, stage
from .evaluate import LM_TASKS, extract_rows

SET_OF = {t: s for s, ts in LM_TASKS.items() if ts for t in ts}
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}


def set_scores(unit_json: dict) -> dict[str, tuple[float, float | None]]:
    """lm-eval task rows -> one score (and stderr) per eval set (MMLU-Pro STEM = n-weighted mean of its subjects)."""
    rows = [r for r in unit_json.get("rows", []) if r["task"] in SET_OF]
    out = {}
    for s in {SET_OF[r["task"]] for r in rows}:
        rs = [r for r in rows if SET_OF[r["task"]] == s]
        w = np.array([r.get("n") or 1 for r in rs], float)
        v = float(np.average([r["value"] for r in rs], weights=w))
        se = [r["stderr"] for r in rs if r.get("stderr") is not None]
        out[s] = (v, float(math.sqrt(np.sum((w / w.sum()) ** 2 * np.array(se) ** 2))) if len(se) == len(rs) else None)
    return out


def ci(values: list[float], stderr: float | None) -> tuple[float, float, float]:
    m = float(np.mean(values))
    if len(values) >= 2:
        h = T95.get(len(values) - 1, 1.96) * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    else:
        h = 1.96 * stderr if stderr is not None else float("nan")
    return m, m - h, m + h


def injection(g: pd.DataFrame, base: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """(trained - held-out) gain over base, with a paper-resampling bootstrap CI."""
    def acc(df):
        return df.groupby(["split", "paper_key"])["correct"].mean()
    a, b = acc(g), acc(base)
    rng = np.random.default_rng(seed)
    est = []
    papers = {s: a.loc[s].index.to_numpy() for s in ("train", "heldout") if s in a.index.get_level_values(0)}
    if len(papers) < 2:
        return {"effect": float("nan"), "lo": float("nan"), "hi": float("nan")}

    def effect(sel):
        return ((a.loc["train"].reindex(sel["train"]).mean() - b.loc["train"].reindex(sel["train"]).mean())
                - (a.loc["heldout"].reindex(sel["heldout"]).mean() - b.loc["heldout"].reindex(sel["heldout"]).mean()))
    point = effect(papers)
    for _ in range(n_boot):
        est.append(effect({s: rng.choice(p, len(p)) for s, p in papers.items()}))
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return {"effect": float(point), "lo": float(lo), "hi": float(hi),
            "gain_train": float(a.loc["train"].mean() - b.loc["train"].mean()),
            "gain_heldout": float(a.loc["heldout"].mean() - b.loc["heldout"].mean())}


@stage("ws4", "results", needs=("eval", "grade"), config_keys=("ws4.results", "ws4.eval"),
       help="science gain, forgetting and knowledge injection per condition, with CIs over seeds / papers")
def run(ctx: Context) -> dict:
    cfg, ecfg = ctx.section("ws4.results"), ctx.section("ws4.eval")
    units_dir = ctx.cfg.stage_dir("ws4", "eval") / "units"
    units = [json.loads(p.read_text()) for p in sorted(units_dir.glob("*/lm_eval.json"))]
    for u in units:                              # re-read lm-eval's raw output: metric choice lives in METRICS
        raw = sorted((units_dir / u["run"] / "lm_eval").rglob("results_*.json"))
        if raw:
            u["rows"] = extract_rows(json.loads(raw[-1].read_text()))
    if not units:
        return ctx.skip("WS4 results", "no evaluated units yet")
    science, general = list(ecfg.get("science", [])), list(ecfg.get("general", []))
    long = []
    for u in units:
        for s, (v, se) in set_scores(u).items():
            long.append({**{k: u[k] for k in ("run", "model", "method", "condition", "seed")}, "set": s, "value": v,
                         "stderr": se, "kind": "science" if s in science else "general"})
    df = pd.DataFrame(long)
    df.to_csv(ctx.out / "scores_long.csv", index=False)
    agg = []
    for (m, meth, cond, s), g in df.groupby(["model", "method", "condition", "set"]):
        mean, lo, hi = ci(g["value"].tolist(), g["stderr"].dropna().mean() if g["stderr"].notna().any() else None)
        agg.append({"model": m, "method": meth, "condition": cond, "set": s, "mean": mean, "lo": lo, "hi": hi,
                    "seeds": len(g), "kind": g["kind"].iloc[0]})
    agg = pd.DataFrame(agg)
    base = agg[agg["method"] == "base"].set_index(["model", "set"])["mean"]
    agg["delta_base"] = agg["mean"] - [base.get((m, s), np.nan) for m, s in zip(agg["model"], agg["set"], strict=True)]
    none = agg[agg["condition"] == "none"].set_index(["model", "method", "set"])["mean"]
    agg["delta_none"] = agg["mean"] - [none.get((m, me, s), np.nan) for m, me, s in
                                       zip(agg["model"], agg["method"], agg["set"], strict=True)]
    agg.to_csv(ctx.out / "scores.csv", index=False)
    summ = agg.groupby(["model", "method", "condition", "kind"])[["mean", "delta_base", "delta_none"]].mean().reset_index()
    wide = summ.pivot_table(index=["model", "method", "condition"], columns="kind",
                            values=["delta_base", "delta_none", "mean"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide.to_csv(ctx.out / "summary.csv", index=False)

    # knowledge injection
    inj_rows = []
    gp = ctx.cfg.stage_dir("ws4", "grade") / "graded.parquet"
    if gp.exists():
        g = pd.read_parquet(gp)
        meta = {u["run"]: u for u in units}
        g["model"] = g["run"].map(lambda r: meta.get(r, {}).get("model"))
        for run, gr in g.groupby("run"):
            u = meta.get(run, {})
            b = g[g["run"] == f"{u.get('model')}__base"]
            row = {"run": run, "model": u.get("model"), "method": u.get("method"), "condition": u.get("condition"),
                   "seed": u.get("seed"), "acc_train": float(gr.loc[gr["split"] == "train", "correct"].mean()),
                   "acc_heldout": float(gr.loc[gr["split"] == "heldout", "correct"].mean())}
            if u.get("method") != "base" and len(b):
                row.update(injection(gr, b, int(cfg.get("n_boot", 2000))))
            inj_rows.append(row)
    inj = pd.DataFrame(inj_rows)
    inj.to_csv(ctx.out / "knowledge_injection.csv", index=False)

    # headline figure: science delta vs base per condition, one panel per method
    style()
    sci = agg[(agg["kind"] == "science") & (agg["method"] != "base")].groupby(["method", "condition"])["delta_base"].mean().reset_index()
    methods = sorted(sci["method"].unique())
    fig, axes = plt.subplots(1, max(len(methods), 1), figsize=(3.2 * max(len(methods), 1), 3.2), sharey=True, squeeze=False)
    for ax, meth in zip(axes[0], methods, strict=False):
        s = sci[sci["method"] == meth]
        ax.bar(s["condition"], s["delta_base"] * 100, color="#4C72B0")
        ax.axhline(0, color="#888", lw=.8)
        ax.set_title(meth, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0][0].set_ylabel("Δ science score vs base (points)")
    figs = save(fig, ctx.out / "figures", "ws4_science_delta")

    cond_order = list(ctx.section("ws4").get("conditions", []))
    best = sci.sort_values("delta_base", ascending=False).head(1)
    kp = [{"label": "units evaluated", "value": len(units), "fmt": "int"},
          {"label": "best science Δ vs base", "value": float(best["delta_base"].iloc[0]) if len(best) else None,
           "fmt": "pct1", "hint": f"{best['method'].iloc[0]} / {best['condition'].iloc[0]}" if len(best) else ""}]
    if len(inj) and "effect" in inj:
        for c in ("summary", "ku", "full"):
            e = inj[inj["condition"] == c]["effect"]
            if len(e):
                kp.append({"label": f"knowledge injection: {c}", "value": float(e.mean()), "fmt": "pct1",
                           "hint": "(trained − held-out) gain over base"})
    table_cols = [{"key": "model", "label": "model"}, {"key": "method", "label": "method"},
                  {"key": "condition", "label": "condition"}, {"key": "set", "label": "eval set"},
                  {"key": "mean", "label": "score", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                  {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "delta_base", "label": "Δ vs base", "fmt": "pct1"},
                  {"key": "delta_none", "label": "Δ vs no-extra-data", "fmt": "pct1"}, {"key": "seeds", "label": "seeds", "fmt": "int"}]
    panels = [{"type": "kpis", "headline": True, "items": kp},
              {"type": "image", "title": "Pushing models: Δ closed-book science score vs the base model", "src": figs["png"],
               "alt": "science delta per condition and method"},
              {"type": "table", "title": "Science benchmarks (main claim)", "columns": table_cols,
               "rows": agg[agg["kind"] == "science"].sort_values(["model", "method", "condition", "set"]).to_dict("records")},
              {"type": "table", "title": "Forgetting: general benchmarks", "columns": table_cols,
               "rows": agg[agg["kind"] == "general"].sort_values(["model", "method", "condition", "set"]).to_dict("records")}]
    if len(inj):
        panels.append({"type": "table", "title": "Knowledge injection: closed-book probe on trained vs held-out papers",
                       "hint": "effect = (trained − held-out) gain over the base model; 95% CI resamples papers",
                       "columns": [{"key": "run", "label": "run"}, {"key": "condition", "label": "condition"},
                                   {"key": "acc_train", "label": "trained papers", "fmt": "pct1"},
                                   {"key": "acc_heldout", "label": "held-out papers", "fmt": "pct1"},
                                   {"key": "effect", "label": "injection effect", "fmt": "pct1"},
                                   {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"}],
                       "rows": inj.to_dict("records")})
    if cond_order:
        panels.append({"type": "markdown", "text": "Conditions: " + ", ".join(cond_order) + ". All built from the same "
                       "papers; document runs are token-matched, QA runs have the same number of pairs per paper."})
    ctx.publish("WS4 results: is the KU a good training signal?", panels,
                summary=f"{len(units)} units; best science Δ "
                        + (f"{best['delta_base'].iloc[0]:+.1%} ({best['method'].iloc[0]}/{best['condition'].iloc[0]})" if len(best) else "n/a"))
    return {"units": len(units)}
