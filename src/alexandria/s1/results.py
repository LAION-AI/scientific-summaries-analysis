"""Stage `results` (S1): the safety picture of the dataset and of the Alexandria skill, and what the release needs.

Rates carry Wilson 95% intervals; the safety-challenge deltas (WMDP uplift, change in harmful responses with
retrieval) are paired per item / prompt with a bootstrap 95% interval. The release checklist turns each finding into
an action (flag / drop records, a `safety_flags` column, a retraction flag, injection sanitisation for the skill).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..runtime import Context, stage
from ..stats import rate_row


def paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean(b - a) over paired units with a bootstrap 95% CI."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if not len(d):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _j(ctx: Context, stage_: str, name: str) -> dict:
    p = ctx.cfg.stage_dir("s1", stage_) / name
    return json.loads(p.read_text()) if p.exists() else {}


def _parts(ctx: Context, stage_: str, pattern: str) -> pd.DataFrame:
    fs = sorted((ctx.cfg.stage_dir("s1", stage_) / "parts").glob(pattern))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


@stage("s1", "results", needs=("pii", "injection", "retractions", "guard", "dual-use", "injection-live", "uplift"),
       config_keys=("s1.results",), help="safety summary: headline rates with CIs, safety-challenge deltas, release checklist")
def run(ctx: Context) -> dict:
    nb = int(ctx.section("s1.results").get("n_boot", 2000))
    pii, inj, ret = _j(ctx, "pii", "pii.json"), _j(ctx, "injection", "injection.json"), _j(ctx, "retractions", "retractions.json")
    grd, du = _j(ctx, "guard", "guard.json"), _j(ctx, "dual-use", "dual_use.json")
    live = _parts(ctx, "injection-live", "*.parquet")
    w, d = _parts(ctx, "uplift", "wmdp_*.parquet"), _parts(ctx, "uplift", "dna_*.parquet")

    up_rows = []
    for (m, b), g in w.groupby(["model", "bench"]) if len(w) else []:
        p = g.pivot_table(index="item_id", columns="arm", values="correct")
        if {"A0", "R"} <= set(p.columns):
            delta, lo, hi = paired_delta(p["A0"].to_numpy(), p["R"].to_numpy(), nb)
            up_rows.append({"model": m, "bench": b, "items": len(p), "acc_A0": float(p["A0"].mean()), "acc_R": float(p["R"].mean()),
                            "uplift": delta, "lo": lo, "hi": hi})
    dna_rows = []
    for (m,), g in d.groupby(["model"]) if len(d) else []:
        p = g.pivot_table(index="prompt_id", columns="arm", values="harmful", aggfunc="max").astype(float)
        r = g.pivot_table(index="prompt_id", columns="arm", values="refusal", aggfunc="max").astype(float)
        delta, lo, hi = paired_delta(p["A0"].to_numpy(), p["R"].to_numpy(), nb) if {"A0", "R"} <= set(p.columns) else (np.nan,) * 3
        for arm in ("A0", "R"):
            ga = g[g["arm"] == arm]
            h, rf = rate_row(int(ga["harmful"].sum()), len(ga)), rate_row(int(ga["refusal"].fillna(False).sum()), len(ga))
            dna_rows.append({"model": m, "arm": arm, "prompts": len(ga), "harmful": h["rate"], "harmful_lo": h["lo"], "harmful_hi": h["hi"],
                             "refusal": rf["rate"], "refusal_lo": rf["lo"], "refusal_hi": rf["hi"],
                             "delta_harmful_R_minus_A0": delta if arm == "R" else None, "delta_lo": lo if arm == "R" else None,
                             "delta_hi": hi if arm == "R" else None})
    up, dn = pd.DataFrame(up_rows), pd.DataFrame(dna_rows)
    up.to_csv(ctx.out / "wmdp_uplift.csv", index=False)
    dn.to_csv(ctx.out / "dna_harm.csv", index=False)
    inj_live = live[live["condition"] == "injected"] if len(live) else live
    live_rate = rate_row(int(inj_live["followed"].sum()), len(inj_live)) if len(inj_live) else None
    mean_uplift = float(up["uplift"].mean()) if len(up) else None
    harm_delta = float(dn["delta_harmful_R_minus_A0"].dropna().mean()) if len(dn) else None

    checklist = [
        {"finding": "PII in summaries", "measured": f"{pii.get('summary', {}).get('k', '?')} of {pii.get('records', '?')} summaries",
         "action": "redact detected PII in summaries; add `safety_flags.pii`; keep author contacts out of KUs"},
        {"finding": "unsafe / controversial summaries (guard model)",
         "measured": f"Unsafe {grd.get('unsafe', {}).get('k', '?')}, non-Safe {grd.get('non_safe', {}).get('k', '?')} of {grd.get('n', '?')}; kappa {grd.get('kappa', float('nan')):.2f}",
         "action": "human review of non-Safe records; `safety_flags.guard` column; drop only confirmed harmful content"},
        {"finding": "dual-use: actionable procedural detail", "measured": f"{du.get('actionable', '?')} of {du.get('records', '?')} judged actionable",
         "action": "expert review of review_packet.csv; drop or redact confirmed records; document the screen in the datasheet"},
        {"finding": "prompt-injection-like text", "measured": f"summaries {inj.get('summary', {}).get('k', '?')}, full texts {inj.get('fulltext', {}).get('k', '?')} of {inj.get('records', '?')}",
         "action": "strip control / invisible characters and chat-template tags in the skill's tool output; mark tool output as untrusted data in the agent prompt"},
        {"finding": "agents follow injected instructions", "measured": f"{live_rate['rate']:.0%} of injected episodes" if live_rate else "n/a",
         "action": "sanitise retrieved text, delimit it, and instruct the agent that retrieved text is data, never instructions; re-test"},
        {"finding": "retracted papers", "measured": f"{ret.get('total', {}).get('k', '?')} records ({ret.get('total', {}).get('rate', float('nan')):.3%})",
         "action": "add a `retracted` flag; exclude retracted papers from the skill's default index"},
        {"finding": "hazardous-knowledge uplift with retrieval (WMDP)", "measured": f"mean Δ accuracy {mean_uplift:+.1%}" if mean_uplift is not None else "n/a",
         "action": "report uplift per domain; if positive and significant, consider filtering hazardous domains from the default index"},
        {"finding": "harmful responses with retrieval (Do-Not-Answer)", "measured": f"mean Δ harmful {harm_delta:+.1%}" if harm_delta is not None else "n/a",
         "action": "if retrieval raises harmful compliance, add a safety system prompt / guard filter in the skill"},
    ]
    pd.DataFrame(checklist).to_csv(ctx.out / "release_checklist.csv", index=False)
    kp = [
        {"label": "summaries with PII", "value": (pii.get("summary") or {}).get("rate"), "fmt": "pct1"},
        {"label": "Unsafe summaries (guard)", "value": (grd.get("unsafe") or {}).get("rate"), "fmt": "pct1"},
        {"label": "actionable dual-use flags", "value": du.get("actionable"), "fmt": "int"},
        {"label": "injection success (live)", "value": live_rate["rate"] if live_rate else None, "fmt": "pct1", "status": "warn"},
        {"label": "retracted records", "value": (ret.get("total") or {}).get("rate"), "fmt": "pct1"},
        {"label": "WMDP uplift with retrieval", "value": mean_uplift, "fmt": "pct1"},
        {"label": "Δ harmful responses with retrieval", "value": harm_delta, "fmt": "pct1"},
    ]
    panels = [{"type": "kpis", "headline": True, "items": kp},
              {"type": "table", "title": "Release recommendations", "rows": checklist,
               "columns": [{"key": "finding", "label": "finding"}, {"key": "measured", "label": "measured"}, {"key": "action", "label": "action"}]},
              {"type": "table", "title": "Safety challenge (a): WMDP accuracy closed book (A0) vs Alexandria retrieval (R)",
               "hint": "uplift = paired Δ accuracy (R - A0), bootstrap 95% CI over items", "rows": up_rows, "columns": [
                   {"key": "model", "label": "model"}, {"key": "bench", "label": "subset"}, {"key": "items", "label": "items", "fmt": "int"},
                   {"key": "acc_A0", "label": "A0", "fmt": "pct1"}, {"key": "acc_R", "label": "R", "fmt": "pct1"},
                   {"key": "uplift", "label": "uplift", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"}]},
              {"type": "table", "title": "Safety challenge (b): Do-Not-Answer harmful / refused responses (Qwen3Guard response mode)",
               "hint": "Δ = paired change in harmful responses with retrieval (R - A0), bootstrap 95% CI over prompts; responses are not stored",
               "rows": dna_rows, "columns": [
                   {"key": "model", "label": "model"}, {"key": "arm", "label": "arm"}, {"key": "prompts", "label": "prompts", "fmt": "int"},
                   {"key": "harmful", "label": "harmful", "fmt": "pct1"}, {"key": "refusal", "label": "refused", "fmt": "pct1"},
                   {"key": "delta_harmful_R_minus_A0", "label": "Δ harmful", "fmt": "pct1"},
                   {"key": "delta_lo", "label": "lo", "fmt": "pct1"}, {"key": "delta_hi", "label": "hi", "fmt": "pct1"}]},
              {"type": "markdown", "text": "Limits: regex PII and injection screens have unmeasured recall (spot-check packets give "
                                           "precision); guard models and the dual-use judge are automatic raters (human review packets "
                                           "provided); WMDP measures hazardous knowledge, not capability to act; the debug run uses "
                                           "the WS1 sample and small benchmark subsets. Gated safety sets (HarmBench, SORRY-Bench, "
                                           "WildJailbreak, Llama-Guard-4) were not used."}]
    ctx.publish("S1 results: safety of the dataset and the skill", panels,
                summary=f"PII {kp[0]['value'] or 0:.1%} of summaries; injection followed {kp[3]['value'] or 0:.0%}; "
                        f"WMDP uplift {mean_uplift or 0:+.1%}")
    return {"checklist": len(checklist)}
