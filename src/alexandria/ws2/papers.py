"""Stage `papers` (WS2): the evaluation papers.

Eligible: full-text input to the summarizer, readable source (not `source_garbled`), not a wrong-paper suspect, a
source of at least `min_source_chars`, a usable KU (not KU-Fixes `needs_regeneration`) and not a dropped duplicate copy. The sample is stratified by domain and deliberately over-samples papers that
WS1 flagged (critical / serious / minor; `flagged_share`), so the before/after-cleaning comparison has a subset of
affected papers large enough to measure on (the overall effect would otherwise be diluted). Deterministic (seed).
"""
from __future__ import annotations

import pandas as pd

from ..papers import paper_table, with_paper_info
from ..runtime import Context, stage
from ..ws1.faith.common import features_table, paper_key, stratified_pick


@stage("ws2", "papers", config_keys=("ws2.papers",),
       help="evaluation papers: full text, readable, stratified by domain, WS1-flagged papers over-sampled")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.papers")
    n = int(cfg.get("n", 100))
    if ctx.cfg.profile == "debug":
        n = int(ctx.cfg.profile_settings.get("ws2_papers", n))
    seed = int(cfg.get("seed", 0))
    f = features_table(ctx, ["file", "row", "paper_id", "group", "domain", "input_type", "has_text", "source_garbled",
                             "wrong_paper_suspect", "text_length", "ku_needs_regeneration", "dup_drop_any"])
    ok = f[f["has_text"].fillna(False) & ~f["source_garbled"].fillna(False) & ~f["wrong_paper_suspect"].fillna(False)
           & f["input_type"].isin(cfg.get("input_types", ["full_text"]))
           & (f["text_length"].fillna(0) >= int(cfg.get("min_source_chars", 5000)))
           & ~(f["ku_needs_regeneration"].fillna(False) & bool(cfg.get("exclude_ku_regeneration", True)))
           & ~(f["dup_drop_any"].fillna(False) & bool(cfg.get("exclude_duplicate_copies", True)))].copy()
    flagged_csv = ctx.cfg.stage_dir("ws1", "quality") / "flagged_papers.csv"
    flagged = pd.read_csv(flagged_csv)[["file", "row", "flags", "severity"]] if flagged_csv.exists() \
        else pd.DataFrame(columns=["file", "row", "flags", "severity"])
    ok = ok.merge(flagged, on=["file", "row"], how="left")
    ok["ws1_flagged"] = ok["flags"].notna()
    n_flag = min(int(round(n * float(cfg.get("flagged_share", 0.3)))), int(ok["ws1_flagged"].sum()))
    strata = list(cfg.get("strata", ["domain"]))
    pick = pd.concat([stratified_pick(ok[ok["ws1_flagged"]], n_flag, strata, seed),
                      stratified_pick(ok[~ok["ws1_flagged"]], n - n_flag, strata, seed)])
    pick["paper_key"] = [paper_key(a, b) for a, b in zip(pick["file"], pick["row"], strict=True)]
    pick = with_paper_info(ctx, pick.sort_values(["file", "row"]).reset_index(drop=True))
    pick.to_parquet(ctx.out / "papers.parquet", index=False)
    dom = pick.groupby("domain", dropna=False).size().sort_values(ascending=False)
    ctx.publish("WS2 evaluation papers", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(pick), "fmt": "int"},
            {"label": "eligible", "value": len(ok), "fmt": "int", "hint": "full text, readable, not wrong-paper"},
            {"label": "WS1-flagged papers", "value": int(pick["ws1_flagged"].sum()), "fmt": "int",
             "hint": f"target share {cfg.get('flagged_share', 0.3):.0%} (before/after-cleaning subset)"},
            {"label": "domains", "value": int(dom.size), "fmt": "int"}]},
        {"type": "bar", "title": "Papers per domain", "horizontal": True, "fmt": "int",
         "labels": [str(x) for x in dom.index], "series": [{"name": "papers", "data": dom.tolist()}]},
        paper_table("Evaluation papers", pick, [("domain", "domain", None), ("ws1_flagged", "WS1 flagged", None),
                                                ("flags", "WS1 flags", None)]),
    ], summary=f"{len(pick)} papers ({int(pick['ws1_flagged'].sum())} WS1-flagged)")
    return {"papers": len(pick), "flagged": int(pick["ws1_flagged"].sum()), "eligible": len(ok)}
