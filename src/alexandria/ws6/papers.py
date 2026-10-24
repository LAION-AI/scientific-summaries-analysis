"""Stages `papers` and `qa` (WS6).

papers  Projector papers (train / dev), split BY PAPER, from the WS2 eligibility pool (full-text input, readable,
        usable KU, not a dropped duplicate copy), stratified by domain, never a WS2 evaluation paper or a WS3 gold
        paper (those are the test set of the extreme-compression point). Each paper's cleaned KU is stored.
qa      QA pairs about the projector papers, generated from the KU prose with the WS4 generator and checks (evidence
        must be in the KU, short verifiable answer, self-contained question): the stage-2 training signal.
"""
from __future__ import annotations

import json

import pandas as pd

from ..data import fetch_records
from ..llm import model_spec, with_model
from ..runtime import Context, log, stage
from ..schema import SUMMARY_FIELDS
from ..ws1.faith.common import features_table, paper_key, stratified_pick
from ..ws2.ku import FIX_COLS, _lookup, apply_fixes
from ..ws4.common import clean_ku, render
from ..ws4.qa_data import _generate, valid
from .common import setting, stage_table


def excluded_papers(ctx: Context) -> set[str]:
    """WS2 evaluation papers and WS3 benchmark gold papers: the test side of WS6."""
    out: set[str] = set()
    for ws, st, fn, col in (("ws2", "papers", "papers.parquet", "paper_key"),
                            ("ws3", "benchmarks", "items.parquet", "gold_paper_key")):
        p = ctx.cfg.stage_dir(ws, st) / fn
        if p.exists():
            out |= set(pd.read_parquet(p, columns=[col])[col].dropna())
    return out


def cleaned_kus(ctx: Context, keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """(file, row) -> cleaned KU (KU-Fixes applied, WS1 normalizer)."""
    recs = fetch_records(ctx.cfg, keys, ["paper_id", *SUMMARY_FIELDS])
    root, files = ctx.cfg.site.ku_fixes, sorted({f for f, _ in keys})
    fixes = _lookup(root / "ku_fix_index.parquet" if root else None, FIX_COLS, files)
    out = {}
    for k in keys:
        rec = recs.get((k[0], int(k[1])))
        if rec is not None:
            fixed, _ = apply_fixes({f: rec.get(f) for f in SUMMARY_FIELDS}, fixes.get((k[0], int(k[1]))))
            out[(k[0], int(k[1]))] = clean_ku(fixed)
    return out


@stage("ws6", "papers", config_keys=("ws6.papers",),
       help="projector train / dev papers (split by paper; never WS2 / WS3 eval papers) with their cleaned KU")
def run_papers(ctx: Context) -> dict:
    cfg = ctx.section("ws6.papers")
    n_train = int(setting(ctx, "train_papers", cfg.get("n_train", 3000)))
    n_dev = int(setting(ctx, "dev_papers", cfg.get("n_dev", 300)))
    f = features_table(ctx, ["file", "row", "domain", "input_type", "has_text", "source_garbled",
                             "wrong_paper_suspect", "text_length", "ku_needs_regeneration", "dup_drop_any"])
    ok = f[f["has_text"].fillna(False) & ~f["source_garbled"].fillna(False) & ~f["wrong_paper_suspect"].fillna(False)
           & (f["input_type"] == "full_text") & (f["text_length"].fillna(0) >= int(cfg.get("min_source_chars", 5000)))
           & ~f["ku_needs_regeneration"].fillna(False) & ~f["dup_drop_any"].fillna(False)].copy()
    ok["paper_key"] = [paper_key(a, b) for a, b in zip(ok["file"], ok["row"], strict=True)]
    excl = excluded_papers(ctx)
    pool = ok[~ok["paper_key"].isin(excl)]
    if len(pool) < n_train + n_dev:
        raise ValueError(f"only {len(pool)} eligible papers for {n_train} train + {n_dev} dev")
    strata, seed = list(cfg.get("strata", ["domain"])), int(cfg.get("seed", 0))
    both = stratified_pick(pool, n_train + n_dev, strata, seed)
    dev = stratified_pick(both, n_dev, strata, seed + 1)
    both["split"] = "train"
    both.loc[both["paper_key"].isin(set(dev["paper_key"])), "split"] = "dev"
    kus = cleaned_kus(ctx, list(zip(both["file"], both["row"], strict=True)))
    both["ku"] = [json.dumps(kus.get((a, int(b)), {}), ensure_ascii=False) for a, b in zip(both["file"], both["row"], strict=True)]
    both = both[both["ku"] != "{}"].sort_values(["split", "file", "row"]).reset_index(drop=True)
    both.to_parquet(ctx.out / "papers.parquet", index=False)
    n_tr, n_dv = int((both["split"] == "train").sum()), int((both["split"] == "dev").sum())
    ctx.publish("WS6 projector papers", [{"type": "kpis", "items": [
        {"label": "train papers", "value": n_tr, "fmt": "int"},
        {"label": "dev papers", "value": n_dv, "fmt": "int"},
        {"label": "excluded (WS2 / WS3 eval)", "value": len(excl), "fmt": "int", "hint": "the test side; never trained on"},
        {"label": "eligible pool", "value": len(pool), "fmt": "int"}]}],
        summary=f"{n_tr} train + {n_dv} dev papers (split by paper)")
    return {"train": n_tr, "dev": n_dv, "excluded": len(excl)}


@stage("ws6", "qa", needs=("papers",), resources="gpu", env="llm", config_keys=("ws6.qa",),
       help="QA pairs about the projector papers from their KU prose (WS4 generator and checks)")
def run_qa(ctx: Context) -> dict:
    cfg = ctx.section("ws6.qa")
    papers = stage_table(ctx, "papers", "papers.parquet")
    items = [{"paper_key": p.paper_key, "condition": "summary", "text": render(json.loads(p.ku), "prose")}
             for p in papers.itertuples()]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"qa: {len(items)} papers with {gen['name']}")
    outs = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    split = dict(zip(papers["paper_key"], papers["split"], strict=True))
    cap, rows = int(cfg.get("pairs_per_paper", 8)), []
    for it, pairs in zip(items, outs, strict=True):
        ok = [p for p in pairs if valid(p, it["text"], float(cfg.get("min_evidence_match", 85)))][:cap]
        rows += [{"pid": f"{it['paper_key']}:{j}", "paper_key": it["paper_key"], "split": split[it["paper_key"]],
                  "question": p["question"].strip(), "answer": p["answer"].strip()} for j, p in enumerate(ok)]
    qa = pd.DataFrame(rows)
    qa.to_parquet(ctx.out / "qa.parquet", index=False)
    by = qa.groupby("split").agg(pairs=("pid", "size"), papers=("paper_key", "nunique")).reset_index()
    ctx.publish("WS6 projector QA pairs", [{"type": "kpis", "items": [
        {"label": "QA pairs", "value": len(qa), "fmt": "int"},
        *[{"label": f"{r.split} pairs", "value": int(r.pairs), "fmt": "int"} for r in by.itertuples()],
        {"label": "generator", "value": gen["name"]}]}], summary=f"{len(qa)} QA pairs on {qa['paper_key'].nunique()} papers")
    return {"pairs": len(qa)}
