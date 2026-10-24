"""Stages `papers` and `probe` (WS4).

papers  The trained papers and the held-out papers (knowledge-injection control), drawn from one eligible pool with
        the WS2 rules (full-text input, readable, usable KU, not a dropped duplicate copy), stratified by domain,
        never a WS2 / WS3 evaluation paper. Each paper also gets its cleaned KU (KU-Fixes + WS1 normalizer) and its
        full text, stored for the later stages (`papers.parquet`, `ku` as JSON).
probe   Closed-book knowledge questions on trained AND held-out papers, written from the full text with the WS2
        question generator and all its checks (evidence in the paper, paraphrased, self-contained, LLM critic).
        Trained papers should improve after training on their representation; held-out papers should not.
"""
from __future__ import annotations

import json

import pandas as pd

from ..data import fetch_records
from ..llm import model_spec, with_model
from ..papers import with_paper_info
from ..qa import normalize_answer
from ..runtime import Context, log, stage
from ..schema import SUMMARY_FIELDS
from ..ws1.faith.common import features_table, paper_key, source_text, stratified_pick
from ..ws2.ku import FIX_COLS, _lookup, apply_fixes
from ..ws2.questions import CRITIC_SCHEMA, _generate, _norm, verify
from .common import clean_ku, setting, stage_table


@stage("ws4", "papers", config_keys=("ws4.papers",),
       help="trained + held-out papers from one eligible pool (WS2 rules), with cleaned KU and full text")
def run_papers(ctx: Context) -> dict:
    cfg = ctx.section("ws4.papers")
    n_train = int(setting(ctx, "train_papers", cfg.get("n_train", 2000)))
    n_held = int(setting(ctx, "heldout_papers", cfg.get("n_heldout", 500)))
    f = features_table(ctx, ["file", "row", "domain", "input_type", "has_text", "source_garbled",
                             "wrong_paper_suspect", "text_length", "ku_needs_regeneration", "dup_drop_any"])
    ok = f[f["has_text"].fillna(False) & ~f["source_garbled"].fillna(False) & ~f["wrong_paper_suspect"].fillna(False)
           & (f["input_type"] == "full_text") & (f["text_length"].fillna(0) >= int(cfg.get("min_source_chars", 5000)))
           & ~f["ku_needs_regeneration"].fillna(False) & ~f["dup_drop_any"].fillna(False)].copy()
    ok["paper_key"] = [paper_key(a, b) for a, b in zip(ok["file"], ok["row"], strict=True)]
    excluded = set()
    if cfg.get("exclude_ws2_ws3", True):
        for ws, st, col in (("ws2", "papers", "paper_key"), ("ws3", "benchmarks", "gold_paper_key")):
            p = ctx.cfg.stage_dir(ws, st) / ("papers.parquet" if st == "papers" else "items.parquet")
            if p.exists():
                excluded |= set(pd.read_parquet(p, columns=[col])[col].dropna())
    ok = ok[~ok["paper_key"].isin(excluded)]
    if len(ok) < n_train + n_held:
        raise ValueError(f"only {len(ok)} eligible papers for {n_train} trained + {n_held} held-out")
    strata, seed = list(cfg.get("strata", ["domain"])), int(cfg.get("seed", 0))
    both = stratified_pick(ok, n_train + n_held, strata, seed)
    held = stratified_pick(both, n_held, strata, seed + 1)
    both["split"] = "train"
    both.loc[both["paper_key"].isin(set(held["paper_key"])), "split"] = "heldout"
    keys = list(zip(both["file"], both["row"], strict=True))
    recs = fetch_records(ctx.cfg, keys, ["paper_id", "text_raw", "text_sanitized", *SUMMARY_FIELDS])
    root, files = ctx.cfg.site.ku_fixes, sorted(both["file"].unique())
    fixes = _lookup(root / "ku_fix_index.parquet" if root else None, FIX_COLS, files)
    kus, fulls = [], []
    for p in both.itertuples():
        rec = recs[(p.file, int(p.row))]
        fixed, _ = apply_fixes({k: rec.get(k) for k in SUMMARY_FIELDS}, fixes.get((p.file, int(p.row))))
        kus.append(json.dumps(clean_ku(fixed), ensure_ascii=False))
        fulls.append(source_text(rec))
    both["ku"], both["full"] = kus, fulls
    both = with_paper_info(ctx, both.sort_values(["split", "file", "row"]).reset_index(drop=True))
    both.to_parquet(ctx.out / "papers.parquet", index=False)
    dom = both.groupby(["domain", "split"]).size().unstack(fill_value=0)
    ctx.publish("WS4 papers", [
        {"type": "kpis", "headline": True, "items": [
            {"label": "trained papers", "value": int((both["split"] == "train").sum()), "fmt": "int"},
            {"label": "held-out papers", "value": int((both["split"] == "heldout").sum()), "fmt": "int",
             "hint": "knowledge-injection control: never trained on"},
            {"label": "eligible pool", "value": len(ok), "fmt": "int", "hint": "WS2 rules, minus WS2 / WS3 eval papers"},
            {"label": "domains", "value": int(dom.shape[0]), "fmt": "int"}]},
        {"type": "bar", "title": "Papers per domain", "horizontal": True, "fmt": "int", "stacked": True,
         "labels": [str(x) for x in dom.index],
         "series": [{"name": s, "data": dom[s].tolist()} for s in dom.columns]},
    ], summary=f"{(both['split'] == 'train').sum()} trained + {(both['split'] == 'heldout').sum()} held-out papers")
    return {"train": int((both["split"] == "train").sum()), "heldout": int((both["split"] == "heldout").sum())}


@stage("ws4", "probe", needs=("papers",), resources="gpu", env="llm", config_keys=("ws4.probe",),
       help="closed-book knowledge questions on trained and held-out papers (WS2 generator and checks)")
def run_probe(ctx: Context) -> dict:
    cfg = {**ctx.cfg.raw.get("ws2", {}).get("questions", {}), **ctx.section("ws4.probe")}
    papers = stage_table(ctx, "papers", "papers.parquet")
    items = [{"paper_key": p.paper_key, "source": p.full} for p in papers.itertuples() if (p.full or "").strip()]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"probe: {len(items)} papers with {gen['name']}")
    raw = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    src = {i["paper_key"]: i["source"] for i in items}
    split = dict(zip(papers["paper_key"], papers["split"], strict=True))
    per, n_copy, rows = int(cfg.get("per_paper", 3)), int(cfg.get("copy_ngram", 8)), []
    for r in raw:
        w = normalize_answer(src[r["paper_key"]]).split()
        grams = {tuple(w[i:i + n_copy]) for i in range(len(w) - n_copy + 1)}
        seen, kept = [], 0
        for j, q in enumerate(r["questions"]):
            reason = verify(q, src[r["paper_key"]], grams, cfg, seen)
            crit = q.get("_critic") or {}
            if reason == "ok":
                failed = [k for k in CRITIC_SCHEMA["required"] if crit.get(k) is not True]
                reason = f"critic:{failed[0]}" if failed else ("over_cap" if kept >= per else "ok")
            if reason == "ok":
                seen.append(_norm(q["question"]))
                kept += 1
            rows.append({"qid": f"{r['paper_key']}:p{j}", "paper_key": r["paper_key"], "split": split[r["paper_key"]],
                         "question": q["question"].strip(), "answer": (q.get("answer") or "").strip(),
                         "qtype": q.get("type"), "status": reason})
    allq = pd.DataFrame(rows)
    qs = allq[allq["status"] == "ok"].reset_index(drop=True)
    qs.to_parquet(ctx.out / "probe.parquet", index=False)
    allq.to_parquet(ctx.out / "all_generated.parquet", index=False)
    by = qs.groupby("split").agg(questions=("qid", "size"), papers=("paper_key", "nunique")).reset_index()
    ctx.publish("WS4 knowledge probe", [
        {"type": "kpis", "items": [
            {"label": "probe questions", "value": len(qs), "fmt": "int"},
            *[{"label": f"{r.split} questions", "value": int(r.questions), "fmt": "int"} for r in by.itertuples()],
            {"label": "kept share", "value": len(qs) / max(len(allq), 1), "fmt": "pct1"}]},
        {"type": "table", "title": "Probe questions per split", "rows": by.to_dict("records"),
         "columns": [{"key": "split", "label": "split"}, {"key": "questions", "label": "questions", "fmt": "int"},
                     {"key": "papers", "label": "papers", "fmt": "int"}]},
    ], summary=f"{len(qs)} probe questions")
    return {"questions": len(qs)}
