"""Stages `papers` (LOGIN node: internet) and `summarize` (GPU) of X2.

papers     arXiv papers submitted on or after `x2.cutoff.after` (later than the training cutoff / release of every
           evaluated model; see the table in configs/x2.yaml), stratified by primary category, newest first; full
           text from arxiv.org/html/<id> (LaTeX-rendered HTML; papers without it are skipped); the licence shown on
           the page is kept per paper. Respects the arXiv API terms (one request per 3 s).
summarize  Incremental-update demo: each new paper gets its KU with the WS5 pipeline (same summariser prompt, 18-field
           JSON schema, WS1 schema validation and verbatim gate), written by the WS5 teacher (Qwen3.5-27B-FP8) or a
           configured smaller model.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import time

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..runtime import Context, log, stage
from ..ws5.common import SYSTEM, ku_schema, overlap, parse_ku, schema_check, user_prompt
from .common import arxiv_listing, fetch_html_text, setting, stage_table


@stage("x2", "papers", config_keys=("x2.cutoff", "x2.papers"),
       help="arXiv papers after every model's cutoff, stratified by category, full text from arXiv HTML (login node)")
def run_papers(ctx: Context) -> dict:
    cfg, cut = ctx.section("x2.papers"), ctx.section("x2.cutoff")
    n = int(setting(ctx, "papers", cfg.get("n", 600)))
    cats = list(cfg.get("categories", ["cs"]))
    after, until = str(cut.get("after")), str(cut.get("until") or dt.date.today().isoformat())
    latest = max(str(m.get("release")) for m in (cut.get("models") or {}).values())
    if after <= latest:
        raise ValueError(f"cutoff {after} is not after the latest model release {latest}")
    windows = month_windows(after, until)                        # spread over the whole post-cutoff span
    want = -(-n // len(cats))                                     # papers per category
    per_window = max(1, -(-want * 3 // len(windows)))             # list 3x: some papers have no HTML / are too short
    rng = random.Random(int(cfg.get("seed", 0)))
    rows = []
    for c in cats:
        listing = [p for w0, w1 in windows
                   for p in arxiv_listing(c, w0, w1, per_window, float(cfg.get("api_delay_s", 3.1)))]
        rng.shuffle(listing)
        kept = 0
        for p in listing:
            if kept >= want:
                break
            text, lic = fetch_html_text(p["arxiv_id"])
            time.sleep(float(cfg.get("html_delay_s", 1.0)))
            if not text or len(text) < int(cfg.get("min_text_chars", 8000)):
                continue
            rows.append({**p, "paper_key": "x2:" + p["arxiv_id"], "stratum": c, "html_license": lic,
                         "full": text[:int(cfg.get("max_text_chars", 200000))], "text_chars": len(text), "rank": kept})
            kept += 1
        log(f"papers: {c}: {kept} papers with HTML full text")
    df = pd.DataFrame(rows).sort_values(["rank", "stratum"]).head(n).drop(columns="rank")   # round robin over strata
    df.to_parquet(ctx.out / "papers.parquet", index=False)
    by = df.groupby("stratum").size()
    ctx.publish("X2 post-cutoff papers", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(df), "fmt": "int"},
            {"label": "submitted on / after", "value": after, "hint": f"latest evaluated-model release: {latest}"},
            {"label": "categories", "value": int(by.size), "fmt": "int"},
            {"label": "median full-text characters", "value": float(df["text_chars"].median()), "fmt": "int"}]},
        {"type": "table", "title": "Model cutoffs the papers must post-date", "rows": [
            {"model": k, **{kk: str(vv) for kk, vv in v.items()}} for k, v in (cut.get("models") or {}).items()],
         "columns": [{"key": "model", "label": "model"}, {"key": "release", "label": "Hub release"},
                     {"key": "cutoff", "label": "stated data cutoff"}, {"key": "source", "label": "source"}]},
        {"type": "bar", "title": "Papers per arXiv category", "horizontal": True, "fmt": "int",
         "labels": list(by.index), "series": [{"name": "papers", "data": by.tolist()}]},
    ], summary=f"{len(df)} papers submitted from {after}")
    return {"papers": len(df), "after": after}


def month_windows(after: str, until: str) -> list[tuple[str, str]]:
    """[after, until] cut into calendar-month windows (YYYY-MM-DD pairs)."""
    a, u = dt.date.fromisoformat(after), dt.date.fromisoformat(until)
    out, cur = [], a
    while cur <= u:
        nxt = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        out.append((cur.isoformat(), min(nxt - dt.timedelta(days=1), u).isoformat()))
        cur = nxt
    return out


def _summarize(llm, tok, spec, items: list[dict], cfg: dict) -> dict:
    prompts = []
    for it in items:
        text, _, cut = truncate_tokens(tok, it["full"], int(cfg.get("source_tokens", 32000)))
        prompts.append(chat_prompt(tok, spec, SYSTEM, user_prompt(text, cut)))
    t0 = time.time()
    outs = llm.generate(prompts, json_params(ku_schema(), int(cfg.get("out_tokens", 6000))), use_tqdm=False)
    return {"seconds": time.time() - t0, "texts": [o.outputs[0].text if o.outputs else None for o in outs]}


@stage("x2", "summarize", needs=("papers",), resources="gpu", env="llm", config_keys=("x2.summarize",),
       help="incremental update: KUs for the new papers with the WS5 pipeline (prompt, schema, validation)")
def run_summarize(ctx: Context) -> dict:
    cfg = dict(ctx.section("x2.summarize"))
    papers = stage_table(ctx, "papers", "papers.parquet")
    name = setting(ctx, "summarizer", cfg.get("model", "qwen3.5-27b"))
    spec = model_spec(ctx.cfg, name)
    if name != "qwen3.5-27b":                                     # smaller summariser: fit its context
        cfg["source_tokens"] = min(int(cfg.get("source_tokens", 32000)), int(spec.get("max_model_len", 32768)) - 5000)
        cfg["out_tokens"] = min(int(cfg.get("out_tokens", 6000)), 4000)
    log(f"summarize: {len(papers)} new papers with {name}")
    res = with_model(spec, _summarize, papers[["paper_key", "full"]].to_dict("records"), cfg,
                     gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.94)) if name == "qwen3.5-27b" else 0.85)
    rows = []
    for p, t in zip(papers.itertuples(), res["texts"], strict=True):
        ku = parse_ku(t)
        sc, ov = schema_check(ku, ["summary_title", "executive_summary", "key_results", "claims"]), overlap(ku, p.full)
        rows.append({"paper_key": p.paper_key, "ku": json.dumps(ku, ensure_ascii=False) if ku else None,
                     "schema_valid": sc["valid"], "filled": sc["filled"], **ov, "summarizer": name})
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.out / "kus.parquet", index=False)
    ctx.publish("X2 KUs for the new papers (incremental update)", [{"type": "kpis", "items": [
        {"label": "KUs written", "value": int(df["ku"].notna().sum()), "fmt": "int"},
        {"label": "schema valid", "value": float(df["schema_valid"].mean()), "fmt": "pct1"},
        {"label": "longest copied span (median words)", "value": float(df["max_span"].median()), "fmt": "int"},
        {"label": "summariser", "value": name},
        {"label": "papers per GPU-hour", "value": 3600 * len(df) / max(res["seconds"], 1e-9), "fmt": "float"}]}],
        summary=f"{int(df['ku'].notna().sum())} KUs by {name}")
    return {"kus": int(df["ku"].notna().sum())}
