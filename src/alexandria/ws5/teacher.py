"""Stages `papers`, `teacher` and `validate` (WS5).

papers    Open-access papers only (permissive sources such as arXiv, or OpenAlex open access), so teacher outputs
          can be released; readable full text; stratified by discipline; split BY PAPER into train / dev / test
          (no overlap), with `holdout_domains` whole disciplines in test only (generalisation to unseen fields).
teacher   The teacher writes one KU per paper (train, dev and test; test outputs are the ceiling baseline).
          provider anthropic: Claude Opus 5.5 through the Message Batches API (50% price; adaptive thinking, effort
          from config, the 18-field JSON schema as structured output, the shared system prompt cached); run on a login
          node. Refuses to send more than `compliance.max_unreviewed_requests` requests until
          `compliance.terms_checked` is set. provider vllm (default): an open-weight teacher on our GPUs
          (Qwen3.5-27B-FP8, Apache-2.0: its outputs may train other models and be released), no API key needed.
validate  Gates every teacher KU before training: WS1 schema validation (load_summary, required fields, no extra
          keys), a verbatim gate (fail at a >= `fail_span`-word span copied from the paper), and a human spot-check
          packet of `spot_check` random valid training KUs (CSV + Markdown with links). Only valid train KUs are
          training data.
"""
from __future__ import annotations

import json
import random
import time

import pandas as pd

from ..data import fetch_records
from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..papers import with_paper_info
from ..runtime import Context, log, stage
from ..ws1.faith.common import features_table, paper_key, source_text, stratified_pick
from .common import SYSTEM, ku_schema, overlap, parse_ku, schema_check, setting, stage_table, user_prompt


@stage("ws5", "papers", config_keys=("ws5.papers",),
       help="open-access papers, split by paper into train / dev / test (+ held-out disciplines in test only)")
def run_papers(ctx: Context) -> dict:
    cfg = ctx.section("ws5.papers")
    n = {s: int(setting(ctx, s, cfg.get(f"n_{s}", 0))) for s in ("train", "dev", "test")}
    f = features_table(ctx, ["file", "row", "group", "domain", "input_type", "has_text", "source_garbled",
                             "wrong_paper_suspect", "text_length", "oa_is_open_access", "ku_needs_regeneration",
                             "dup_drop_any"])
    oa = f["group"].astype(str).str.startswith(tuple(cfg.get("open_access_groups", ["arxiv/"]))) | \
        f["oa_is_open_access"].fillna(False).astype(bool)
    ok = f[oa & f["has_text"].fillna(False) & ~f["source_garbled"].fillna(False)
           & ~f["wrong_paper_suspect"].fillna(False) & (f["input_type"] == "full_text")
           & (f["text_length"].fillna(0) >= int(cfg.get("min_source_chars", 5000)))
           & ~f["ku_needs_regeneration"].fillna(False) & ~f["dup_drop_any"].fillna(False)].copy()
    ok["paper_key"] = [paper_key(a, b) for a, b in zip(ok["file"], ok["row"], strict=True)]
    seed, strata = int(cfg.get("seed", 0)), list(cfg.get("strata", ["domain"]))
    rng = random.Random(seed)
    doms = sorted(ok["domain"].dropna().unique())
    hd = cfg.get("holdout_domains", 1)                   # a count (random disciplines) or a list of discipline names
    held = sorted(d for d in hd if d in doms) if isinstance(hd, list) else \
        sorted(rng.sample(doms, min(int(hd), max(len(doms) - 1, 0))))
    pool_held, pool = ok[ok["domain"].isin(held)], ok[~ok["domain"].isin(held)]
    test_held = stratified_pick(pool_held, min(len(pool_held), n["test"] // 4), strata, seed)
    test_in = stratified_pick(pool, n["test"] - len(test_held), strata, seed)
    rest = pool[~pool["paper_key"].isin(set(test_in["paper_key"]))]
    dev = stratified_pick(rest, n["dev"], strata, seed + 1)
    train = stratified_pick(rest[~rest["paper_key"].isin(set(dev["paper_key"]))], n["train"], strata, seed + 2)
    parts = [train.assign(split="train"), dev.assign(split="dev"), test_in.assign(split="test", heldout_domain=False),
             test_held.assign(split="test", heldout_domain=True)]
    papers = pd.concat(parts, ignore_index=True)
    papers["heldout_domain"] = papers["heldout_domain"].fillna(False).astype(bool)
    assert papers["paper_key"].is_unique, "a paper is in two splits"
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)), ["text_raw", "text_sanitized"])
    papers["source"] = [source_text(recs.get((a, int(b))) or {}) for a, b in zip(papers["file"], papers["row"], strict=True)]
    papers = with_paper_info(ctx, papers)
    papers.to_parquet(ctx.out / "papers.parquet", index=False)
    by = papers.groupby("split").size()
    ctx.publish("WS5 papers", [
        {"type": "kpis", "items": [
            *[{"label": f"{s} papers", "value": int(by.get(s, 0)), "fmt": "int"} for s in ("train", "dev", "test")],
            {"label": "held-out disciplines", "value": ", ".join(held) or "none", "hint": "in test only"},
            {"label": "open-access pool", "value": len(ok), "fmt": "int"}]},
        {"type": "bar", "title": "Papers per discipline and split", "horizontal": True, "fmt": "int",
         "labels": sorted(papers["domain"].astype(str).unique()),
         "series": [{"name": s, "data": [int(((papers["split"] == s) & (papers["domain"].astype(str) == d)).sum())
                                          for d in sorted(papers["domain"].astype(str).unique())]}
                    for s in ("train", "dev", "test")]},
    ], summary=f"{by.get('train', 0)} / {by.get('dev', 0)} / {by.get('test', 0)} papers (train / dev / test)")
    return {k: int(v) for k, v in by.items()}


# ---------------------------------------------------------------- teacher
def _anthropic_batch(ctx: Context, cfg: dict, items: list[dict]) -> list[dict]:
    """Claude teacher through the Message Batches API (login node). Results are keyed by custom_id."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    comp = ctx.section("ws5.compliance")
    if len(items) > int(comp.get("max_unreviewed_requests", 50)) and not comp.get("terms_checked"):
        raise PermissionError(
            f"{len(items)} teacher requests: check the teacher's terms on training other models with its outputs and "
            "on releasing data / weights (CC-BY) first, then set ws5.compliance.terms_checked=true")
    client = anthropic.Anthropic()
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]   # shared prefix, cached
    reqs = [Request(custom_id=it["paper_key"], params=MessageCreateParamsNonStreaming(
        model=cfg.get("model", "claude-opus-5-5"), max_tokens=int(cfg.get("max_tokens", 32000)), system=system,
        thinking={"type": "adaptive"}, output_config={"effort": cfg.get("effort", "high"),
                                                      "format": {"type": "json_schema", "schema": ku_schema()}},
        messages=[{"role": "user", "content": user_prompt(it["source"][:int(cfg.get("max_source_chars", 200000))],
                                                         len(it["source"]) > int(cfg.get("max_source_chars", 200000)))}]))
        for it in items]
    batch = client.messages.batches.create(requests=reqs)
    ctx.write_json("anthropic_batch.json", {"id": batch.id, "requests": len(reqs)})
    log(f"teacher: batch {batch.id} with {len(reqs)} requests")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log(f"teacher: {b.processing_status}, processing {b.request_counts.processing}")
        time.sleep(int(cfg.get("poll_seconds", 60)))
    out = {}
    for r in client.messages.batches.results(batch.id):
        row = {"paper_key": r.custom_id, "status": r.result.type, "text": None, "tokens_in": None, "tokens_out": None}
        if r.result.type == "succeeded":
            m = r.result.message
            row.update(text=next((c.text for c in m.content if c.type == "text"), None), stop_reason=m.stop_reason,
                       tokens_in=m.usage.input_tokens + (m.usage.cache_read_input_tokens or 0)
                       + (m.usage.cache_creation_input_tokens or 0), tokens_out=m.usage.output_tokens)
        out[r.custom_id] = row
    return [out.get(it["paper_key"], {"paper_key": it["paper_key"], "status": "missing"}) for it in items]


def _vllm_teacher(llm, tok, spec, items: list[dict], cfg: dict) -> list[dict]:
    prompts = []
    for it in items:
        text, n, cut = truncate_tokens(tok, it["source"], int(cfg.get("source_tokens", 32000)))
        prompts.append(chat_prompt(tok, spec, SYSTEM, user_prompt(text, cut)))
    t0 = time.time()
    outs = llm.generate(prompts, json_params(ku_schema(), int(cfg.get("out_tokens", 6000))), use_tqdm=False)
    sec = time.time() - t0
    return [{"paper_key": it["paper_key"], "status": "succeeded", "text": o.outputs[0].text if o.outputs else None,
             "tokens_in": len(o.prompt_token_ids or []), "tokens_out": len(o.outputs[0].token_ids) if o.outputs else 0,
             "gpu_seconds": sec / len(items)} for it, o in zip(items, outs, strict=True)]


@stage("ws5", "teacher", needs=("papers",), resources="gpu", env="llm", config_keys=("ws5.teacher",),
       help="teacher KUs for train / dev / test papers (open-weight Qwen3.5-27B on a 40 GB slice, or Claude batches)")
def run_teacher(ctx: Context) -> dict:
    cfg = ctx.section("ws5.teacher")
    provider = setting(ctx, "teacher_provider", cfg.get("provider", "vllm"))
    papers = stage_table(ctx, "papers", "papers.parquet")
    items = [{"paper_key": p.paper_key, "source": p.source} for p in papers.itertuples()]
    if provider == "anthropic":
        rows, name = _anthropic_batch(ctx, cfg, items), cfg.get("model", "claude-opus-5-5")
    else:
        spec = model_spec(ctx.cfg, setting(ctx, "teacher_model", cfg.get("local_model", "qwen3.5-27b")))
        cfg = {**cfg, "source_tokens": setting(ctx, "teacher_source_tokens", cfg.get("source_tokens", 32000)),
               "out_tokens": setting(ctx, "teacher_out_tokens", cfg.get("out_tokens", 6000))}
        extra = dict(cfg.get("vllm_kwargs") or {})
        if cfg.get("cpu_offload_gb"):          # weights that do not fit the GPU slice live in CPU memory (slower)
            extra["cpu_offload_gb"] = float(cfg["cpu_offload_gb"])
        spec = {**spec, "vllm_kwargs": {**(spec.get("vllm_kwargs") or {}), **extra}}
        log(f"teacher: {spec['name']} ({spec['hf']}, licence {spec.get('license', '?')}) for {len(items)} papers"
            + (f", {cfg['cpu_offload_gb']} GB offloaded to CPU" if cfg.get("cpu_offload_gb") else ""))
        rows = with_model(spec, _vllm_teacher, items, cfg, gpu_memory_utilization=float(
            setting(ctx, "teacher_gpu_memory", cfg.get("gpu_memory_utilization", 0.85))))
        name = spec["name"]
    df = pd.DataFrame(rows).merge(papers[["paper_key", "split"]], on="paper_key")
    df["teacher"] = name
    df.to_parquet(ctx.out / "teacher.parquet", index=False)
    ok = int((df["status"] == "succeeded").sum())
    ctx.publish("WS5 teacher KUs", [{"type": "kpis", "items": [
        {"label": "teacher", "value": name},
        {"label": "KUs written", "value": ok, "fmt": "int"},
        {"label": "failed / missing", "value": len(df) - ok, "fmt": "int"},
        {"label": "output tokens per paper (median)", "value": float(df["tokens_out"].median()), "fmt": "int"}]}],
        summary=f"{ok} teacher KUs by {name}")
    return {"teacher": name, "succeeded": ok}


@stage("ws5", "validate", needs=("teacher",), config_keys=("ws5.validate",),
       help="gate teacher KUs: WS1 schema validation, verbatim-span gate, human spot-check packet")
def run_validate(ctx: Context) -> dict:
    cfg = ctx.section("ws5.validate")
    papers = stage_table(ctx, "papers", "papers.parquet").set_index("paper_key")
    t = stage_table(ctx, "teacher", "teacher.parquet")
    rows = []
    for r in t.itertuples():
        ku = parse_ku(r.text)
        sc = schema_check(ku, list(cfg.get("required_fields", [])))
        ov = overlap(ku, papers.at[r.paper_key, "source"], int(cfg.get("span_k", 8)))
        verbatim_ok = ov["max_span"] < int(cfg.get("fail_span", 12))
        rows.append({"paper_key": r.paper_key, "split": r.split, "schema_valid": sc["valid"], "filled": sc["filled"],
                     "issues": sc["issues"], **ov, "verbatim_ok": verbatim_ok,
                     "passed": bool(sc["valid"] and verbatim_ok), "ku": json.dumps(ku, ensure_ascii=False) if ku else None})
    v = pd.DataFrame(rows)
    v.to_parquet(ctx.out / "validated.parquet", index=False)
    k = int(setting(ctx, "spot_check", cfg.get("spot_check", 50)))
    spot = v[(v["split"] == "train") & v["passed"]].sample(n=min(k, int(((v["split"] == "train") & v["passed"]).sum())),
                                                               random_state=0)
    spot = spot.merge(papers[["title", "links"]].reset_index(), on="paper_key", how="left")
    spot[["paper_key", "title", "links", "max_span"]].assign(faithful="", coverage="", style="", notes="") \
        .to_csv(ctx.out / "spot_check.csv", index=False)
    with open(ctx.out / "spot_check.md", "w") as fh:
        fh.write("# WS5 teacher spot check\nFor each KU: is every statement supported by the paper (faithful)? "
                 "Are the main results there (coverage)? Own words (style)? Fill the columns in spot_check.csv.\n\n")
        for s in spot.itertuples():
            fh.write(f"## {s.title} ({s.paper_key})\n\n```json\n{s.ku}\n```\n\n")
    by = v.groupby("split").agg(kus=("paper_key", "size"), schema_valid=("schema_valid", "mean"),
                                verbatim_ok=("verbatim_ok", "mean"), passed=("passed", "mean"),
                                max_span=("max_span", "median")).reset_index()
    ctx.publish("WS5 teacher validation", [
        {"type": "kpis", "items": [
            {"label": "train KUs passing every gate", "value": int(((v["split"] == "train") & v["passed"]).sum()), "fmt": "int"},
            {"label": "schema valid", "value": float(v["schema_valid"].mean()), "fmt": "pct1"},
            {"label": f"verbatim gate (< {cfg.get('fail_span', 12)} words)", "value": float(v["verbatim_ok"].mean()), "fmt": "pct1"},
            {"label": "spot-check packet", "value": len(spot), "fmt": "int", "hint": "spot_check.csv / .md for a human"}]},
        {"type": "table", "title": "Validation per split", "rows": by.to_dict("records"), "columns": [
            {"key": "split", "label": "split"}, {"key": "kus", "label": "KUs", "fmt": "int"},
            {"key": "schema_valid", "label": "schema valid", "fmt": "pct1"},
            {"key": "verbatim_ok", "label": "verbatim gate", "fmt": "pct1"}, {"key": "passed", "label": "passed", "fmt": "pct1"},
            {"key": "max_span", "label": "longest copied span (median words)", "fmt": "int"}]},
    ], summary=f"{int(v['passed'].sum())} of {len(v)} teacher KUs pass")
    return {"passed": int(v["passed"].sum()), "total": len(v)}
