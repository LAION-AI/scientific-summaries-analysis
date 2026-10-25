"""Stage `ablation-gen` (WS2, GPU): the structure ablation's generated contexts.

Is it the schema or just "a summary" that helps? One local generator (ws2.ablation.generator; the released KUs come
from Gemini 2.0 Flash Lite, which we cannot run) reads the same source of every WS2 paper and writes
  ku_local     a KU in the 18-field schema (WS5 prompt + JSON schema), rendered like ku_clean;
  prose_local  an unstructured prose summary (no headings, lists or field names) that asks for the same content,
               length-matched to that paper's ku_local: generated with a word target, regenerated once with a
               corrected target if its token count is off by more than `tolerance`, and cut at a sentence
               boundary if it is still too long. Achieved lengths are recorded.
Same model, same source, same length: the only difference between the two arms is structure. The field-group
ablations of the released KU need no generation (common.ablation_texts, built in `contexts`).
"""
from __future__ import annotations

import json
import re

import pandas as pd

from ..data import fetch_records
from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..runtime import Context, log, stage
from ..ws1.faith.common import source_text
from ..ws5.common import SYSTEM as KU_SYSTEM
from ..ws5.common import ku_schema, parse_ku, user_prompt
from .common import load_stage_table, render_ku_clean

PROSE_SYSTEM = ("You write a summary of one scientific paper for researchers who will not read the paper. Write in "
                "your own words: do not copy sentences or long phrases from the paper. Report numbers exactly as the "
                "paper states them. Never invent results. Write continuous prose in paragraphs only: no headings, no "
                "lists or bullet points, no field names, no JSON, no markdown. Cover the research context and "
                "question, the methods and procedures, the key results with the numbers the paper reports, their "
                "interpretation, the limitations, the main claims with their evidence, and, if the paper reports "
                "them, data and code availability, robustness checks, ethical considerations and key figures.")
PROSE_USER = "PAPER{note}:\n<<<\n{text}\n>>>\n\nWrite the summary in about {words} words."
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def cut_at_sentence(tok, text: str, max_tokens: int) -> str:
    """The longest prefix of whole sentences within max_tokens (falls back to a token cut)."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = tok.decode(ids[:max_tokens])
    parts = _SENT_END.split(head)
    return " ".join(parts[:-1]).strip() if len(parts) > 1 else head


def _generate(llm, tok, spec, items: list[dict], cfg: dict) -> list[dict]:
    n_src, tol = int(cfg.get("source_tokens", 8000)), float(cfg.get("tolerance", 0.10))
    srcs = []
    for it in items:
        text, _, cut = truncate_tokens(tok, it["source"], n_src)
        srcs.append((text, cut))
    ku_out = llm.generate([chat_prompt(tok, spec, KU_SYSTEM, user_prompt(t, c)) for t, c in srcs],
                          json_params(ku_schema(), int(cfg.get("ku_max_tokens", 5000))), use_tqdm=False)
    rows = []
    for it, o in zip(items, ku_out, strict=True):
        ku = parse_ku(o.outputs[0].text if o.outputs else "")
        text = render_ku_clean({**ku, "claims": json.dumps(ku.get("claims") or [])})[0] if ku else ""
        rows.append({"paper_key": it["paper_key"], "ku_json": json.dumps(ku, ensure_ascii=False) if ku else None,
                     "ku_text": text, "ku_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
                     "ku_words": len(text.split())})

    def prose_round(targets: dict[int, int]) -> dict[int, str]:
        idx = sorted(targets)
        outs = llm.generate([chat_prompt(tok, spec, PROSE_SYSTEM, PROSE_USER.format(
            note=" (first part; the rest is cut)" if srcs[i][1] else "", text=srcs[i][0], words=targets[i]))
            for i in idx], [json_params(None, int(rows[i]["ku_tokens"] * 1.6) + 64) for i in idx], use_tqdm=False)
        return {i: (o.outputs[0].text.strip() if o.outputs else "") for i, o in zip(idx, outs, strict=True)}

    def ntok(t: str) -> int:
        return len(tok(t, add_special_tokens=False)["input_ids"])

    todo = {i: max(r["ku_words"], 50) for i, r in enumerate(rows) if r["ku_tokens"] > 0}
    first = prose_round(todo)
    best = {i: first[i] for i in todo}
    redo = {}
    for i, t in first.items():
        ratio = ntok(t) / rows[i]["ku_tokens"]
        if abs(ratio - 1) > tol and ratio > 0:
            redo[i] = max(int(todo[i] / ratio), 50)                   # corrected word target
    if redo:
        second = prose_round(redo)
        for i, t in second.items():
            if abs(ntok(t) / rows[i]["ku_tokens"] - 1) < abs(ntok(best[i]) / rows[i]["ku_tokens"] - 1):
                best[i] = t
    for i, r in enumerate(rows):
        t = best.get(i, "")
        if t and ntok(t) > r["ku_tokens"] * (1 + tol):
            t = cut_at_sentence(tok, t, int(r["ku_tokens"] * (1 + tol)))
        r.update(prose_text=t, prose_tokens=ntok(t) if t else 0, regenerated=i in redo,
                 length_ratio=(ntok(t) / r["ku_tokens"]) if t and r["ku_tokens"] else None)
    return rows


@stage("ws2", "ablation-gen", needs=("papers",), resources="gpu", env="llm", config_keys=("ws2.ablation",),
       help="structure ablation: one local model writes a KU and a length-matched prose summary of every WS2 paper")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.ablation")
    papers = load_stage_table(ctx, "papers", "papers.parquet")
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)), ["text_raw", "text_sanitized"])
    items = [{"paper_key": p.paper_key, "source": source_text(recs.get((p.file, int(p.row))) or {})}
             for p in papers.itertuples()]
    items = [i for i in items if i["source"].strip()]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"ablation-gen: {len(items)} papers with {gen['name']}")
    rows = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    df = pd.DataFrame(rows).assign(generator=gen["name"])
    df.to_parquet(ctx.out / "generated.parquet", index=False)
    ok = df[df["ku_tokens"] > 0]
    within = ((ok["length_ratio"] - 1).abs() <= float(cfg.get("tolerance", 0.10))).mean() if len(ok) else 0.0
    ctx.publish("WS2 structure ablation: generated contexts", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(df), "fmt": "int"},
            {"label": "valid local KUs", "value": len(ok), "fmt": "int"},
            {"label": "prose within ±10% of KU tokens", "value": float(within), "fmt": "pct1"},
            {"label": "median prose / KU tokens", "value": float(ok["length_ratio"].median()) if len(ok) else None, "fmt": "float"},
            {"label": "regenerated for length", "value": int(ok["regenerated"].sum()), "fmt": "int"},
            {"label": "generator", "value": gen["name"]}]},
        {"type": "table", "title": "Length matching per paper (tokens of the generator's tokenizer)",
         "columns": [{"key": "paper_key", "label": "paper"}, {"key": "ku_tokens", "label": "KU tokens", "fmt": "int"},
                     {"key": "prose_tokens", "label": "prose tokens", "fmt": "int"},
                     {"key": "length_ratio", "label": "prose / KU", "fmt": "float"},
                     {"key": "regenerated", "label": "regenerated"}],
         "rows": df[["paper_key", "ku_tokens", "prose_tokens", "length_ratio", "regenerated"]].to_dict("records")},
    ], summary=f"{len(ok)} KU / prose pairs; {within:.0%} length-matched within 10%")
    return {"papers": len(df), "valid": len(ok), "within_tolerance": float(within)}
