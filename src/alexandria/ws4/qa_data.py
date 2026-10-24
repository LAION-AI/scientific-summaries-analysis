"""Stage `qa-data` (WS4, GPU): instruction / QA pairs generated FROM each representation.

The same generator (ws4.qa_data.generator, Qwen3-8B), the same prompt and the same number of pairs per paper for
every condition: from the prose summary (`summary`), the JSON KU (`ku`), the full text (`full`), and a rotated other
surface form of the KU (`diverse`: YAML / HTML / bullet Q&A). A pair is kept only if its evidence quote is found in
THAT representation, the answer is short and the question is self-contained (names the study / system, no "this
paper"), so pairs are verifiable (exact / numeric match) and usable as GRPO prompts too. Per paper, every condition
keeps the same count: min over conditions of the valid pairs, capped at `pairs_per_paper`.

Also written: `qa_generic.parquet` (WebInstructSub science QA, as many pairs as one paper condition: the generic SFT
control) and `chat_replay.parquet` (Tulu-3 SFT prompt / completion pairs: the replay mixed into every QA run).
"""
from __future__ import annotations

import json

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, parse_json, truncate_tokens, with_model
from ..runtime import Context, log, stage
from .common import PAPER_CONDITIONS, load_resource, render, stage_table

SYSTEM = """You write question-answer pairs for training a model on the content of one scientific study.
You see one representation of the study (a summary, a structured record, or the full text).
Rules for every pair:
1. The question asks about a specific result, number, method, setting, comparison or limitation of THIS study.
2. Self-contained: never write "this paper", "the authors", "the study" or "the proposed method"; name the system, method, material or setting concretely.
3. The answer is short and verifiable: a number with its unit, a name, yes/no, or a phrase of at most 10 words.
4. "evidence" is an exact quote (at most 40 words) from the text you were given that contains the answer.
Answer with JSON only."""
USER = "STUDY ({what}):\n<<<\n{text}\n>>>\n\nWrite {k} question-answer pairs."
WHAT = {"summary": "summary", "ku": "structured record (JSON)", "full": "full text", "diverse": "structured record"}
DIVERSE_FORMATS = ("yaml", "html", "qa_bullets")


def schema(k: int) -> dict:
    return {"type": "object", "properties": {"pairs": {"type": "array", "minItems": 1, "maxItems": k, "items": {
        "type": "object", "properties": {"question": {"type": "string", "maxLength": 300},
                                         "answer": {"type": "string", "maxLength": 80},
                                         "evidence": {"type": "string", "maxLength": 400}},
        "required": ["question", "answer", "evidence"], "additionalProperties": False}}},
        "required": ["pairs"], "additionalProperties": False}


def representation(cond: str, paper: dict, i: int) -> str:
    if cond == "summary":
        return render(paper["ku"], "prose")
    if cond == "ku":
        return render(paper["ku"], "json")
    if cond == "full":
        return paper["full"]
    return render(paper["ku"], DIVERSE_FORMATS[i % len(DIVERSE_FORMATS)])


def valid(pair: dict, text: str, min_match: float) -> bool:
    import re

    from rapidfuzz import fuzz
    q, a, ev = (pair.get(k) or "" for k in ("question", "answer", "evidence"))
    if not q or not a or len(a.split()) > 12 or re.search(r"\b(this (paper|study)|the authors)\b", q, re.I):
        return False
    norm = " ".join(text.lower().split())
    return len(ev) >= 10 and fuzz.partial_ratio(" ".join(ev.lower().split()), norm) >= min_match


def _generate(llm, tok, spec, items: list[dict], cfg: dict) -> list[list[dict]]:
    k = int(cfg.get("generate_per_paper", 14))
    max_src = min(int(cfg.get("max_source_tokens", 12000)),
                  int(spec.get("max_model_len", 32768)) - int(cfg.get("max_tokens", 3000)) - 512)
    prompts = []
    for it in items:
        text, _, _ = truncate_tokens(tok, it["text"], max_src)
        it["shown"] = text
        prompts.append(chat_prompt(tok, spec, SYSTEM, USER.format(what=WHAT[it["condition"]], text=text, k=k)))
    outs = llm.generate(prompts, json_params(schema(k), int(cfg.get("max_tokens", 3000)),
                                             float(cfg.get("temperature", 0.0))), use_tqdm=False)
    return [[p for p in ((parse_json(o.outputs[0].text if o.outputs else "") or {}).get("pairs") or [])
             if isinstance(p, dict)] for o in outs]


@stage("ws4", "qa-data", needs=("papers",), resources="gpu", env="llm", config_keys=("ws4.qa_data",),
       help="QA pairs generated from each representation (same generator, same count per paper) + controls")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws4.qa_data")
    conds = [c for c in ctx.section("ws4").get("conditions", []) if c in PAPER_CONDITIONS]
    papers = stage_table(ctx, "papers", "papers.parquet")
    papers = papers[papers["split"] == "train"].reset_index(drop=True)
    plist = [{"paper_key": p.paper_key, "ku": json.loads(p.ku), "full": p.full} for p in papers.itertuples()]
    items = [{"paper_key": p["paper_key"], "condition": c, "text": representation(c, p, i)}
             for i, p in enumerate(plist) for c in conds]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"qa-data: {len(items)} prompts ({len(plist)} papers x {len(conds)} conditions) with {gen['name']}")
    outs = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    min_match, cap = float(cfg.get("min_evidence_match", 85)), int(cfg.get("pairs_per_paper", 8))
    good: dict[tuple[str, str], list[dict]] = {}
    all_rows = []
    for it, pairs in zip(items, outs, strict=True):
        ok = [p for p in pairs if valid(p, it["text"], min_match)]
        good[(it["paper_key"], it["condition"])] = ok
        all_rows.append({"paper_key": it["paper_key"], "condition": it["condition"], "generated": len(pairs),
                         "valid": len(ok)})
    rows = []
    for p in plist:
        n = min([cap] + [len(good[(p["paper_key"], c)]) for c in conds])
        for c in conds:
            for j, pr in enumerate(good[(p["paper_key"], c)][:n]):
                rows.append({"pid": f"{p['paper_key']}:{c}:{j}", "paper_key": p["paper_key"], "condition": c,
                             "question": pr["question"].strip(), "answer": pr["answer"].strip(),
                             "evidence": pr["evidence"], "generator": gen["name"]})
    qa = pd.DataFrame(rows)
    for c in conds:
        qa[qa["condition"] == c].to_parquet(ctx.out / f"qa_{c}.parquet", index=False)
    per_cond = len(qa) // max(len(conds), 1)
    r = ctx.cfg.raw["ws4_resources"]["webinstruct"]
    gen_df = load_resource(ctx, "webinstruct", max_rows=max(per_cond * 3, 1000)).sample(frac=1.0, random_state=0)
    gen_df = gen_df.rename(columns={r["question"]: "question", r["answer"]: "answer"})[["question", "answer"]].head(per_cond)
    gen_df.assign(condition="generic", paper_key=None).to_parquet(ctx.out / "qa_generic.parquet", index=False)
    share = float(ctx.section("ws4.data").get("replay_share", 0.25))
    n_rep = int(per_cond * share / (1 - share))
    tulu = load_resource(ctx, "tulu3", max_rows=max((n_rep + per_cond) * 3, 1000)).sample(frac=1.0, random_state=0)
    rep = []
    for msgs in tulu["messages"]:
        msgs = [dict(m) for m in msgs]
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
            rep.append({"prompt": json.dumps(msgs[:-1], ensure_ascii=False), "completion": msgs[-1]["content"]})
        if len(rep) >= n_rep + per_cond:          # QA runs use the first n_rep; `none` uses all (same example count)
            break
    pd.DataFrame(rep).to_parquet(ctx.out / "chat_replay.parquet", index=False)
    stats = pd.DataFrame(all_rows).groupby("condition")[["generated", "valid"]].sum().reset_index()
    stats["kept"] = stats["condition"].map(qa.groupby("condition").size()).fillna(0).astype(int)
    ctx.publish("WS4 QA training data", [
        {"type": "kpis", "items": [
            {"label": "pairs per condition", "value": per_cond, "fmt": "int", "hint": "equal per paper across conditions"},
            {"label": "papers", "value": len(plist), "fmt": "int"},
            {"label": "generic control pairs", "value": len(gen_df), "fmt": "int", "hint": "WebInstructSub"},
            {"label": "replay chat pairs", "value": len(rep), "fmt": "int", "hint": "Tulu-3 SFT mixture"},
            {"label": "generator", "value": gen["name"]}]},
        {"type": "table", "title": "Generated, valid and kept pairs per condition", "rows": stats.to_dict("records"),
         "columns": [{"key": "condition", "label": "condition"}, {"key": "generated", "label": "generated", "fmt": "int"},
                     {"key": "valid", "label": "valid", "fmt": "int"}, {"key": "kept", "label": "kept (count-matched)", "fmt": "int"}]},
    ], summary=f"{per_cond} pairs per condition")
    return {"pairs_per_condition": per_cond, "conditions": conds}
