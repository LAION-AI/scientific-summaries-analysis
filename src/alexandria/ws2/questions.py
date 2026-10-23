"""Stage `questions` (WS2, GPU): the QA bench, generated from each paper's source text.

A generator LLM (ws2.questions.generator, default Qwen3-8B; a different family from most answerers) writes up to
`per_paper` short-answer questions per paper as JSON (vLLM structured outputs), each with a gold answer, a type and an
exact evidence quote. Questions must be
  * specific to the paper (results, numbers, methods, conditions, limitations), not general knowledge;
  * self-contained: they name the system / method / setting instead of "this paper", so the same question also
    works in open-corpus mode (WS3: retrieve the paper first);
  * paraphrased: no run of >= 5 consecutive words copied from the paper (reduces the lexical advantage of the
    full-text arm, a known bias of questions generated from the source).
Kept only if verified, in two passes (the outcome of every generated question is stored):
  1. rules: the evidence quote is found in the source (fuzzy partial match >= min_evidence_match), the answer is
     short and is not a reference to notation / numbering (equation, figure, table, section numbers), the question
     does not copy a >= copy_ngram-word run from the source and is not a near-duplicate of another one;
  2. an LLM critic (same model, JSON verdict, sees question + answer + evidence, not the paper) checks that the
     question identifies the study, has one unambiguous answer, is not trivial, and cannot be answered from general
     textbook knowledge — the answerability / specificity filters used for generated science QA benchmarks.
The prompt asks for a mix of question types (findings and comparisons, methods and settings, limitations) with at
most two bare-number answers per paper, so the bench is not dominated by numeric lookups.
"""
from __future__ import annotations

import re

import pandas as pd

from ..data import fetch_records
from ..llm import chat_prompt, json_params, model_spec, parse_json, truncate_tokens, with_model
from ..qa import normalize_answer
from ..runtime import Context, log, stage
from ..ws1.faith.common import source_text
from .common import load_stage_table

QTYPES = ["numeric_result", "finding", "method", "setup", "comparison", "limitation", "definition"]

SYSTEM = """You write exam questions that test whether a reader knows the specific content of one scientific paper.

Rules for every question:
1. It can only be answered with details from this paper (a result, number, method, dataset, condition, comparison or limitation stated in it), not from general knowledge of the field.
2. It is self-contained: never write "this paper", "the authors", "the study" or "the proposed method". Name the topic, system, method or setting concretely enough that someone searching the literature could find the paper.
3. Paraphrase: do not copy any run of 5 or more consecutive words from the paper.
4. The answer is short and unambiguous: a number with its unit, a name, or a phrase of at most 12 words.
5. "evidence" is an exact quote from the paper (at most 40 words) that contains the answer.
6. Never ask about notation, symbols, or the numbering of equations, figures, tables, sections or references, and never about author names.
Mix of questions: about half on findings and comparisons (what was found, which is better, what effect or trend), the rest on methods, settings and limitations or conditions. At most two questions whose answer is a bare number. Answer with JSON only."""

USER = """PAPER ({completeness}):
<<<
{source}
>>>

Write {k} questions."""


def schema(k: int) -> dict:
    return {"type": "object", "properties": {"questions": {
        "type": "array", "minItems": 1, "maxItems": k, "items": {
            "type": "object",
            "properties": {"question": {"type": "string", "maxLength": 400},
                           "answer": {"type": "string", "maxLength": 120},
                           "type": {"type": "string", "enum": QTYPES},
                           "evidence": {"type": "string", "maxLength": 400}},
            "required": ["question", "answer", "type", "evidence"], "additionalProperties": False}}},
        "required": ["questions"], "additionalProperties": False}


_WS = re.compile(r"\s+")
_TRIVIAL = re.compile(r"^\s*(eq(uation)?s?\.?|fig(ure)?s?\.?|tab(le)?s?\.?|sec(tion)?s?\.?|ref(erence)?s?\.?|"
                      r"appendix|theorem|lemma|corollary|proposition|definition)\s*[\dIVXivx(\[]", re.I)

CRITIC_SYSTEM = """You review exam questions about scientific papers before they enter a benchmark. You see the question, the gold answer and the evidence quote from the paper, not the paper itself.
Judge each criterion:
- identifies_study: the question names the system, method, material, dataset or setting concretely enough that a researcher searching the literature could find the study it is about.
- single_answer: the question has one unambiguous correct answer, and the gold answer is it.
- not_trivial: answering requires knowing the paper's content, not notation, symbols or the numbering of equations, figures, tables or sections.
- needs_paper: a well-read expert could NOT answer it from general textbook knowledge alone.
Answer with JSON only."""
CRITIC_SCHEMA = {"type": "object", "properties": {k: {"type": "boolean"} for k in
                 ("identifies_study", "single_answer", "not_trivial", "needs_paper")},
                 "required": ["identifies_study", "single_answer", "not_trivial", "needs_paper"],
                 "additionalProperties": False}


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower()).strip()


def longest_copied_run(question: str, source_words: set[tuple[str, ...]], n: int) -> bool:
    w = normalize_answer(question).split()
    return any(tuple(w[i:i + n]) in source_words for i in range(len(w) - n + 1))


def verify(q: dict, source: str, src_ngrams: set, cfg: dict, seen: list[str]) -> str:
    """'ok' or the reason a generated question is dropped."""
    from rapidfuzz import fuzz
    if len((q.get("answer") or "").split()) > int(cfg.get("max_answer_words", 15)):
        return "answer_too_long"
    if _TRIVIAL.search(q.get("answer") or ""):
        return "trivial_reference"
    if (q.get("question") or "").lower().count("this paper") or re.search(r"\b(the authors|this study)\b",
                                                                         q.get("question", ""), re.I):
        return "not_self_contained"
    ev = _norm(q.get("evidence"))
    if len(ev) < 10 or fuzz.partial_ratio(ev, _norm(source)) < float(cfg.get("min_evidence_match", 90)):
        return "evidence_not_in_paper"
    if longest_copied_run(q["question"], src_ngrams, int(cfg.get("copy_ngram", 8))):
        return "copies_paper_text"
    if any(fuzz.token_set_ratio(_norm(q["question"]), s) >= 90 for s in seen):
        return "duplicate"
    return "ok"


def _generate(llm, tok, spec, papers: list[dict], cfg: dict) -> list[dict]:
    k = int(cfg.get("generate_per_paper", 8))
    max_src = min(int(cfg.get("max_source_tokens", 24000)),
                  int(spec.get("max_model_len", 32768)) - int(cfg.get("max_tokens", 3000)) - 1024)
    prompts = []
    for p in papers:
        src, _, cut = truncate_tokens(tok, p["source"], max_src)
        prompts.append(chat_prompt(tok, spec, SYSTEM, USER.format(
            completeness=f"first {max_src} tokens" if cut else "complete", source=src, k=k)))
    outs = llm.generate(prompts, json_params(schema(k), int(cfg.get("max_tokens", 3000)),
                                             float(cfg.get("temperature", 0.0))), use_tqdm=False)
    gen = [{"paper_key": p["paper_key"], "questions": [q for q in ((parse_json(o.outputs[0].text if o.outputs else "")
                                                                    or {}).get("questions") or [])
                                                       if isinstance(q, dict) and q.get("question")]}
           for p, o in zip(papers, outs, strict=True)]
    # critic pass over every candidate (cheap: no paper text in the prompt)
    flat = [(gi, qi, q) for gi, g in enumerate(gen) for qi, q in enumerate(g["questions"])]
    crit = llm.generate([chat_prompt(tok, spec, CRITIC_SYSTEM, f"Question: {q['question']}\nGold answer: "
                                     f"{q.get('answer', '')}\nEvidence: {q.get('evidence', '')}") for _, _, q in flat],
                        json_params(CRITIC_SCHEMA, 120), use_tqdm=False)
    for (gi, qi, _), o in zip(flat, crit, strict=True):
        gen[gi]["questions"][qi]["_critic"] = parse_json(o.outputs[0].text if o.outputs else "") or {}
    return gen


@stage("ws2", "questions", needs=("papers",), resources="gpu", env="llm", config_keys=("ws2.questions",),
       help="QA bench: verified, self-contained, paraphrased short-answer questions per paper (LLM + checks)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.questions")
    papers = load_stage_table(ctx, "papers", "papers.parquet")
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)),
                         ["paper_id", "text_raw", "text_sanitized"])
    items = [{"paper_key": p.paper_key, "source": source_text(recs.get((p.file, int(p.row))) or {})}
             for p in papers.itertuples()]
    items = [i for i in items if i["source"].strip()]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"questions: {len(items)} papers with {gen['name']}")
    raw = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    use_critic = bool(cfg.get("critic", True))
    by_key = {i["paper_key"]: i["source"] for i in items}
    per_paper = int(cfg.get("per_paper", 5))
    n_copy = int(cfg.get("copy_ngram", 8))
    rows = []
    for r in raw:
        src = by_key[r["paper_key"]]
        w = normalize_answer(src).split()
        grams = {tuple(w[i:i + n_copy]) for i in range(len(w) - n_copy + 1)}
        seen: list[str] = []
        kept = 0
        for j, q in enumerate(r["questions"]):
            reason = verify(q, src, grams, cfg, seen)
            crit = q.get("_critic") or {}
            if reason == "ok" and use_critic:
                failed = [k for k in CRITIC_SCHEMA["required"] if crit.get(k) is not True]
                reason = f"critic:{failed[0]}" if failed else "ok"
            if reason == "ok" and kept >= per_paper:
                reason = "over_per_paper_cap"
            if reason == "ok":
                seen.append(_norm(q["question"]))
                kept += 1
            rows.append({"qid": f"{r['paper_key']}:q{j}", "paper_key": r["paper_key"], "question": q["question"].strip(),
                         "answer": q.get("answer", "").strip(), "qtype": q.get("type"),
                         "evidence": q.get("evidence", ""), "status": reason, "generator": gen["name"],
                         **{f"critic_{k}": crit.get(k) for k in CRITIC_SCHEMA["required"]}})
    allq = pd.DataFrame(rows)
    allq.to_parquet(ctx.out / "all_generated.parquet", index=False)
    qs = allq[allq["status"] == "ok"].merge(papers[["paper_key", "file", "row", "paper_id", "title", "group", "domain",
                                                    "ws1_flagged", "links"]], on="paper_key", how="left")
    qs.to_parquet(ctx.out / "questions.parquet", index=False)
    status = allq["status"].value_counts()
    qt = qs["qtype"].value_counts()
    ctx.publish("WS2 QA bench (questions)", [
        {"type": "kpis", "items": [
            {"label": "questions kept", "value": len(qs), "fmt": "int"},
            {"label": "papers with questions", "value": int(qs["paper_key"].nunique()), "fmt": "int"},
            {"label": "generated", "value": len(allq), "fmt": "int"},
            {"label": "kept share", "value": len(qs) / max(len(allq), 1), "fmt": "pct1",
             "hint": "passed every verification check"},
            {"label": "generator", "value": gen["name"]}]},
        {"type": "bar", "title": "Verification outcome of generated questions", "horizontal": True, "fmt": "int",
         "labels": list(status.index), "series": [{"name": "questions", "data": status.tolist()}]},
        {"type": "bar", "title": "Kept questions by type", "horizontal": True, "fmt": "int",
         "labels": list(qt.index), "series": [{"name": "questions", "data": qt.tolist()}]},
    ], summary=f"{len(qs)} verified questions on {qs['paper_key'].nunique()} papers")
    return {"questions": len(qs), "generated": len(allq), "papers": int(qs["paper_key"].nunique())}
