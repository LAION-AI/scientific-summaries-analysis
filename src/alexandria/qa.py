"""Question answering pieces shared by WS2 and WS3: answer prompt, answer normalisation, model-free scores and the
LLM grader.

Grading follows SimpleQA (Wei et al., 2024, openai/simple-evals): a judge sees question, gold answer and predicted
answer and returns CORRECT / INCORRECT / NOT_ATTEMPTED, so hedged or empty answers are not counted as wrong ones
(accuracy = CORRECT / all; "attempted" accuracy is reported too). Model-free scores (normalised exact match and
token F1, SQuAD-style; numbers compared with a relative tolerance) are kept next to the judge as a robustness check.
"""
from __future__ import annotations

import re
import string
from collections import Counter

ANSWER_SYSTEM = ("You answer questions about scientific research. Reply with the short answer only: a number with "
                 "its unit, a name, or a phrase of at most 12 words. If you do not know, reply exactly: I don't know.")


def answer_user(question: str, context: str | None) -> str:
    if not context:
        return f"Question: {question}"
    return f"Context:\n<<<\n{context}\n>>>\n\nQuestion: {question}"


GRADES = ("CORRECT", "INCORRECT", "NOT_ATTEMPTED")
GRADE_SCHEMA = {"type": "object", "properties": {"grade": {"type": "string", "enum": list(GRADES)}},
                "required": ["grade"], "additionalProperties": False}

GRADER_SYSTEM = """You grade answers to questions about scientific papers against a gold answer.
Return CORRECT, INCORRECT or NOT_ATTEMPTED:
- CORRECT: the predicted answer contains the gold answer's key information and nothing that contradicts it. Paraphrases, synonyms, abbreviations, different but equivalent units, and extra correct detail are fine. Numbers must match to the precision of the gold answer (e.g. gold 3.2 -> 3.2 or 3.21 is correct, 3 is not; gold "about 40%" -> 38-42% is correct).
- INCORRECT: the predicted answer contradicts the gold answer, gives a different value or entity, or hedges between several candidates of which at least one is wrong.
- NOT_ATTEMPTED: the predicted answer does not give the key information and does not contradict it (e.g. "I don't know", a refusal, or an answer that only restates the question).
Judge only whether the predicted answer matches the gold answer; do not judge whether the gold answer is true.
Answer with JSON: {"grade": "..."}"""


def grader_user(question: str, gold: str, predicted: str) -> str:
    return f"Question: {question}\nGold answer: {gold}\nPredicted answer: {predicted}"


# ---------------------------------------------------------------- model-free scores
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def normalize_answer(s: str | None) -> str:
    s = (s or "").lower().replace("−", "-")
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {".", "-"})
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split()).strip(" .")


def is_abstention(s: str | None) -> bool:
    t = normalize_answer(s)
    return t == "" or t.startswith(("i dont know", "i do not know", "unknown", "not stated", "cannot determine",
                                    "not mentioned", "not provided"))


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def exact_match(pred: str | None, gold: str | None, rel_tol: float = 0.01) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g or (len(g) > 3 and g in p and len(p) <= 3 * len(g)):
        return True
    gn, pn = _numbers(g), _numbers(p)
    if gn and pn and len(gn) == 1:                  # one gold number: any predicted number within tolerance
        return any(abs(x - gn[0]) <= rel_tol * max(abs(gn[0]), 1e-9) for x in pn)
    return False


def token_f1(pred: str | None, gold: str | None) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    k = sum(common.values())
    if k == 0:
        return 0.0
    prec, rec = k / len(p), k / len(g)
    return 2 * prec * rec / (prec + rec)


def ngram_overlap(a: str | None, b: str | None, n: int = 5) -> float:
    """Share of word n-grams of `a` that also occur in `b` (lexical-bias covariate: question vs context)."""
    aw, bw = normalize_answer(a).split(), normalize_answer(b).split()
    if len(aw) < n or len(bw) < n:
        return 0.0
    ag = {tuple(aw[i:i + n]) for i in range(len(aw) - n + 1)}
    bg = {tuple(bw[i:i + n]) for i in range(len(bw) - n + 1)}
    return len(ag & bg) / len(ag)


def grade_batch(llm, tok, spec, items: list[dict], max_tokens: int = 16) -> list[str]:
    """LLM-judge grades for [{question, gold, predicted}] (abstentions short-circuit to NOT_ATTEMPTED)."""
    from .llm import chat_prompt, json_params, parse_json
    out: list[str | None] = [("NOT_ATTEMPTED" if is_abstention(it["predicted"]) else None) for it in items]
    todo = [i for i, g in enumerate(out) if g is None]
    if todo:
        prompts = [chat_prompt(tok, spec, GRADER_SYSTEM, grader_user(items[i]["question"], items[i]["gold"],
                                                                     items[i]["predicted"])) for i in todo]
        res = llm.generate(prompts, json_params(GRADE_SCHEMA, max_tokens), use_tqdm=False)
        for i, r in zip(todo, res, strict=True):
            v = parse_json(r.outputs[0].text if r.outputs else "") or {}
            out[i] = v.get("grade") if v.get("grade") in GRADES else "INCORRECT"
    return out  # type: ignore[return-value]
