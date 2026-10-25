"""X4 shared pieces (pure functions, unit-tested): paper-level KU views of the WS3 index, topic groups, the field
router, the passages of the three RAG arms, and the rule checks of generated items.

Field-level access is what a structured corpus adds: a question about a paper's *limitations* can be answered by
searching only the limitations of the literature, and a question that compares papers can be answered from the same
field of each. The retrieval arms differ only in what they put in front of the model, under one token budget:
  R-fulltext  top-k full-text chunks
  R-KU        the whole KUs of the top-k papers (all summary fields of each)
  R-field     top-k units of the KU field the question asks about (rule-based field router)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

FIELD_ORDER = ["executive_summary", "research_question_hypothesis", "methodological_details", "procedures_architectures",
               "key_results", "interpretation_implications", "contradictions_limitations", "claims"]
TO_SKILL_FIELD = {"limitations": "limitations", "methods": "methods", "results": "key_results", "claims": "claims"}
_ROUTES = [  # (skill field, pattern): first match wins; "any" otherwise
    ("limitations", r"\b(limitation|limitations|caveat|weakness|drawback|shortcoming|threat|constraint|cannot|fails?)\b"),
    ("claims", r"\b(claims?|argues?|contradict|disagree|agree|supports? the (view|idea)|asserts?)\b"),
    ("methods", r"\b(method|methods|dataset|data set|setup|set-up|procedure|architecture|sample|participants|"
                r"model(s)? (is|are|was|were) used|how (was|were|is|are|did)|technique|protocol|design)\b"),
    ("key_results", r"\b(result|results|find|found|findings|report(s|ed)?|achiev\w*|accuracy|improv\w*|higher|lower|"
                    r"best|outperform\w*|value|rate|effect)\b"),
]
_NOT_SELF_CONTAINED = re.compile(r"\b((this|the) (paper|study|work|article|authors|researchers)\b|according to the|"
                                 r"paper [0-9A-Z]\b|P[0-9]\b)", re.I)


def paper_fields(units: pd.DataFrame) -> dict[str, dict[str, str]]:
    """paper_key -> {KU field: text} from the WS3 index summary units (claims concatenated)."""
    s = units[units["kind"] == "summary"]
    out: dict[str, dict[str, str]] = {}
    for (pk, f), g in s.groupby(["paper_key", "field"], sort=False):
        out.setdefault(pk, {})[f] = "\n".join(g["text"].astype(str))
    return out


def paper_vectors(units: pd.DataFrame, emb: np.ndarray, center: bool = False) -> tuple[list[str], np.ndarray]:
    """Mean of each paper's summary-unit embeddings, L2-normalised (topic vector). `center` subtracts the corpus mean
    first: raw bge-m3 means are anisotropic (every pair has cosine ~0.7), centring makes similarity topical."""
    s_idx = np.flatnonzero(units["kind"].to_numpy() == "summary")
    keys = units["paper_key"].to_numpy()[s_idx]
    order = list(dict.fromkeys(keys))
    pos = {k: i for i, k in enumerate(order)}
    vec = np.zeros((len(order), emb.shape[1]), np.float32)
    cnt = np.zeros(len(order), np.float32)
    for row, k in enumerate(keys):
        vec[pos[k]] += np.asarray(emb[row], np.float32)
        cnt[pos[k]] += 1
    vec /= np.maximum(cnt, 1)[:, None]
    if center:
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        vec -= vec.mean(0, keepdims=True)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    return order, vec


def topic_groups(keys: list[str], vec: np.ndarray, size: int, min_sim: float, n: int, rng: np.random.Generator) -> list[list[str]]:
    """Up to n groups: a random seed paper + its size-1 nearest neighbours (cosine >= min_sim); no repeated group."""
    sims = vec @ vec.T
    seen, groups = set(), []
    for i in rng.permutation(len(keys)):
        nn = [j for j in np.argsort(-sims[i]) if j != i][: size - 1]
        if len(nn) < size - 1 or sims[i, nn[-1]] < min_sim:
            continue
        g = tuple(sorted([keys[i], *[keys[j] for j in nn]]))
        if g in seen:
            continue
        seen.add(g)
        groups.append(list(g))
        if len(groups) >= n:
            break
    return groups


def route_field(question: str) -> str:
    """The KU field a question asks about (skill field name), from keywords; 'any' if none matches."""
    q = question.lower()
    for field, pat in _ROUTES:
        if re.search(pat, q):
            return field
    return "any"


def ku_text(fields: dict[str, str], max_chars: int | None = None) -> str:
    parts = [f"## {f}\n{fields[f]}" for f in FIELD_ORDER if fields.get(f)]
    t = "\n".join(parts)
    return t[:max_chars] if max_chars else t


def check_item(q: dict, field_text: str, others: list[str], max_words: int, min_match: float) -> str:
    """'ok' or the reason a generated item is dropped (rule checks; the LLM verifier comes after)."""
    from rapidfuzz import fuzz
    question, answer, ev = (str(q.get(k) or "") for k in ("question", "answer", "evidence"))
    if not question or not answer:
        return "empty"
    if len(answer.split()) > max_words:
        return "answer_too_long"
    if _NOT_SELF_CONTAINED.search(question):
        return "not_self_contained"
    if answer.lower().strip(" .") in question.lower():
        return "answer_in_question"
    norm = lambda s: " ".join(s.lower().split())              # noqa: E731
    if len(ev) < 10 or fuzz.partial_ratio(norm(ev), norm(field_text)) < min_match:
        return "evidence_not_in_field"
    if any(fuzz.partial_ratio(norm(ev), norm(o)) >= 95 for o in others):
        return "evidence_also_in_other_paper"
    return "ok"


def gold_recall(seen: list[str], gold: list[str]) -> float | None:
    gold = [g for g in gold if g]
    if not gold:
        return None
    return sum(g in set(seen) for g in gold) / len(gold)
