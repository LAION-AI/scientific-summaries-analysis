"""WS5 shared pieces: the KU output schema, the summarisation prompt, output validation and the verbatim gate.

A KU here is exactly the release's 18-field structured summary (SUMMARY_FIELDS), written as one JSON object, so
teacher, student, released-v2.0 and extractor outputs are all scored the same way.
"""
from __future__ import annotations

import json
from typing import Any

from ..runtime import Context
from ..schema import SUMMARY_FIELDS, load_summary
from ..textutil import tokens
from ..ws1.lexical import kgram_set, longest_span

FIELD_SPEC = {
    "summary_title": "the paper's title",
    "summary_authors": "the authors as listed in the paper, or an empty string",
    "field_subfield": "discipline and subfield, e.g. 'Physics - Condensed Matter'",
    "type_of_paper": "e.g. empirical, theoretical, methodological, review",
    "executive_summary": "a dense 300-500 word overview: problem, approach, main results with numbers, significance",
    "research_context": "background, prior work and the gap this paper addresses",
    "research_question_hypothesis": "the research questions / hypotheses and what would support or refute them",
    "methodological_details": "design, data, participants or materials, measures, analysis",
    "procedures_architectures": "procedures, models, architectures, algorithms and settings, specific enough to reproduce",
    "key_results": "the main quantitative and qualitative results, with the numbers the paper reports",
    "interpretation_implications": "how the results answer the questions and what they imply",
    "contradictions_limitations": "limitations, threats to validity, inconsistencies",
    "claims": "the paper's main claims, each with supporting and contradicting evidence and implications",
    "data_code_availability": "data and code availability (links, licences) if stated, else an empty string",
    "robustness_ablation_notes": "robustness checks and ablations if any, else an empty string",
    "ethical_considerations": "ethical considerations if any, else an empty string",
    "key_figures_tables": "the most important figures / tables and what they show",
    "three_takeaways": "three takeaways for a researcher, as three short paragraphs",
}
SYSTEM = ("You write a structured summary (a knowledge unit) of one scientific paper for researchers who will not "
          "read the paper. Write in your own words: do not copy sentences or long phrases from the paper. Report "
          "numbers exactly as the paper states them. Never invent results; if a field does not apply, give an "
          "empty string. Answer with the JSON object only.\n\nFields:\n"
          + "\n".join(f"- {k}: {v}" for k, v in FIELD_SPEC.items()))
USER = "PAPER{note}:\n<<<\n{text}\n>>>"


def ku_schema() -> dict:
    """JSON schema of one KU (structured outputs for the Claude API and vLLM)."""
    claim = {"type": "object", "properties": {k: {"type": "string"} for k in
                                              ("description", "supporting_evidence", "contradicting_evidence",
                                               "implications")},
             "required": ["description", "supporting_evidence", "contradicting_evidence", "implications"],
             "additionalProperties": False}
    props = {f: ({"type": "array", "items": claim} if f == "claims" else {"type": "string"}) for f in SUMMARY_FIELDS}
    return {"type": "object", "properties": props, "required": list(SUMMARY_FIELDS), "additionalProperties": False}


def user_prompt(text: str, cut: bool) -> str:
    return USER.format(note=" (first part; the rest is cut)" if cut else "", text=text)


def parse_ku(text: str | None) -> dict | None:
    s = (text or "").strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1]
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def schema_check(ku: dict | None, required: list[str]) -> dict[str, Any]:
    """WS1 schema validation of one KU: parses, validates as `Summary` (load_summary), required fields present and
    not placeholders. Returns {valid, filled (share of the 18 fields), issues}."""
    if ku is None:
        return {"valid": False, "filled": 0.0, "issues": "unparsable"}
    rec = {f: (json.dumps(ku[f]) if f == "claims" and not isinstance(ku.get(f), str) else ku.get(f)) for f in SUMMARY_FIELDS}
    norm = load_summary(rec)
    s, issues = norm["summary"], list(norm.get("issues") or [])
    extra = sorted(set(ku) - set(SUMMARY_FIELDS))
    missing = [f for f in required if not (s.get(f) if f != "claims" else s.get("claims"))]
    issues += [f"extra_key:{k}" for k in extra] + [f"missing:{f}" for f in missing]
    filled = sum(1 for f in SUMMARY_FIELDS if (s.get(f) if f != "claims" else s.get("claims"))) / len(SUMMARY_FIELDS)
    return {"valid": not missing and not extra, "filled": filled, "issues": ";".join(issues) or None}


def ku_text(ku: dict | None) -> str:
    if not ku:
        return ""
    parts = []
    for f in SUMMARY_FIELDS:
        v = ku.get(f)
        if f == "claims" and isinstance(v, list):
            v = " ".join(str(c.get("description", "")) + " " + str(c.get("supporting_evidence", ""))
                         for c in v if isinstance(c, dict))
        if v:
            parts.append(str(v))
    return "\n".join(parts)


BIBLIOGRAPHIC = ("summary_title", "summary_authors", "field_subfield", "type_of_paper")   # copied by design


def overlap(ku: dict | None, source: str, k: int = 8) -> dict[str, Any]:
    """Verbatim overlap with the source over the content fields (title, authors, field and paper type are copied by
    design and excluded): longest copied span (words), and the share of the KU's word k-grams found in the source."""
    if not ku:
        return {"max_span": 0, "copied_kgram_share": 0.0}
    src = kgram_set(tokens(source), k)
    best = 0
    for f in (f for f in SUMMARY_FIELDS if f not in BIBLIOGRAPHIC):
        v = ku.get(f)
        txt = json.dumps(v) if isinstance(v, list) else (v or "")
        best = max(best, longest_span(tokens(txt), src, k))
    all_toks = tokens(ku_text({f: v for f, v in ku.items() if f not in BIBLIOGRAPHIC}))
    grams = [hash(tuple(all_toks[i:i + k])) for i in range(len(all_toks) - k + 1)]
    return {"max_span": best, "copied_kgram_share": (sum(g in src for g in grams) / len(grams)) if grams else 0.0}


def setting(ctx: Context, key: str, default: Any) -> Any:
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"ws5_{key}", default)
    return default


def stage_table(ctx: Context, stage: str, name: str):
    import pandas as pd
    p = ctx.cfg.stage_dir("ws5", stage) / name
    if not p.exists():
        raise FileNotFoundError(f"ws5/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)
