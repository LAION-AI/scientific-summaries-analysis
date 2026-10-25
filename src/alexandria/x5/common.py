"""X5 helpers: counterfactual answers, KU sentence edits and their checks, and the gold paper's index units.

A KU here is the cleaned dict of its fields (ws4.common.clean_ku: claims as a list). A sentence is addressed as
(field, claim index or None, claim key or None, sentence index). Edits replace whole sentences; every check is
done on normalised text (qa.normalize_answer), so formatting differences do not hide a leftover original answer.
"""
from __future__ import annotations

import copy
import random
import re
from typing import Any

from ..qa import normalize_answer
from ..schema import PROSE_FIELDS

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
CLAIM_KEYS = ("description", "supporting_evidence", "contradicting_evidence", "implications")


def first_number(s: str) -> re.Match | None:
    return _NUM.search(s or "")


_FORMULA = re.compile(r"[=\[\](){}^∥⁰¹²³⁴⁵⁶⁷⁸⁹_\\|]")
_RANGE = re.compile(r"\s*(?:-|–|—|to|and)\s*$")


def _scaled(raw: str, f: float) -> str:
    v = float(raw.replace(",", ""))
    dec = len(raw.split(".")[1]) if "." in raw else 0
    nv = v * f
    if dec == 0 and nv != int(nv):
        return f"{nv:.1f}"
    return f"{nv:,.{dec}f}" if "," in raw else f"{nv:.{dec}f}"


def perturb_number(answer: str, seed: str, factors: list[float]) -> str | None:
    """The answer with its value scaled by a factor (both ends of a range; same decimals, thousands separators
    kept); None if the answer has no number, is a formula or a name (GPT-4), has more than two numbers, or every
    factor gives the same text."""
    nums = list(_NUM.finditer(answer or ""))
    if not nums or len(nums) > 2 or _FORMULA.search(answer) or len(re.findall(r"[A-Za-z]{2,}", answer)) > 6:
        return None
    m = nums[0]
    if (m.start() > 0 and answer[m.start() - 1] in "-/") or re.search(r",(?!\d{3}\b)", m.group(0)):
        return None
    scale = [m]
    if len(nums) == 2 and _RANGE.match(answer[m.end():nums[1].start()]):
        scale.append(nums[1])
    fs = list(factors)
    random.Random(seed).shuffle(fs)
    for f in fs:
        out, last = [], 0
        for x in scale:
            out += [answer[last:x.start()], _scaled(x.group(0), f)]
            last = x.end()
        new = "".join(out) + answer[last:]
        pct = "%" in answer
        if pct and any(abs(float(t.replace(",", ""))) > 100 for t in _NUM.findall(new)):
            continue
        if normalize_answer(new) != normalize_answer(answer):
            return new
    return None


def sentences(ku: dict) -> list[dict]:
    """Every sentence of the KU's narrative fields and claims, with its address."""
    out = []
    for f in PROSE_FIELDS:
        for j, s in enumerate(_SENT.split(str(ku.get(f) or ""))):
            if s.strip():
                out.append({"field": f, "claim": None, "key": None, "j": j, "text": s.strip()})
    for ci, c in enumerate(ku.get("claims") or []):
        for k in CLAIM_KEYS:
            for j, s in enumerate(_SENT.split(str(c.get(k) or ""))):
                if s.strip():
                    out.append({"field": "claims", "claim": ci, "key": k, "j": j, "text": s.strip()})
    return out


def rank_candidates(sents: list[dict], question: str, answer: str, n: int) -> list[dict]:
    """Sentences most likely to state the answer: answer-token overlap (weighted) + question-token overlap."""
    a, q = set(normalize_answer(answer).split()), set(normalize_answer(question).split())

    def score(s):
        w = set(normalize_answer(s["text"]).split())
        return 3 * len(a & w) + len(q & w) + (5 if normalize_answer(answer) in normalize_answer(s["text"]) else 0)
    return sorted(sents, key=score, reverse=True)[:n]


def _split(text: str) -> list[str]:
    return [s for s in _SENT.split(str(text or "")) if s.strip()]


def apply_edits(ku: dict, edits: list[tuple[dict, str]]) -> dict:
    """Replace addressed sentences (address dict, new sentence) in a deep copy of the KU."""
    out = copy.deepcopy(ku)
    for addr, new in edits:
        if addr["field"] == "claims":
            c = out["claims"][addr["claim"]]
            parts = _split(c.get(addr["key"]))
            if addr["j"] < len(parts):
                parts[addr["j"]] = new
            c[addr["key"]] = " ".join(parts)
        else:
            parts = _split(out.get(addr["field"]))
            if addr["j"] < len(parts):
                parts[addr["j"]] = new
            out[addr["field"]] = " ".join(parts)
    return out


def replace_everywhere(ku: dict, old: str, new: str) -> dict:
    """Case-insensitive literal replacement of `old` in every narrative field and claim (left-over mentions)."""
    if not old.strip():
        return ku
    pat = re.compile(re.escape(old.strip()), re.I)
    out = copy.deepcopy(ku)
    for f in PROSE_FIELDS:
        if out.get(f):
            out[f] = pat.sub(new, str(out[f]))
    for c in out.get("claims") or []:
        for k in CLAIM_KEYS:
            if c.get(k):
                c[k] = pat.sub(new, str(c[k]))
    return out


def ku_text(ku: dict) -> str:
    parts = [str(ku.get(f) or "") for f in PROSE_FIELDS]
    parts += [str(c.get(k) or "") for c in ku.get("claims") or [] for k in CLAIM_KEYS]
    return " ".join(parts)


def mentions(text: str, answer: str) -> bool:
    a = normalize_answer(answer)
    return bool(a) and f" {a} " in f" {normalize_answer(text)} "


def check(orig_ku: dict, cf_ku: dict, answer: str, cf_answer: str) -> str:
    """'ok' or the reason the counterfactual is rejected."""
    if not cf_answer or normalize_answer(cf_answer) == normalize_answer(answer):
        return "same_answer"
    na, nc = normalize_answer(answer), normalize_answer(cf_answer)
    if na in nc or nc in na:
        return "nested_answers"
    t = ku_text(cf_ku)
    if not mentions(t, cf_answer):
        return "cf_answer_not_in_edited_ku"
    if (len(na) >= 3 or any(ch.isdigit() for ch in na)) and mentions(t, answer):
        return "original_answer_left"
    if ku_text(orig_ku) == t:
        return "no_change"
    return "ok"


def summary_units(ku: dict, cfg: dict | None = None) -> list[dict]:
    """The paper's KU units exactly as the WS3 index builds them (summary fields + claims), from a KU dict."""
    import json

    from ..ws3.index import paper_units
    rec = {**{k: v for k, v in ku.items() if k != "claims"}, "claims": json.dumps(ku.get("claims") or [])}
    return [u for u in paper_units(rec, cfg or {}) if u["kind"] == "summary"]


def swap_units(retrieved: list[dict], gold: str, orig_units: list[dict], cf_units: list[dict],
               counterfactual: bool) -> list[str]:
    """Texts of the retrieved units, with the gold paper's units rebuilt from the original or the edited KU (same
    field, same order) so the two conditions differ only by the edit; gold units with no rebuilt match are dropped."""
    pools = {}
    src = cf_units if counterfactual else orig_units
    for u in src:
        pools.setdefault(u["field"], []).append(u["text"])
    taken: dict[str, int] = {}
    out = []
    for r in retrieved:
        if r["paper_key"] != gold:
            out.append(r["text"])
            continue
        i = taken.get(r["field"], 0)
        pool = pools.get(r["field"], [])
        if i < len(pool):
            out.append(pool[i])
            taken[r["field"]] = i + 1
    return out


def label(vs_orig: str, vs_cf: str) -> str:
    """Outcome of one answer from the two SimpleQA grades."""
    o, c = vs_orig == "CORRECT", vs_cf == "CORRECT"
    if o and not c:
        return "original"
    if c and not o:
        return "counterfactual"
    if o and c:
        return "both"
    if vs_orig == "NOT_ATTEMPTED" and vs_cf == "NOT_ATTEMPTED":
        return "abstain"
    return "other"


def summarize_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    return {k: sum(r["label"] == k for r in rows) / n if n else float("nan")
            for k in ("original", "counterfactual", "both", "abstain", "other")}
