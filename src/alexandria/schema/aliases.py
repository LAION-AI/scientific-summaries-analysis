"""Key aliasing: map variant keys onto canonical ones, and propose new aliases for human review.

resolve("Research Question") -> KeyMatch("research_question_hypothesis", "alias")
Resolution order: exact -> canon() equality (case/space/punctuation) -> reviewed alias table -> unknown.
Fuzzy matching is only used to PROPOSE candidates (discover_candidates); it never resolves a key by itself.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

import yaml

from .fields import CANONICAL_COLUMNS, CLAIM_KEYS

Kind = Literal["columns", "claims"]
_CANONICAL: dict[str, tuple[str, ...]] = {"columns": tuple(CANONICAL_COLUMNS), "claims": CLAIM_KEYS}


def canon(name: object) -> str:
    """'Research_Question / Hypothesis' -> 'researchquestionhypothesis'."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@dataclass(frozen=True)
class KeyMatch:
    key: str | None          # canonical key, or None (off-schema / unknown)
    how: str                 # exact | canon | alias | off_schema | unknown


@lru_cache(maxsize=1)
def alias_table() -> dict[str, dict[str, str | None]]:
    text = resources.files("alexandria.resources").joinpath("aliases.yaml").read_text()
    raw = yaml.safe_load(text) or {}
    return {kind: {canon(k): v for k, v in (raw.get(kind) or {}).items()} for kind in _CANONICAL}


def resolve(name: str, kind: Kind = "columns", table: Mapping[str, Mapping[str, str | None]] | None = None) -> KeyMatch:
    canonical = _CANONICAL[kind]
    if name in canonical:
        return KeyMatch(name, "exact")
    c = canon(name)
    for k in canonical:
        if canon(k) == c:
            return KeyMatch(k, "canon")
    tbl = (table or alias_table()).get(kind, {})
    if c in tbl:
        target = tbl[c]
        return KeyMatch(target, "alias") if target else KeyMatch(None, "off_schema")
    return KeyMatch(None, "unknown")


def discover_candidates(observed: Mapping[str, int] | Iterable[str], kind: Kind = "columns",
                        min_score: float = 70.0, top: int = 3) -> list[dict]:
    """Rank canonical keys for every observed key that does not resolve. For review only.

    Score = max(rapidfuzz ratio, partial_ratio) on canon() forms, boosted when the variant is a prefix of the
    canonical key (abbreviations such as "hyp" / "hypo" -> research_question_hypothesis via the "hypothesis" token).
    """
    from rapidfuzz import fuzz
    counts = dict(observed) if isinstance(observed, Mapping) else {k: 1 for k in observed}
    canonical = _CANONICAL[kind]
    out = []
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        m = resolve(key, kind)
        if m.how in ("exact", "canon"):
            continue
        c = canon(key)
        scored = []
        for k in canonical:
            ck = canon(k)
            tokens = [canon(t) for t in k.split("_")]
            prefix = bool(c) and any(t.startswith(c) for t in tokens) and len(c) >= 3
            score = max(fuzz.ratio(c, ck), fuzz.partial_ratio(c, ck) if len(c) >= 4 else 0, 90.0 if prefix else 0)
            scored.append((score, k))
        scored.sort(reverse=True)
        best = [(round(s, 1), k) for s, k in scored[:top] if s >= min_score]
        out.append({"key": key, "count": n, "resolved_as": m.key, "resolution": m.how,
                    "candidates": [{"canonical": k, "score": s} for s, k in best]})
    return out
