"""Text checks shared by the normalizer and the WS1 stages. Pure functions, no I/O, unit-tested.

Placeholder rules are copied from KU-Fixes build_fix_index.py so both projects flag the same values.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .schema.fields import CANONICAL_COLUMNS

# ---------------------------------------------------------------- empty / placeholder values (KU-Fixes rules)
_PLACEHOLDER = re.compile(
    r"(?i)^\s*[\[(]?\s*(unknown|n/?a|none|null|not (provided|specified|"
    r"available|mentioned|stated|given|listed))\s*[\])]?\.?\s*$")
_NOT_PROVIDED = re.compile(
    r"(?i)^\s*(the )?(paper'?s? )?(title|authors?|author names?)( of the paper)?"
    r" ((is|are|was|were) )?(not|n't) (provided|specified|listed|mentioned|"
    r"available|given|stated|included)")
# Short "nothing to report" sentences, e.g. "No ethical considerations are mentioned in the paper."
# Legitimate for the optional fields, so reported separately from placeholders.
_NA_STATEMENT = re.compile(
    r"(?i)^\s*(there (is|are|were) )?(no|not any|none)\b[^.]{0,120}\b(mention|mentioned|reported|discussed|"
    r"provided|specified|available|stated|described|noted|identified|given)\b[^.]{0,60}\.?\s*$|"
    r"^\s*(the )?(paper|study|article|authors?|text)\b[^.]{0,80}\b(does|do|did) not (mention|report|discuss|"
    r"provide|specify|describe|state|include|address)\b[^.]{0,120}\.?\s*$")


def value_status(text: str | None) -> str:
    """ok | empty | placeholder | not_provided | na_statement"""
    s = (text or "").strip()
    if not s:
        return "empty"
    if _PLACEHOLDER.match(s):
        return "placeholder"
    if _NOT_PROVIDED.match(s):
        return "not_provided"
    if len(s) < 300 and _NA_STATEMENT.match(s):
        return "na_statement"
    return "ok"


# ---------------------------------------------------------------- refusals, boilerplate, leaks
REFUSAL = re.compile(
    r"(?i)\b(as an (ai|a\.i\.|language model)|i('m| am) (sorry|unable|not able)|i (cannot|can't|can not) "
    r"(provide|summari[sz]e|access|help|complete|fulfil)|i do not have access|unable to (access|summari[sz]e) "
    r"the (paper|document|text)|the provided text (is|does not|doesn't) (appear|seem|contain)|"
    r"no (paper|text|document) (was|has been) provided|please provide the (paper|text|document))")
MARKUP_LEAK = re.compile(r"```|<\/?(json|xml|response|answer|output)>|^\s*[{\[]\s*\"|\"\s*:\s*\"|\\n\\n|\[insert ",
                         re.I | re.M)
_FIELD_NAMES = sorted({c for c in CANONICAL_COLUMNS if "_" in c}, key=len, reverse=True)
KEY_LEAK = re.compile(r"(?i)(?<![\w])(" + "|".join(map(re.escape, _FIELD_NAMES)) + r"|hyp_\w*)\s*[\"']?\s*[:=]")


def refusal(text: str | None) -> bool:
    return bool(text) and bool(REFUSAL.search(text))


def markup_leak(text: str | None) -> bool:
    return bool(text) and bool(MARKUP_LEAK.search(text))


def key_leak(text: str | None) -> list[str]:
    """Canonical key names written into a value, e.g. 'key_results: ...' -> ['key_results']."""
    return sorted({m.group(1).lower() for m in KEY_LEAK.finditer(text or "")})


# ---------------------------------------------------------------- truncation
_TERMINAL = set(".!?)\"'”’]}*:;%")


def ends_mid_sentence(text: str | None) -> bool:
    """True when prose ends without terminal punctuation (generation stopped mid-sentence).
    Ignores very short values and values ending in a citation-like token or a number with a unit."""
    s = (text or "").rstrip()
    if len(s) < 40:
        return False
    if s[-1] in _TERMINAL:
        return False
    return not re.search(r"(\bet al|\d)\s*$", s[-12:])


# ---------------------------------------------------------------- tokens, words, numbers, entities
_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text: str | None) -> list[str]:
    return _WORD.findall(text or "")


def word_count(text: str | None) -> int:
    return len(words(text))


def tokens(text: str | None) -> list[str]:
    """Lower-cased word tokens for verbatim-overlap counting (punctuation dropped, accents folded)."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", t)


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return frozenset(ENGLISH_STOP_WORDS)
    except ImportError:                           # pragma: no cover
        return frozenset("the a an and or of to in for on with by is are was were be as at from that this".split())


def content_words(text: str | None, min_len: int = 4) -> set[str]:
    sw = stopwords()
    return {w for w in tokens(text) if len(w) >= min_len and w not in sw and not w.isdigit()}


_NUMBER = re.compile(r"(?<![\w.])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)


def numbers(text: str | None) -> set[str]:
    """Normalised numeric literals: '1,200' -> '1200', '0.50' -> '0.5', '−3' -> '-3'. Single digits 0-9 are
    dropped (list markers, 'two-stage' rewrites) because they match almost any source by chance."""
    out = set()
    for m in _NUMBER.findall(text or ""):
        s = m.replace(",", "").replace("−", "-")
        try:
            v = float(s)
        except ValueError:
            continue
        if abs(v) < 10 and "." not in s:
            continue
        out.add(("%f" % v).rstrip("0").rstrip("."))
    return out


_ENTITY = re.compile(r"\b(?:[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)*|[A-Za-z]+-\d[\w.-]*)\b")


def entities(text: str | None) -> set[str]:
    """Named-thing heuristics without an NER model: acronyms and mixed-case/alphanumeric names
    (BERT, ResNet-50, GPT-4, CRISPR-Cas9, SrTiO3). Lower precision than NER, but cheap and deterministic."""
    return {m for m in _ENTITY.findall(text or "") if len(m) >= 2}


_ASCII_WORD = re.compile(r"\b[a-z]{2,}\b")
_COMMON = frozenset("the of and to in is that for are with as on by this be we from an at which it or can".split())


def garbled_fraction(text: str | None, max_chars: int = 20000) -> float | None:
    """Share of letters outside ASCII in the source: ~0.00-0.03 for normal English text, > 0.3 for PDFs whose fonts
    were extracted without a Unicode map ('ËÔ Ö ÐÐÝ ×ÝÑÑ ØÖ' for 'Spherically symmetric'). Combined with the absence
    of common English function words, which garbled text lacks. None for an empty source."""
    s = (text or "")[:max_chars]
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 200:
        return None
    non_ascii = sum(1 for c in letters if ord(c) > 127) / len(letters)
    words_ = _ASCII_WORD.findall(s.lower())
    common = sum(1 for w in words_ if w in _COMMON) / max(len(words_), 1)
    return non_ascii if common < 0.08 else non_ascii * 0.5


def non_latin_fraction(text: str | None) -> float:
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if not ("LATIN" in unicodedata.name(ch, ""))) / len(letters)


# ---------------------------------------------------------------- names
def name_tokens(text: str | None) -> set[str]:
    """Name words (>= 3 letters, accents folded) from any author-list format: 'A B Last (Inst); C Last2',
    'First Last, First2 Last2', 'Last, First'. Order- and format-independent, so lists can be compared by Jaccard.
    Initials and affiliations in parentheses are ignored."""
    t = re.sub(r"\([^)]*\)", " ", text or "")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z]+", t) if len(w) >= 3 and w not in {"and", "et", "al", "the"}}


def canon_name(s: str) -> str:
    t = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", t)
