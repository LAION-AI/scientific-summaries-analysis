"""S1 shared pieces: record sampling, PII and prompt-injection detectors, the dual-use topic prefilter.

Every detector returns categories and counts, never the matched text: S1 outputs (panels, CSVs, review packets) hold
record ids, fields and categories only, so the safety audit does not redistribute personal data or hazardous text.
"""
from __future__ import annotations

import random
import re
from typing import Any

import pandas as pd

from ..data import input_shards, iter_batches, release_shards
from ..runtime import Context
from ..schema import SUMMARY_FIELDS
from ..ws1.faith.common import paper_key


def setting(ctx: Context, key: str, default: Any) -> Any:
    """A debug-profile setting (profiles.debug.s1_<key>) wins over the default."""
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"s1_{key}", default)
    return default


def stage_table(ctx: Context, stage: str, name: str) -> pd.DataFrame:
    p = ctx.cfg.stage_dir("s1", stage) / name
    if not p.exists():
        raise FileNotFoundError(f"s1/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


def sample_records(ctx: Context, columns: list[str]) -> pd.DataFrame:
    """Debug: every record of the WS1 debug sample. Full: `shards` random release shards x `rows_per_shard` rows."""
    cfg = ctx.section("s1.sample")
    cols = list(dict.fromkeys(columns))
    if ctx.cfg.profile == "debug":
        shards, per = input_shards(ctx.cfg), None
    else:
        rng = random.Random(int(cfg.get("seed", 0)))
        allsh = release_shards(ctx.cfg)
        shards, per = rng.sample(allsh, min(int(cfg.get("shards", 400)), len(allsh))), int(cfg.get("rows_per_shard", 250))
    frames = []
    for sh in shards:
        for tbl in iter_batches(ctx.cfg, sh, cols, batch_size=per or 4096):
            frames.append(tbl.to_pandas())
            if per:
                break
    df = pd.concat(frames, ignore_index=True)
    df["paper_key"] = [paper_key(f, r) for f, r in zip(df["_file"], df["_row"], strict=True)]
    return df


# ---------------------------------------------------------------- PII
def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    s = 0
    for i, v in enumerate(reversed(d)):
        if i % 2:
            v *= 2
            v = v - 9 if v > 9 else v
        s += v
    return s % 10 == 0


_PII = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?:(?<=\s)|^)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}(?=\s|$|[,.;)])"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "us_ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "date_of_birth": re.compile(r"\b(?:date of birth|born on|d\.?o\.?b\.?)\s*[:\-]?\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}", re.I),
    "medical_record_id": re.compile(r"\b(?:MRN|medical record (?:number|no\.?)|patient (?:id|no\.?|number))\s*[:#]?\s*[A-Z0-9-]*\d{4,}", re.I),
    "postal_address": re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Drive|Dr\.)"
                                 r"(?:,\s*[A-Z][a-z]+)?(?:,?\s*\d{5})?"),
    "url_social": re.compile(r"(?:twitter\.com|x\.com|facebook\.com|instagram\.com|linkedin\.com/in|tiktok\.com)/[A-Za-z0-9_.-]{2,}", re.I),
}


def pii_hits(text: str | None) -> dict[str, int]:
    """Category -> number of matches (credit-card candidates must pass the Luhn check)."""
    t = text or ""
    out = {}
    for cat, rx in _PII.items():
        m = rx.findall(t)
        if cat == "credit_card":
            m = [x for x in m if luhn_ok(x)]
        if m:
            out[cat] = len(m)
    return out


def pii_redact(text: str | None, keep: set[str] = frozenset()) -> tuple[str, dict[str, int]]:
    """Replace every PII match (as pii_hits counts them) by `[REDACTED:<category>]` -> (text, counts)."""
    t, counts = text or "", {}
    for cat, rx in _PII.items():
        if cat in keep:
            continue
        def sub(m, cat=cat):
            if cat == "credit_card" and not luhn_ok(m.group(0)):
                return m.group(0)
            counts[cat] = counts.get(cat, 0) + 1
            return f"[REDACTED:{cat}]"
        t = rx.sub(sub, t)
    return t, counts


# ---------------------------------------------------------------- prompt injection
_INJECTION = {
    "override_instructions": re.compile(r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)"
                                        r"\s+(?:instructions|prompts?|rules|messages)", re.I),
    "role_hijack": re.compile(r"\byou are now\b|\bact as (?:an?|the) (?:ai|assistant|system)\b|\bnew instructions?:", re.I),
    "chat_template_tags": re.compile(r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>|\[/?INST\]|<<SYS>>|<start_of_turn>"),
    "role_lines": re.compile(r"(?m)^\s*(?:###\s*)?(?:system|assistant)\s*:\s"),
    "tool_call_json": re.compile(r"[{,]\s*\"(?:action|tool_call|function_call)\"\s*:\s*\"", re.I),
    "tool_call_tags": re.compile(r"</?tool_call>|</?function_call>"),
    "system_prompt_mention": re.compile(r"\b(?:reveal|print|show)\s+(?:your\s+)?system prompt\b", re.I),
    "invisible_chars": re.compile("[​‌‍⁠﻿­]"),
    "bidi_controls": re.compile("[‪-‮⁦-⁩]"),
    "unicode_tags": re.compile("[\U000e0000-\U000e007f]"),
    "llm_boilerplate": re.compile(r"\bas an ai (?:language )?model\b|\bi cannot (?:fulfill|comply with) (?:this|that) request\b", re.I),
}


def injection_hits(text: str | None) -> dict[str, int]:
    t = text or ""
    return {cat: len(m) for cat, rx in _INJECTION.items() if (m := rx.findall(t))}


# ---------------------------------------------------------------- dual-use prefilter
def compile_terms(terms: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {dom: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(ts, key=len, reverse=True)) + r")\b", re.I)
            for dom, ts in terms.items() if ts}


def prefilter(text: str | None, rx: dict[str, re.Pattern]) -> dict[str, int]:
    """Domain -> number of topic-term mentions (a hit means 'review', not 'hazardous')."""
    t = text or ""
    return {dom: len(m) for dom, r in rx.items() if (m := r.findall(t))}


def summary_text(rec: dict, fields: list[str] | None = None) -> str:
    return "\n".join(f"{f}: {rec.get(f)}" for f in (fields or SUMMARY_FIELDS) if rec.get(f))
