"""`load_summary(record) -> canonical dict`: the one cleaned view of a record that every workstream uses.

    from alexandria.schema import load_summary
    rec = load_summary(row)                   # row: dict from the release parquet (any key spelling)
    rec["summary"]["key_results"], rec["summary"]["claims"][0]["description"], rec["issues"]

What it does (and records in rec["issues"]):
  * keys: exact / case-and-punctuation / reviewed alias table (resources/aliases.yaml); unknown keys -> rec["extras"]
  * values: stripped; empty and placeholder values ("N/A", "[Unknown]", "Title not provided") -> None
  * claims: JSON parsed; claim sub-keys aliased; off-schema sub-keys kept per claim under extras; non-dict items dropped
  * a real value always wins over an empty duplicate spelling; two different real values keep the first and log it
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .aliases import resolve
from .fields import CLAIM_KEYS, SUMMARY_FIELDS
from .model import CanonicalRecord, Claim, Provenance, Summary

_META_SKIP = set(SUMMARY_FIELDS) | {"paper_id", "_file", "_row", "_group", "text_sanitized", "text_raw"}


def parse_claims(raw: Any, issues: list[str]) -> tuple[list[dict], list[dict]]:
    """-> (canonical claims, per-claim extras)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        issues.append("claims:empty")
        return [], []
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            issues.append("claims:bad_json")
            return [], []
    if isinstance(value, dict):                   # a single claim object instead of a list
        value = [value]
        issues.append("claims:not_list")
    if not isinstance(value, list):
        issues.append("claims:not_list")
        return [], []
    out, extras = [], []
    for i, item in enumerate(value):
        if not isinstance(item, Mapping):
            issues.append("claims:non_dict_item")
            continue
        claim: dict[str, str] = {}
        extra: dict[str, Any] = {}
        for k, v in item.items():
            m = resolve(str(k), "claims")
            if m.key is None:
                extra[str(k)] = v
                issues.append(f"claims:off_schema_key={k}")
                continue
            if m.how == "alias":
                issues.append(f"claims:alias={k}->{m.key}")
            text = "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
            if not claim.get(m.key):
                claim[m.key] = text.strip()
        missing = [k for k in CLAIM_KEYS if k not in claim]
        if missing:
            issues.append("claims:missing_keys=" + ",".join(missing))
        if not claim.get("description"):
            issues.append(f"claims:{i}:no_description")
            continue
        out.append({k: claim.get(k, "") for k in CLAIM_KEYS})
        extras.append(extra)
    return out, extras


def load_summary(record: Mapping[str, Any], *, null_placeholders: bool = True, as_model: bool = False):
    from ..textutil import value_status          # local: textutil imports schema.fields (avoids an import cycle)
    issues: list[str] = []
    fields: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    seen_from: dict[str, str] = {}
    for key, value in record.items():
        m = resolve(str(key), "columns")
        if m.key is None:
            if not str(key).startswith("_"):
                extras[str(key)] = value
                issues.append(f"key:{m.how}={key}")
            continue
        if m.how == "alias":
            issues.append(f"key:alias={key}->{m.key}")
        if m.key in fields and fields[m.key] not in (None, ""):
            if value not in (None, "") and value != fields[m.key]:
                issues.append(f"key:conflict={key}~{seen_from[m.key]}")
            continue
        fields[m.key] = value
        seen_from[m.key] = str(key)

    summary: dict[str, Any] = {}
    for f in SUMMARY_FIELDS:
        if f == "claims":
            continue
        v = fields.get(f)
        v = v.strip() if isinstance(v, str) else v
        status = value_status(v if isinstance(v, str) else None)
        if status in ("empty", "placeholder", "not_provided"):
            if status != "empty":
                issues.append(f"{f}:{status}")
            if null_placeholders or status == "empty":
                v = None
        summary[f] = v
    claims, claim_extras = parse_claims(fields.get("claims"), issues)
    summary["claims"] = claims
    if any(claim_extras):
        extras["claims_extras"] = claim_extras

    metadata = {k: v for k, v in fields.items() if k not in _META_SKIP}
    rec = CanonicalRecord(
        paper_id=fields.get("paper_id"),
        provenance=Provenance(file=record.get("_file"), row=record.get("_row"), group=record.get("_group")),
        summary=Summary(**{**summary, "claims": [Claim(**c) for c in claims]}),
        metadata=metadata, extras=extras, issues=issues)
    return rec if as_model else rec.model_dump()
