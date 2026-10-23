"""WS2 shared pieces: the arms, how each representation of a paper is rendered, and small table helpers."""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ..runtime import Context
from ..schema import PROSE_FIELDS, load_summary

# Representation arms (what the answerer gets as context). Order = table / figure order.
ARMS = {
    "closed_book": "model alone (no context)",
    "full_text": "full paper text (first N characters)",
    "full_text_matched": "full paper text cut to the clean summary's token count (compression control)",
    "ku_raw": "the paper's KU (one per paper: its 18-field summary) as released (placeholders, raw claims JSON)",
    "ku_clean": "the paper's KU after KU-Fixes + WS1 cleaning (fixes applied, placeholders removed, claims parsed)",
}
SUMMARY_ORDER = ["summary_title", "summary_authors", "field_subfield", "type_of_paper", *PROSE_FIELDS, "claims"]
_MARKUP = re.compile(r"(\*\*|__|^#+\s*|^\s*[-*]\s+|```(?:json)?)", re.M)


def render_ku_raw(rec: dict[str, Any]) -> str:
    """The KU (all 18 summary fields) exactly as stored in the release, one labelled block per field."""
    parts = []
    for f in SUMMARY_ORDER:
        v = rec.get(f)
        if v is None or str(v).strip() == "":
            continue
        parts.append(f"## {f}\n{str(v).strip()}")
    return "\n\n".join(parts)


def render_ku_clean(rec: dict[str, Any]) -> tuple[str, list[str]]:
    """The WS1-cleaned view (pass the KU with KU-Fixes applied) (alexandria.schema.load_summary): resolved key aliases, placeholders / 'not provided'
    removed, markup stripped, claims parsed into a readable list. Returns (text, normaliser issues)."""
    norm = load_summary(rec)
    s = norm["summary"]
    parts = []
    for f in SUMMARY_ORDER:
        if f == "claims":
            claims = s.get("claims") or []
            lines = []
            for c in claims:
                line = f"- {c.get('description', '').strip()}"
                if c.get("supporting_evidence"):
                    line += f" Evidence: {str(c['supporting_evidence']).strip()}"
                if c.get("contradicting_evidence"):
                    line += f" Counter-evidence: {str(c['contradicting_evidence']).strip()}"
                lines.append(line)
            if lines:
                parts.append("## claims\n" + "\n".join(lines))
            continue
        v = s.get(f)
        if v is None or str(v).strip() == "":
            continue
        parts.append(f"## {f}\n{_MARKUP.sub('', str(v)).strip()}")
    return "\n\n".join(parts), list(norm.get("issues") or [])


def load_stage_table(ctx: Context, stage: str, name: str, ws: str = "ws2") -> pd.DataFrame:
    p = ctx.cfg.stage_dir(ws, stage) / name
    if not p.exists():
        raise FileNotFoundError(f"{ws}/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


def load_parts(ctx: Context, stage: str, ws: str = "ws2") -> pd.DataFrame:
    import duckdb
    d = ctx.cfg.stage_dir(ws, stage) / "parts"
    files = sorted(str(p) for p in d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    return duckdb.sql(f"select * from read_parquet({files!r}, union_by_name=true)").df()


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)
