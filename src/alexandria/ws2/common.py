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
    # structure ablation (stage `ablation-gen` + `contexts`): same model / same length structured vs prose, and
    # field-group ablations of the released (clean) KU
    "ku_local": "KU written by the local generator from the same source (18-field schema, rendered like ku_clean)",
    "prose_local": "unstructured prose summary by the same local generator, same source, length-matched to ku_local",
    "ku_minus_context": "ku_clean without executive summary, research context, research question",
    "ku_minus_methods": "ku_clean without methodological details and procedures",
    "ku_minus_results": "ku_clean without key results and interpretation",
    "ku_minus_claims": "ku_clean without the claims list",
    "ku_minus_limitations": "ku_clean without contradictions and limitations",
    "ku_minus_takeaways": "ku_clean without the three take-aways",
    "ku_results_claims": "ku_clean reduced to key results, interpretation and claims (+ bibliographic fields)",
    "ku_flat": "ku_clean content with headings removed and field order shuffled (structure-vs-content control)",
}
BASE_ARMS = ["closed_book", "full_text", "full_text_matched", "ku_raw", "ku_clean"]
BIBLIOGRAPHIC = ["summary_title", "summary_authors", "field_subfield", "type_of_paper"]
ABLATION_GROUPS = {          # field groups dropped one at a time (ku_minus_<group>)
    "context": ["executive_summary", "research_context", "research_question_hypothesis"],
    "methods": ["methodological_details", "procedures_architectures"],
    "results": ["key_results", "interpretation_implications"],
    "claims": ["claims"],
    "limitations": ["contradictions_limitations"],
    "takeaways": ["three_takeaways"],
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


def render_ku_clean(rec: dict[str, Any], keep: list[str] | None = None, drop: list[str] = (),
                    shuffle_seed: int | None = None, headings: bool = True) -> tuple[str, list[str]]:
    """The WS1-cleaned view (pass the KU with KU-Fixes applied) (alexandria.schema.load_summary): resolved key
    aliases, placeholders / 'not provided' removed, markup stripped, claims parsed into a readable list.
    Ablations: `keep` (only these fields), `drop` (all but these), `shuffle_seed` (random field order) and
    `headings=False` (no field labels, claims as plain sentences). Returns (text, normaliser issues)."""
    norm = load_summary(rec)
    s = norm["summary"]
    blocks = []
    for f in SUMMARY_ORDER:
        if (keep is not None and f not in keep) or f in drop:
            continue
        if f == "claims":
            lines = []
            for c in s.get("claims") or []:
                line = c.get("description", "").strip()
                if c.get("supporting_evidence"):
                    line += f" Evidence: {str(c['supporting_evidence']).strip()}"
                if c.get("contradicting_evidence"):
                    line += f" Counter-evidence: {str(c['contradicting_evidence']).strip()}"
                lines.append(f"- {line}" if headings else line)
            if lines:
                blocks.append((f, "\n".join(lines)))
            continue
        v = s.get(f)
        if v is None or str(v).strip() == "":
            continue
        blocks.append((f, _MARKUP.sub("", str(v)).strip()))
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(blocks)
    text = "\n\n".join(f"## {f}\n{v}" if headings else v for f, v in blocks)
    return text, list(norm.get("issues") or [])


def ablation_texts(rec: dict[str, Any], seed: int) -> dict[str, str]:
    """The field-group ablations of one clean KU (arms ku_minus_<group>, ku_results_claims, ku_flat)."""
    out = {f"ku_minus_{g}": render_ku_clean(rec, drop=fields)[0] for g, fields in ABLATION_GROUPS.items()}
    keep = BIBLIOGRAPHIC + ABLATION_GROUPS["results"] + ABLATION_GROUPS["claims"]
    out["ku_results_claims"] = render_ku_clean(rec, keep=keep)[0]
    out["ku_flat"] = render_ku_clean(rec, shuffle_seed=seed, headings=False)[0]
    return out


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
