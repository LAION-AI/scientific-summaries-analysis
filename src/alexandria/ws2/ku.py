"""Stage `ku` (WS2): the knowledge unit (KU) of each evaluation paper.

In laion/Scientific-Summaries one row = one paper = one KU: the paper's 18-field structured summary (SUMMARY_FIELDS;
the KU-Fixes errata count 22.1M KUs for 22.1M papers). Nothing is generated here. Per paper the stage stores
  ku_raw    the 18 fields exactly as released
  ku_fixed  the same KU with the official KU-Fixes applied (site.ku_fixes, ku_fix_index.parquet): title / authors
            replaced by the index's fix (placeholders -> fix or null), claims sub-keys normalised (claims_fixed)
plus the KU's fault flags (title / authors / JSON faults, needs_regeneration) and duplicate-cluster status.
Exactly one KU per paper is enforced. Datasets without KU-Fixes get ku_fixed = ku_raw.
"""
from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from ..data import fetch_records
from ..runtime import Context, log, stage
from ..schema import SUMMARY_FIELDS
from .common import load_stage_table

FIX_COLS = ["file", "row_index", "title_fault", "title_fixed", "authors_fault", "authors_fixed", "json_faults",
            "claims_fixed", "needs_regeneration", "n_blank_summary_fields"]
DUP_COLS = ["file", "row_index", "tier", "cluster_size", "keep", "keep_strong"]


def _lookup(path, cols: list[str], files: list[str]) -> dict[tuple[str, int], dict]:
    if path is None or not path.exists():
        return {}
    rows = pq.read_table(path, columns=cols, filters=[("file", "in", files)]).to_pylist()
    return {(r["file"], int(r["row_index"])): r for r in rows}


def apply_fixes(ku: dict, fix: dict | None) -> tuple[dict, list[str]]:
    """The KU with the KU-Fixes corrections of its index row (None = KU not in the index, nothing to fix)."""
    out, applied = dict(ku), []
    if not fix:
        return out, applied
    if fix.get("title_fault"):
        out["summary_title"] = fix.get("title_fixed")
        applied.append(f"title:{fix['title_fault']}")
    if fix.get("authors_fault"):
        out["summary_authors"] = fix.get("authors_fixed")
        applied.append(f"authors:{fix['authors_fault']}")
    if fix.get("json_faults") and fix.get("claims_fixed") is not None:
        out["claims"] = fix["claims_fixed"]
        applied.append("claims")
    return out, applied


def filled(ku: dict) -> int:
    return sum(1 for f in SUMMARY_FIELDS if ku.get(f) not in (None, "", "[]"))


@stage("ws2", "ku", needs=("papers",), config_keys=("ws2.ku",),
       help="one KU per paper: the dataset row's 18-field structured summary, as released and with KU-Fixes applied")
def run(ctx: Context) -> dict:
    papers = load_stage_table(ctx, "papers", "papers.parquet")
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)), ["paper_id", *SUMMARY_FIELDS])
    root, files = ctx.cfg.site.ku_fixes, sorted(papers["file"].unique())
    fixes = _lookup(root / "ku_fix_index.parquet" if root else None, FIX_COLS, files)
    dups = _lookup(root / "duplicates.parquet" if root else None, DUP_COLS, files)
    log(f"ku: {len(papers)} papers; KU-Fixes {'on' if root else 'off'} ({len(fixes)} faulty, {len(dups)} in dup clusters)")
    out = []
    for p in papers.itertuples():
        key = (p.file, int(p.row))
        rec = recs.get(key)
        if rec is None:
            raise KeyError(f"paper {p.paper_key} ({p.file} row {p.row}) not found in the dataset")
        raw = {f: rec.get(f) for f in SUMMARY_FIELDS}
        fix, dup = fixes.get(key), dups.get(key)
        fixed, applied = apply_fixes(raw, fix)
        out.append({"paper_key": p.paper_key, "ku_id": p.paper_key, "file": p.file, "row": int(p.row),
                    "paper_id": rec.get("paper_id"),
                    "ku_raw": json.dumps(raw, ensure_ascii=False, default=str),
                    "ku_fixed": json.dumps(fixed, ensure_ascii=False, default=str),
                    "fields_filled": filled(raw), "fixes_applied": ",".join(applied) or None,
                    "title_fault": (fix or {}).get("title_fault"), "authors_fault": (fix or {}).get("authors_fault"),
                    "json_faults": (fix or {}).get("json_faults"),
                    "needs_regeneration": bool((fix or {}).get("needs_regeneration")),
                    "dup_tier": (dup or {}).get("tier"), "dup_keep": None if dup is None else bool(dup.get("keep"))})
    kus = pd.DataFrame(out)
    per = kus.groupby("paper_key").size()
    if len(kus) != len(papers) or per.max() != 1 or set(per.index) != set(papers["paper_key"]):
        raise ValueError(f"expected exactly one KU per paper: {len(kus)} KUs for {len(papers)} papers")
    kus.to_parquet(ctx.out / "kus.parquet", index=False)
    n_fixed = int(kus["fixes_applied"].notna().sum())
    ctx.publish("WS2 knowledge units", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(papers), "fmt": "int"},
            {"label": "KUs", "value": len(kus), "fmt": "int", "hint": "one per paper (the dataset row's 18-field summary)"},
            {"label": "KUs fixed by KU-Fixes", "value": n_fixed, "fmt": "int"},
            {"label": "need regeneration", "value": int(kus["needs_regeneration"].sum()), "fmt": "int"},
            {"label": "in a duplicate cluster", "value": int(kus["dup_tier"].notna().sum()), "fmt": "int"},
            {"label": "fields filled (median of 18)", "value": float(kus["fields_filled"].median()), "fmt": "int"}]},
        {"type": "table", "title": "KUs changed by KU-Fixes",
         "columns": [{"key": "paper_key", "label": "paper"}, {"key": "paper_id", "label": "paper_id"},
                     {"key": "fixes_applied", "label": "fixes"}, {"key": "fields_filled", "label": "fields filled", "fmt": "int"}],
         "rows": kus[kus["fixes_applied"].notna()][["paper_key", "paper_id", "fixes_applied", "fields_filled"]]
                .to_dict("records")},
    ], summary=f"{len(kus)} KUs for {len(papers)} papers ({n_fixed} fixed)")
    return {"kus": len(kus), "papers": len(papers), "fixed": n_fixed}
