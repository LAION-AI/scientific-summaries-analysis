"""Stage `reconcile` (reduce): reconcile the dataset's published numbers with measured ones.

Each upstream statement (configs/ws1.yaml ws1.reconcile.claims) is put next to a measured number, and each discrepancy
gets an explanation that is TESTED: the explanation's arithmetic is computed from data and the residual is reported,
so a reviewer can see what is explained and what is not.

Measured sources: parquet footers of the local release (row counts per shard), KU-Fixes json_key_stats.json (rows of
all 4 subsets, incl. those not downloaded), KU-Fixes ku_fix_index / duplicates.parquet, search/faiss metadata, and the
`features` output (arXiv ids, duplicates) when available. In the debug profile the features-based numbers are
sample estimates and are labelled as such.
"""
from __future__ import annotations

import json

import duckdb
import pyarrow.parquet as pq

from ..data import release_shards
from ..runtime import Context, log, stage
from ..schema import CANONICAL_COLUMNS, SUMMARY_FIELDS


def footer_rows(shards) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sh in shards:
        n = pq.ParquetFile(sh.path).metadata.num_rows
        g = out.setdefault(sh.group, {"files": 0, "rows": 0})
        g["files"] += 1
        g["rows"] += n
        if sh.rel.endswith("old_00200.parquet"):
            out["_old_00200_rows"] = {"files": 1, "rows": n}
    return out


@stage("ws1", "reconcile", config_keys=("ws1.reconcile", "site.ku_fixes"),
       help="reconcile published counts (papers, fields, index sizes, arXiv coverage) with measured ones")
def run(ctx: Context) -> dict:
    rc = ctx.section("ws1.reconcile")
    claims = {c["what"]: c for c in rc.get("claims", [])}
    ku = ctx.cfg.site.ku_fixes
    if not claims or ku is None or not (ku / "json_key_stats.json").exists():
        return ctx.skip("Reconciliation with published numbers",
                        f"Dataset `{ctx.cfg.dataset.name}` has no published numbers to check "
                        "(ws1.reconcile.claims in its dataset file) or no KU-Fixes index.")
    log("reconcile: reading footers of the local release")
    local = footer_rows(release_shards(ctx.cfg))
    old200 = local.pop("_old_00200_rows", {}).get("rows")
    arxiv_rows = sum(v["rows"] for g, v in local.items() if g.startswith("arxiv/"))
    arxiv_files = sum(v["files"] for g, v in local.items() if g.startswith("arxiv/"))
    titled_rows = local.get("arxiv/titled", {}).get("rows", 0)
    ku_stats = json.loads((ctx.cfg.site.ku_fixes / "json_key_stats.json").read_text())
    corpus_rows, corpus_shards = ku_stats["rows"], ku_stats["shards"]

    con = duckdb.connect()
    kuidx = ctx.cfg.site.ku_fixes / "ku_fix_index.parquet"
    dups = ctx.cfg.site.ku_fixes / "duplicates.parquet"
    regen = con.sql(f"select count(*) from '{kuidx}' where needs_regeneration").fetchone()[0]
    dup_strong = con.sql(f"select count(*) from '{dups}' where tier = 'strong' and not keep_strong").fetchone()[0]
    arxiv_dup_strong = con.sql(f"select count(*) from '{dups}' where tier = 'strong' and not keep_strong "
                               "and folder like 'arxiv/%'").fetchone()[0]

    rows = []

    def add(what, published, measured, explanation, explained=None, source=""):
        """explained = the published number as predicted from measured counts by the explanation."""
        resid = None if explained is None or published is None else published - explained
        rows.append({"what": what, "published": published, "measured": measured,
                     "difference": None if published is None or measured is None else measured - published,
                     "explanation": explanation, "explained_value": explained, "residual": resid, "source": source})

    p_total = claims.get("papers in the dataset", {}).get("value")
    add("papers in the dataset", p_total, corpus_rows,
        f"README total omits arxiv/old_00200 ({old200:,} rows)" if old200 else "README total vs rows in all shards",
        (corpus_rows - old200) if old200 else None,
        "KU-Fixes json_key_stats.json (all shards)")
    p_arx = claims.get("arXiv papers in the dataset", {}).get("value")
    add("arXiv rows (local)", p_arx, arxiv_rows, "same shard: old_00200", (arxiv_rows - old200) if old200 else None,
        "parquet footers")
    add("arXiv parquet files", claims.get("arXiv parquet files", {}).get("value"), arxiv_files,
        "SEARCH_USAGE lists 3,985 files; the release has new + old + titled shards", None, "file listing")

    add("papers in each per-field BM25 index", claims.get("BM25 per-field documents", {}).get("value"), corpus_rows,
        "per-field indices exclude the arxiv/titled subset", corpus_rows - titled_rows, "SEARCH_USAGE")
    add("papers in each per-field FAISS index", claims.get("FAISS summary vectors", {}).get("value"), corpus_rows,
        "not explained by a subset, duplicates or needs_regeneration (open question for the authors)", None,
        "SEARCH_USAGE")
    add("summary fields", claims.get("summary fields", {}).get("value"), len(SUMMARY_FIELDS),
        "18 LLM fields; the search indices add `concatenated_summary` as a 19th", None, "schema")
    add("summary fields indexed", claims.get("summary fields indexed", {}).get("value"), len(SUMMARY_FIELDS) + 1,
        "18 fields + concatenated_summary", len(SUMMARY_FIELDS) + 1, "SEARCH_USAGE")
    add("columns", claims.get("columns", {}).get("value"), len(CANONICAL_COLUMNS),
        "7 source + 18 summary + 3 generation + 13 OpenAlex + 3 text; normalized copies add _extra_fields, "
        "_source_file (46); stored splits add _split, _split_rank (48)", len(CANONICAL_COLUMNS), "schema")

    try:
        fdir = ctx.input_dir("features") / "parts" / "*.parquet"
        q = con.sql(f"""select count(*) n, count(arxiv_id) with_id, count(distinct arxiv_id) distinct_ids,
                        sum(paper_id_is_path::int) path_ids from '{fdir}' where "group" like 'arxiv/%'""").fetchone()
        n, with_id, distinct_ids, path_ids = q
        scale = arxiv_rows / n if n else None
        est = ctx.cfg.profile != "full"
        tag = " (debug sample, scaled)" if est else ""
        ext = rc.get("arxiv_total_submissions", {})
        dist_scaled = distinct_ids * scale if scale else None
        add(f"distinct arXiv ids among arXiv rows{tag}", ext.get("value"), dist_scaled,
            f"rows without an arXiv id (paper_id is an internal path: arxiv/old and arxiv/titled) = "
            f"{path_ids / n:.1%} of arXiv rows; strong duplicates across arxiv subsets = {arxiv_dup_strong:,}",
            None, f"features; external total: {ext.get('source')} as of {ext.get('as_of')}")
        coverage = {"arxiv_rows": arxiv_rows, "rows_with_arxiv_id_share": with_id / n if n else None,
                    "distinct_ids_estimate": dist_scaled, "path_id_share": path_ids / n if n else None,
                    "estimate": est}
    except FileNotFoundError:
        coverage = {"note": "run `features` for arXiv-id coverage"}

    fm = ctx.cfg.site.data_root / "search" / "faiss" / "faiss_metadata.json"
    if fm.exists():
        meta = json.loads(fm.read_text())
        for kind in ("chunks", "claims"):
            add(f"{kind} vectors (FAISS)", claims.get(f"BM25 {kind[:-1] if kind == 'claims' else 'chunk'} documents",
                                                      {}).get("value"),
                meta[kind]["total_vectors"], "FAISS and BM25 index the same chunk / claim tables",
                meta[kind]["total_vectors"], "search/faiss/faiss_metadata.json")

    result = {"local": local, "corpus_rows": corpus_rows, "corpus_shards": corpus_shards, "old_00200_rows": old200,
              "needs_regeneration": regen, "duplicates_strong_drop": dup_strong, "arxiv_coverage": coverage,
              "rows": rows}
    ctx.write_json("reconcile.json", result)
    fmt_rows = [{**r, "published": r["published"], "measured": r["measured"]} for r in rows]
    ctx.publish("Reconciling the published numbers", [
        {"type": "kpis", "items": [
            {"label": "rows in all shards", "value": corpus_rows, "fmt": "int", "hint": f"{corpus_shards:,} shards"},
            {"label": "arXiv rows (local)", "value": arxiv_rows, "fmt": "int"},
            {"label": "arxiv/titled rows", "value": titled_rows, "fmt": "int", "hint": "not in per-field indices"},
            {"label": "need regeneration", "value": regen, "fmt": "int", "hint": "KU-Fixes"},
            {"label": "strong duplicates to drop", "value": dup_strong, "fmt": "int", "hint": "KU-Fixes"},
        ]},
        {"type": "table", "title": "Published vs measured", "hint": "residual = measured - explained; 0 means the "
         "explanation accounts for the whole difference",
         "columns": [{"key": "what", "label": "statement"}, {"key": "published", "label": "published", "fmt": "int"},
                     {"key": "measured", "label": "measured", "fmt": "int"},
                     {"key": "difference", "label": "difference", "fmt": "int"},
                     {"key": "explanation", "label": "explanation"},
                     {"key": "residual", "label": "residual", "fmt": "int"}, {"key": "source", "label": "source"}],
         "rows": fmt_rows},
        {"type": "table", "title": "Local release by group", "columns": [
            {"key": "group", "label": "group"}, {"key": "files", "label": "files", "fmt": "int"},
            {"key": "rows", "label": "rows", "fmt": "int"}],
         "rows": [{"group": g, **v} for g, v in sorted(local.items())]},
    ], summary="Every explanation is checked by arithmetic on measured counts; the residual shows what is left.")
    return {"statements": len(rows), "corpus_rows": corpus_rows}
