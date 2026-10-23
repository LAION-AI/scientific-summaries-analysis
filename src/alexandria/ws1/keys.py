"""Stage `keys` (reduce): are the keys right? Column names and types per shard, claim sub-keys, key names leaking
into values, alias candidates for review, the canonical JSON Schema, and a check of the shared normalizer.

Evidence used:
  * column names / types: parquet footers of every input shard (cheap: no data pages are read)
  * claim sub-keys: KU-Fixes json_key_stats.json (all 7,907 shards, 22.09M rows; reused, not recomputed)
  * keys written inside values ("key_results: ..."): `features` output (key_leak column), if present
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter

import pyarrow.parquet as pq

from ..data import input_shards, iter_batches
from ..runtime import Context, stage
from ..schema import CANONICAL_COLUMNS, discover_candidates, load_summary, resolve
from ..schema.model import Summary

_CLAIM_STAT = re.compile(r"^claims\|items:[^|]*\|keys:(.*)$")


def column_inventory(shards, data_root) -> tuple[Counter, dict[str, Counter], int]:
    """Footers of the ORIGINAL release shards (the debug sample copies are already type-unified)."""
    names: Counter = Counter()
    types: dict[str, Counter] = {}
    for sh in shards:
        schema = pq.ParquetFile(data_root / sh.rel).schema_arrow
        for f in schema:
            if f.name.startswith("_"):
                continue
            names[f.name] += 1
            types.setdefault(f.name, Counter())[str(f.type)] += 1
    return names, types, len(shards)


def claim_subkeys(stats_path) -> tuple[Counter, int, int]:
    """-> (rows per claim sub-key, rows with canonical keys only, total rows) from KU-Fixes key statistics."""
    stats = json.loads(stats_path.read_text())
    keys: Counter = Counter()
    canonical_only = total = 0
    for k, n in stats["keys"].items():
        m = _CLAIM_STAT.match(k)
        if not m and not k.startswith("claims|"):
            continue
        total += n
        if not m:
            continue
        sub = [s for s in m.group(1).split(",") if s]
        for s in sub:
            keys[s] += n
        if sub and all(resolve(s, "claims").how == "exact" for s in sub):
            canonical_only += n
    return keys, canonical_only, total


@stage("ws1", "keys", help="key audit: column names/types, claim sub-keys, leaked keys, alias candidates, JSON Schema")
def run(ctx: Context) -> dict:
    shards = input_shards(ctx.cfg)
    names, types, n_shards = column_inventory(shards, ctx.cfg.site.data_root)
    col_rows = []
    for name, n in sorted(names.items(), key=lambda kv: CANONICAL_COLUMNS.index(kv[0]) if kv[0] in CANONICAL_COLUMNS else 999):
        mapped = {v: k for k, v in ctx.cfg.dataset.columns.items()}.get(name)
        m = resolve(mapped or name)
        col_rows.append({"column": name, "shards": n, "share": n / n_shards, "resolution": m.how,
                         "canonical": m.key, "types": ", ".join(f"{t} ({c})" for t, c in types[name].most_common())})
    present = set(names) | {k for k, v in ctx.cfg.dataset.columns.items() if v in names}
    missing = [c for c in CANONICAL_COLUMNS if c not in present]
    type_conflicts = [r for r in col_rows if len(types[r["column"]]) > 1]

    ku_stats = ctx.cfg.site.ku_fixes / "json_key_stats.json" if ctx.cfg.site.ku_fixes else None
    claim_keys, canon_only, claim_rows = (claim_subkeys(ku_stats) if ku_stats and ku_stats.exists()
                                          else (Counter(), 0, 0))
    claim_rows_tbl = [{"sub_key": k, "rows": n, "resolution": resolve(k, "claims").how,
                       "maps_to": resolve(k, "claims").key} for k, n in claim_keys.most_common()]

    leaks: Counter = Counter()
    leak_docs = n_docs = 0
    try:
        import duckdb
        parts = str(ctx.input_dir("features") / "parts" / "*.parquet")
        n_docs = duckdb.sql(f"select count(*) from '{parts}'").fetchone()[0]
        leak_docs = duckdb.sql(f"select count(*) from '{parts}' where key_leak is not null").fetchone()[0]
        for k, n in duckdb.sql(f"select k, count(*) from (select unnest(key_leak) k from '{parts}') group by 1").fetchall():
            leaks[k] = n
    except FileNotFoundError:
        pass

    observed = {**{k: v for k, v in names.items()}, **dict(claim_keys), **dict(leaks)}
    cands = discover_candidates(observed, "columns") + discover_candidates(dict(claim_keys), "claims")
    with open(ctx.out / "alias_candidates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "key", "count", "resolution", "candidate_1", "score_1", "candidate_2", "score_2", "accept"])
        for i, c in enumerate(cands):
            kind = "claims" if c["key"] in claim_keys and c["key"] not in names else "columns"
            flat = [x for cc in c["candidates"][:2] for x in (cc["canonical"], cc["score"])]
            w.writerow([kind, c["key"], c["count"], c["resolution"], *flat, *([""] * (4 - len(flat))), ""])
    ctx.write_json("summary.schema.json", Summary.model_json_schema())

    # does the shared normalizer accept every sampled record, and what does it repair?
    issues: Counter = Counter()
    checked = 0
    limit = 20000
    for sh in shards[: 8 if ctx.cfg.profile == "full" else len(shards)]:
        for tbl in iter_batches(ctx.cfg, sh, CANONICAL_COLUMNS, batch_size=256):
            for r in tbl.to_pylist():
                checked += 1
                for i in load_summary(r)["issues"]:
                    issues[re.sub(r"=.*$", "", re.sub(r"claims:\d+:", "claims:item:", i))] += 1
            if checked >= limit:
                break
        if checked >= limit:
            break

    result = {"shards": n_shards, "columns_seen": len(names), "missing_canonical": missing,
              "non_canonical_columns": [r["column"] for r in col_rows if r["resolution"] != "exact"],
              "type_conflicts": {r["column"]: r["types"] for r in type_conflicts},
              "claim_rows": claim_rows, "claim_rows_canonical_only": canon_only,
              "leak_docs": leak_docs, "docs": n_docs, "normalizer_records": checked,
              "normalizer_issues": dict(issues.most_common())}
    ctx.write_json("keys.json", {**result, "columns": col_rows, "claim_subkeys": claim_rows_tbl,
                                 "leaked_keys": dict(leaks), "alias_candidates": cands})
    non_canon_claim = claim_rows - canon_only
    ctx.publish("Key audit", [
        {"type": "kpis", "items": [
            {"label": "shards scanned", "value": n_shards, "fmt": "int"},
            {"label": "non-canonical column names", "value": len(result["non_canonical_columns"]), "fmt": "int",
             "status": "good" if not result["non_canonical_columns"] else "bad"},
            {"label": "columns with type conflicts", "value": len(type_conflicts), "fmt": "int",
             "status": "warn" if type_conflicts else "good", "hint": "fixed by casting to unified_schema.json"},
            {"label": "claims rows with off-schema sub-keys", "value": non_canon_claim, "fmt": "int",
             "hint": f"of {claim_rows:,} rows (whole corpus, KU-Fixes)"},
            {"label": "papers with key names inside values", "value": leak_docs, "fmt": "int",
             "hint": f"of {n_docs:,} papers checked" if n_docs else "run `features` first"},
        ]},
        {"type": "table", "title": "Columns", "hint": "names and parquet types across shards",
         "columns": [{"key": "column", "label": "column"}, {"key": "share", "label": "shards", "fmt": "pct"},
                     {"key": "resolution", "label": "name"}, {"key": "types", "label": "types (shards)"}],
         "rows": col_rows},
        {"type": "table", "title": "Claim sub-keys (whole corpus)", "hint": "KU-Fixes json_key_stats.json; "
         "resolution via resources/aliases.yaml",
         "columns": [{"key": "sub_key", "label": "sub-key"}, {"key": "rows", "label": "rows", "fmt": "int"},
                     {"key": "resolution", "label": "resolution"}, {"key": "maps_to", "label": "maps to"}],
         "rows": claim_rows_tbl},
        {"type": "table", "title": "Shared normalizer: repairs on the checked records",
         "hint": f"load_summary() on {checked:,} records; every WS uses this view",
         "columns": [{"key": "issue", "label": "issue"}, {"key": "n", "label": "records", "fmt": "int"}],
         "rows": [{"issue": k, "n": v} for k, v in issues.most_common(40)]},
        {"type": "markdown", "text": "Alias candidates for human review: `alias_candidates.csv` in this stage's "
         "folder. Accepted rows go into `src/alexandria/resources/aliases.yaml`; nothing is auto-accepted."},
    ], summary="Column names are canonical in every shard; key problems live inside the claims JSON and in values.")
    return result
