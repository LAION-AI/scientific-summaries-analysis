"""Stages `lexical` (map) and `lexical-report` (reduce): verbatim overlap and number / entity precision against the
source text.

Why: the release is described as summaries, not excerpts, so long verbatim spans matter (copyright, memorisation);
and numbers / named things in a summary that never occur in the source are cheap, high-precision hallucination
evidence. Only records with source text are measured (text_raw, else text_sanitized, capped at max_source_chars);
the share without text is reported, never imputed.

Per record and field (output is WIDE, one row per record):
    lcs__<f>      longest common token span with the source, in tokens (textutil.tokens: lower-case, ascii-folded,
                  punctuation dropped). Computed from k-gram membership, k = min(span_lengths): the longest run r of
                  consecutive field k-grams that all occur in the source gives span r + k - 1. Exact for a span copied
                  from one passage; a run could in principle stitch two adjacent source passages, so it is an upper
                  bound in that rare case. Spans shorter than k are reported as 0. Null = empty field.
    tok__<f>      field length in tokens
    num_n__<f>, num_ok__<f>   numeric literals in the field / found among the source's numeric literals
    ent_n__<f>, ent_ok__<f>   entity-like strings (acronyms, mixed-case / alphanumeric names) / found verbatim
                  (case-sensitive substring) in the source
The has-match-at-k flags of the plan are lcs__<f> >= k, derived in the report (so any k can be added later).
`claims` = every parsed claim's description, evidence and implications joined. input_type comes from `features`
(joined on file, row), never recomputed.
"""
from __future__ import annotations

import pyarrow as pa

from .. import textutil as tu
from ..config import Config
from ..data import Shard, iter_batches
from ..runtime import Context, stage
from ..schema import parse_claims
from ..stats import rate_row

COMPARISON_NOTE = ("For comparison, the SCI-JEPA KU run found 12-token verbatim matches in 94–97% of documents, "
                   "with spans up to 86 tokens.")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def kgram_set(toks: list[str], k: int) -> set[int]:
    return {hash(tuple(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def longest_span(field_toks: list[str], source_kgrams: set[int], k: int) -> int:
    """Longest verbatim span (tokens) of field_toks in the source, from k-gram membership; 0 if shorter than k."""
    best = run = 0
    for i in range(len(field_toks) - k + 1):
        if hash(tuple(field_toks[i:i + k])) in source_kgrams:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best + k - 1 if best else 0


def number_support(field: str | None, source_numbers: set[str]) -> tuple[int, int]:
    nums = tu.numbers(field)
    return len(nums), len(nums & source_numbers)


def entity_support(field: str | None, source: str) -> tuple[int, int]:
    ents = tu.entities(field)
    return len(ents), sum(1 for e in ents if e in source)


def claims_text(raw: str | None) -> str:
    claims, _ = parse_claims(raw, [])
    return "\n".join(" ".join(v for v in (c["description"], c["supporting_evidence"], c["contradicting_evidence"],
                                          c["implications"]) if v) for c in claims)


def field_measures(text: str | None, src_text: str, src_kgrams: set[int], src_numbers: set[str], k: int) -> dict:
    toks = tu.tokens(text)
    if not toks:
        return {"lcs": None, "tok": 0, "num_n": 0, "num_ok": 0, "ent_n": 0, "ent_ok": 0}
    num_n, num_ok = number_support(text, src_numbers)
    ent_n, ent_ok = entity_support(text, src_text)
    return {"lcs": longest_span(toks, src_kgrams, k), "tok": len(toks), "num_n": num_n, "num_ok": num_ok,
            "ent_n": ent_n, "ent_ok": ent_ok}


# ---------------------------------------------------------------- map
def lexical_shard(cfg: Config, shard: Shard) -> pa.Table:
    lx = cfg.raw["ws1"].get("lexical") or {}
    fields = list(lx.get("fields") or ["executive_summary", "key_results", "claims"])
    k = int(min(lx.get("span_lengths") or [8, 12, 20]))
    cap = int(lx.get("max_source_chars", 100000))
    cols: dict[str, list] = {"file": [], "row": [], "group": [], "has_source": [], "src_tokens": []}
    for f in fields:
        for m in ("lcs", "tok", "num_n", "num_ok", "ent_n", "ent_ok"):
            cols[f"{m}__{f}"] = []
    for tbl in iter_batches(cfg, shard, ["text_raw", "text_sanitized", *fields], batch_size=256):
        for r in tbl.to_pylist():
            src = ((r.get("text_raw") or "").strip() and r["text_raw"]) or r.get("text_sanitized") or ""
            src = src[:cap]
            cols["file"].append(r["_file"]), cols["row"].append(r["_row"]), cols["group"].append(r["_group"])
            has = bool(src.strip())
            src_toks = tu.tokens(src) if has else []
            cols["has_source"].append(has), cols["src_tokens"].append(len(src_toks))
            kg, nums = (kgram_set(src_toks, k), tu.numbers(src)) if has else (set(), set())
            for f in fields:
                text = claims_text(r.get(f)) if f == "claims" else r.get(f)
                meas = field_measures(text, src, kg, nums, k) if has else dict.fromkeys(
                    ("lcs", "tok", "num_n", "num_ok", "ent_n", "ent_ok"))
                for m, v in meas.items():
                    cols[f"{m}__{f}"].append(v)
    types = {"file": pa.string(), "row": pa.int64(), "group": pa.string(), "has_source": pa.bool_()}
    return pa.table({c: pa.array(v, types.get(c, pa.int64())) for c, v in cols.items()})


@stage("ws1", "lexical", kind="map", resources="cpu", config_keys=("ws1.lexical",),
       help="verbatim overlap + number/entity precision of summary fields vs the source text")
def run_lexical(ctx: Context) -> dict:
    return ctx.map_shards(lexical_shard)


# ---------------------------------------------------------------- reduce
def _latex(df, path) -> None:
    try:
        df.to_latex(path, index=False, float_format="%.3f")
    except ImportError:                                    # pragma: no cover  (jinja2 missing)
        with open(path, "w") as fh:
            fh.write(" & ".join(df.columns) + r" \\" + "\n")
            for row in df.itertuples(index=False):
                fh.write(" & ".join(str(x) for x in row) + r" \\" + "\n")


@stage("ws1", "lexical-report", kind="reduce", needs=("lexical", "features"), resources="cpu",
       config_keys=("ws1.lexical",), help="verbatim-overlap and precision tables by group / input type / field")
def run_report(ctx: Context) -> dict:
    import duckdb
    import pandas as pd
    lx = ctx.section("ws1.lexical")
    ks = sorted(int(x) for x in (lx.get("span_lengths") or [8, 12, 20]))
    lex = str(ctx.input_dir("lexical") / "parts" / "*.parquet")
    feat = str(ctx.input_dir("features") / "parts" / "*.parquet")
    con = duckdb.connect()
    con.execute(f"""CREATE VIEW j AS SELECT l.*, coalesce(f.input_type, 'unknown') AS input_type
                    FROM read_parquet('{lex}') l LEFT JOIN read_parquet('{feat}', union_by_name=true) f USING (file, row)""")
    fields = [c.split("__", 1)[1] for c in con.execute("SELECT * FROM j LIMIT 0").fetchdf().columns
              if c.startswith("lcs__")]
    lcs_list = "list_max([" + ", ".join(f"lcs__{f}" for f in fields) + "])"
    con.execute(f"CREATE VIEW d AS SELECT *, {lcs_list} AS lcs_doc FROM j")

    coverage = con.execute("""SELECT "group", count(*) n, sum(has_source::INT) n_src FROM j GROUP BY 1 ORDER BY 1""").fetchall()

    def span_rows(by: str, col: str) -> list[dict]:
        rows = []
        q = ", ".join(f"sum(({col} >= {k})::INT)" for k in ks)
        for r in con.execute(f"""SELECT {by}, count({col}), {q}, median({col}), quantile_cont({col}, 0.95), max({col})
                                 FROM d WHERE {col} IS NOT NULL GROUP BY ROLLUP({by}) ORDER BY {by} NULLS LAST""").fetchall():
            key, n = r[0] if r[0] is not None else "all", int(r[1])
            row = {by: key, "n": n}
            for i, k in enumerate(ks):
                rr = rate_row(r[2 + i] or 0, n)
                row.update({f"match{k}": rr["rate"], f"match{k}_lo": rr["lo"], f"match{k}_hi": rr["hi"]})
            row.update(median=r[2 + len(ks)], p95=r[3 + len(ks)], max=r[4 + len(ks)])
            rows.append(row)
        return rows

    doc_by_group, doc_by_type = span_rows('"group"', "lcs_doc"), span_rows("input_type", "lcs_doc")
    doc_by_group = [{"group": r.pop('"group"'), **r} for r in doc_by_group]

    field_rows = []
    for f in fields:
        q = ", ".join(f"sum((lcs__{f} >= {k})::INT)" for k in ks)
        n, *rest = con.execute(f"""SELECT count(lcs__{f}), {q}, median(lcs__{f}), quantile_cont(lcs__{f}, 0.95),
                                   max(lcs__{f}), sum(num_n__{f}), sum(num_ok__{f}), sum(ent_n__{f}), sum(ent_ok__{f})
                                   FROM j WHERE lcs__{f} IS NOT NULL""").fetchone()
        row = {"field": f, "n": int(n or 0)}
        for i, k in enumerate(ks):
            row[f"match{k}"] = rate_row(rest[i] or 0, n or 0)["rate"]
        med, p95, mx, nn, nok, en, eok = rest[len(ks):]
        num, ent = rate_row(nok or 0, nn or 0), rate_row(eok or 0, en or 0)
        row.update(median=med, p95=p95, max=mx, numbers=int(nn or 0), numbers_ok=int(nok or 0),
                   entities_ok=int(eok or 0), number_precision=num["rate"],
                   number_lo=num["lo"], number_hi=num["hi"], entities=int(en or 0), entity_precision=ent["rate"],
                   entity_lo=ent["lo"], entity_hi=ent["hi"])
        field_rows.append(row)

    prec_type = []
    for itype, nn, nok, en, eok in con.execute(
            "SELECT input_type, " + ", ".join(
                f"sum({'+'.join(f'coalesce({m}__{f}, 0)' for f in fields)})" for m in ("num_n", "num_ok", "ent_n", "ent_ok"))
            + " FROM j WHERE has_source GROUP BY 1 ORDER BY 1").fetchall():
        num, ent = rate_row(nok or 0, nn or 0), rate_row(eok or 0, en or 0)
        prec_type.append({"input_type": itype, "numbers": int(nn or 0), "number_precision": num["rate"],
                          "number_lo": num["lo"], "number_hi": num["hi"], "entities": int(en or 0),
                          "entity_precision": ent["rate"], "entity_lo": ent["lo"], "entity_hi": ent["hi"]})

    for name, rows in (("spans_by_group", doc_by_group), ("spans_by_input_type", doc_by_type),
                       ("by_field", field_rows), ("precision_by_input_type", prec_type)):
        df = pd.DataFrame(rows)
        df.to_csv(ctx.out / f"{name}.csv", index=False)
        _latex(df, ctx.out / f"{name}.tex")

    allrow = next((r for r in doc_by_group if r["group"] == "all"), {"n": 0})
    n_tot = sum(int(n) for _, n, _ in coverage)
    n_src = sum(int(s or 0) for *_, s in coverage)
    span_cols = [{"key": f"match{k}", "label": f">= {k} tok", "fmt": "pct1"} for k in ks] + [
        {"key": "median", "label": "median span", "fmt": "float"}, {"key": "p95", "label": "p95", "fmt": "float"},
        {"key": "max", "label": "max", "fmt": "int"}]
    k12 = 12 if 12 in ks else ks[len(ks) // 2]
    ctx.publish("Verbatim overlap and number / entity precision", [
        {"type": "kpis", "items": [
            {"label": "records with source text", "value": (n_src / n_tot) if n_tot else 0.0, "fmt": "pct1",
             "hint": f"{n_src:,} of {n_tot:,}; the rest cannot be measured"},
            {"label": f"docs with a >= {k12}-token verbatim span", "value": allrow.get(f"match{k12}"), "fmt": "pct1"},
            {"label": "longest span (tokens)", "value": allrow.get("max"), "fmt": "int"},
            {"label": "number precision (all fields)",
             "value": rate_row(sum(r["numbers_ok"] for r in field_rows),
                               sum(r["numbers"] for r in field_rows))["rate"], "fmt": "pct1"},
            {"label": "entity precision (all fields)",
             "value": rate_row(sum(r["entities_ok"] for r in field_rows),
                               sum(r["entities"] for r in field_rows))["rate"], "fmt": "pct1"}]},
        {"type": "markdown", "text": f"Document level = longest span over all measured fields. {COMPARISON_NOTE}"},
        {"type": "bar", "title": "Documents with a >= k-token verbatim span, by group", "fmt": "pct1",
         "hint": "95% Wilson intervals", "labels": [r["group"] for r in doc_by_group],
         "series": [{"name": f">= {k}", "data": [r[f"match{k}"] for r in doc_by_group],
                     "lo": [r[f"match{k}_lo"] for r in doc_by_group], "hi": [r[f"match{k}_hi"] for r in doc_by_group]}
                    for k in ks]},
        {"type": "table", "title": "Verbatim spans by group (document level)",
         "columns": [{"key": "group", "label": "group"}, {"key": "n", "label": "docs", "fmt": "int"}, *span_cols],
         "rows": doc_by_group},
        {"type": "table", "title": "Verbatim spans by input type (document level)",
         "columns": [{"key": "input_type", "label": "input type"}, {"key": "n", "label": "docs", "fmt": "int"}, *span_cols],
         "rows": doc_by_type},
        {"type": "table", "title": "Per field: spans, number and entity precision", "columns": [
            {"key": "field", "label": "field"}, {"key": "n", "label": "n", "fmt": "int"}, *span_cols,
            {"key": "number_precision", "label": "number prec.", "fmt": "pct1"},
            {"key": "numbers", "label": "numbers", "fmt": "int"},
            {"key": "entity_precision", "label": "entity prec.", "fmt": "pct1"},
            {"key": "entities", "label": "entities", "fmt": "int"}], "rows": field_rows},
        {"type": "bar", "title": "Number precision by field", "fmt": "pct1", "hint": "share of numbers found in the source",
         "labels": [r["field"] for r in field_rows], "horizontal": True,
         "series": [{"name": "number precision", "data": [r["number_precision"] for r in field_rows],
                     "lo": [r["number_lo"] for r in field_rows], "hi": [r["number_hi"] for r in field_rows]},
                    {"name": "entity precision", "data": [r["entity_precision"] for r in field_rows],
                     "lo": [r["entity_lo"] for r in field_rows], "hi": [r["entity_hi"] for r in field_rows]}]},
        {"type": "table", "title": "Precision by input type", "columns": [
            {"key": "input_type", "label": "input type"},
            {"key": "number_precision", "label": "number prec.", "fmt": "pct1"},
            {"key": "numbers", "label": "numbers", "fmt": "int"},
            {"key": "entity_precision", "label": "entity prec.", "fmt": "pct1"},
            {"key": "entities", "label": "entities", "fmt": "int"}], "rows": prec_type},
    ], summary=f"{allrow.get(f'match{k12}') or 0:.1%} of documents with source text contain a >= {k12}-token "
               f"verbatim span (max {allrow.get('max')}). {COMPARISON_NOTE}")
    return {"records": n_tot, "with_source": n_src, "fields": fields, "doc_match": {
        k: allrow.get(f"match{k}") for k in ks}, "max_span": allrow.get("max")}
