"""Stage `quality` (reduce): the data-quality flag table (counts per group with Wilson 95% intervals), per-field
status rates, every flagged paper with its links (flagged_papers.csv), a LaTeX table and a figure for the paper.

Input: `features` parts. Optional: `neardup` clusters (near-identical summaries of different papers).
Every flag is one SQL predicate over the features rows (FLAGS below), so the paper's table and the code are the same
list. A summarizer-model split is reported too, although arxiv/pubmed contain a single model.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ..figures import grouped_bars, plt, save, style
from ..papers import paper_table, with_paper_info
from ..runtime import Context, stage
from ..schema import OPTIONAL_FIELDS, PROSE_FIELDS
from ..stats import wilson

CORE = [f for f in PROSE_FIELDS if f not in OPTIONAL_FIELDS]
_CORE_BAD = " or ".join(f"status__{f} in ('empty','placeholder','not_provided')" for f in CORE)

# (key, label, category, severity, SQL predicate). Severity: how much the flag can mislead a downstream user.
FLAGS = [
    ("wrong_paper", "summary does not match the paper (content/title+authors)", "identity", "critical",
     "wrong_paper_suspect"),
    ("garbled_source", "summarized from unreadable source text (PDF font mojibake)", "identity", "critical",
     "source_garbled"),
    ("regen", "needs regeneration (>= 10 of 18 fields blank; KU-Fixes)", "completeness", "critical",
     "ku_needs_regeneration"),
    ("core_missing", "a core prose field is empty or a placeholder", "completeness", "serious", _CORE_BAD),
    ("title_placeholder", "title is a placeholder ('[Unknown]'; KU-Fixes)", "completeness", "serious",
     "ku_title_fault is not null"),
    ("authors_placeholder", "authors are a placeholder (KU-Fixes)", "completeness", "serious",
     "ku_authors_fault is not null"),
    ("no_claims", "no parsable claims", "structure", "serious", "n_claims = 0"),
    ("claims_bad_json", "claims JSON does not parse", "structure", "serious", "claims_bad_json"),
    ("claims_off_schema", "claims use off-schema / misspelled sub-keys", "structure", "minor",
     "claims_off_schema or claims_missing_keys"),
    ("claims_no_support", "a claim has empty supporting evidence", "structure", "minor", "claims_empty_support > 0"),
    ("truncated_field", "a prose field ends mid-sentence", "truncation", "serious", "n_fields_truncated > 0"),
    ("output_capped", "generation hit the output-token cap", "truncation", "serious", "output_capped"),
    ("input_truncated", "source text clipped at the input cap (100k chars)", "truncation", "minor", "input_truncated"),
    ("refusal", "refusal / 'as an AI' boilerplate in a field", "boilerplate", "serious", "refusal"),
    ("markup_leak", "JSON / markdown markup leaked into prose", "boilerplate", "minor", "markup_leak"),
    ("key_leak", "a field name written inside a value", "boilerplate", "minor", "key_leak is not null"),
    ("non_english", "summary not in English", "language", "minor", "summary_non_english"),
    ("len_short", "a core field is anomalously short for its input type (robust z < -3.5)", "length", "minor",
     "len_short"),
    ("len_long", "a core field is anomalously long for its input type (robust z > 3.5)", "length", "minor",
     "len_long"),
    ("dup_strong", "paper duplicate to drop (strong keys; KU-Fixes)", "duplication", "serious", "dup_drop_strong"),
    ("dup_any", "paper duplicate to drop (strong + probable; KU-Fixes)", "duplication", "minor", "dup_drop_any"),
    ("abstract_only", "summarized from metadata + abstract only", "input", "info",
     "input_type in ('metadata_only','abstract_only')"),
    ("no_year", "no publication year (source, OpenAlex or arXiv id)", "metadata", "info", "year is null"),
    ("no_openalex", "no OpenAlex match", "metadata", "info", "not coalesce(has_openalex_match, false)"),
    ("retracted", "retracted (OpenAlex)", "metadata", "info", "coalesce(oa_is_retracted, false)"),
]
SEVERITY_ORDER = {"critical": 0, "serious": 1, "minor": 2, "info": 3}


def flag_table(con, src: str, by: str, flags: list[tuple]) -> pd.DataFrame:
    sel = ", ".join(f"sum(case when ({sql}) then 1 else 0 end) as \"{k}\"" for k, _, _, _, sql in flags)
    df = con.sql(f"select {by} as grp, count(*) as n, {sel} from {src} group by 1 "
                 f"union all select 'all', count(*), {sel} from {src}").df()
    rows = []
    for _, r in df.iterrows():
        for k, label, cat, sev, _ in flags:
            p, lo, hi = wilson(int(r[k]), int(r["n"]))
            rows.append({"group": r["grp"], "flag": k, "label": label, "category": cat, "severity": sev,
                         "k": int(r[k]), "n": int(r["n"]), "rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def length_outliers(df: pd.DataFrame, fields: list[str], z_max: float = 3.5, min_group: int = 30) -> pd.DataFrame:
    """Per paper and field: modified z-score (Iglewicz & Hoaglin, 1993) of log word count against papers of the
    same input type (whole field when that input type has < min_group papers). Only fields with status 'ok'.
    Returns file, row, len_short, len_long, len_fields (which fields, with their z)."""
    z_all = pd.DataFrame(index=df.index)
    for f in fields:
        x = np.log1p(df[f"words__{f}"].where(df[f"status__{f}"] == "ok").astype(float))
        z = pd.Series(np.nan, index=df.index)
        for _, idx in df.groupby(df["input_type"].fillna("NA")).groups.items():
            ref = x.loc[idx] if x.loc[idx].notna().sum() >= min_group else x
            med = ref.median()
            mad = (ref - med).abs().median()
            if mad and np.isfinite(mad):
                z.loc[idx] = 0.6745 * (x.loc[idx] - med) / mad
        z_all[f] = z
    short, long_ = z_all.lt(-z_max), z_all.gt(z_max)
    which = [", ".join(f"{f} (z={z_all.at[i, f]:.1f})" for f in fields if short.at[i, f] or long_.at[i, f])
             for i in df.index]
    return pd.DataFrame({"file": df["file"], "row": df["row"], "len_short": short.any(axis=1),
                         "len_long": long_.any(axis=1), "len_fields": which})


def documented_vs_delivered(df: pd.DataFrame, spec: dict[str, list[int]]) -> list[dict]:
    """README length targets vs the delivered word counts (status 'ok' fields only)."""
    rows = []
    for f, (lo, hi) in spec.items():
        w = df.loc[df[f"status__{f}"] == "ok", f"words__{f}"].dropna().astype(float)
        if w.empty:
            continue
        rows.append({"field": f, "target": f"{lo}–{hi}", "n": len(w), "p10": w.quantile(.1), "median": w.median(),
                     "p90": w.quantile(.9), "max": w.max(), "in_target": float(w.between(lo, hi).mean()),
                     "below": float((w < lo).mean()), "median_vs_target": float(w.median() / lo)})
    return rows


@stage("ws1", "quality", needs=("features",), config_keys=("ws1.checks", "ws1.length", "ws1.length_spec_words",
                                                             "ws1.flagged"),
       help="quality-flag table with 95% CIs per group, field status, flagged papers with links, LaTeX + figure")
def run(ctx: Context) -> dict:
    flags = list(FLAGS)
    con = duckdb.connect()
    parts = str(ctx.input_dir("features") / "parts" / "*.parquet")
    con.sql(f"create view f_raw as select * from read_parquet('{parts}', union_by_name=true)")
    con.sql("create view f as select * from f_raw")
    try:
        nd = ctx.input_dir("neardup") / "clusters.parquet"
        if nd.exists():
            con.sql(f"create view nd as select * from '{nd}'")
            con.sql("create or replace view f as select f.*, (nd.cluster_id is not null and not "
                    "coalesce(nd.category = 'known_duplicate', false)) as near_dup_new from f_raw f left join nd "
                    "on f.file = nd.file and f.row = nd.row")
            flags.append(("near_dup", "near-identical summary of a different paper (MinHash)", "duplication",
                          "minor", "near_dup_new"))
    except FileNotFoundError:
        pass

    # length: outliers against the delivered distribution (per-paper flag) + README targets (dataset-level finding)
    lcfg = ctx.section("ws1.length")
    len_fields = list(lcfg.get("fields") or CORE)
    spec = {k: v for k, v in (ctx.cfg.raw["ws1"].get("length_spec_words") or {}).items() if v}
    lcols = ", ".join(f'"words__{f}", "status__{f}"' for f in dict.fromkeys([*len_fields, *spec]))
    ldf = con.sql(f'select file, "row", input_type, {lcols} from f').df()
    lo_df = length_outliers(ldf, len_fields, float(lcfg.get("robust_z", 3.5)), int(lcfg.get("min_group", 30)))
    con.register("len_out", lo_df)
    con.sql("create or replace table f_base as select * from f")
    con.sql('create or replace view f as select b.*, coalesce(l.len_short, false) as len_short, '
            'coalesce(l.len_long, false) as len_long, l.len_fields from f_base b left join len_out l '
            'on b.file = l.file and b."row" = l."row"')
    spec_rows = documented_vs_delivered(ldf, spec)
    pd.DataFrame(spec_rows).to_csv(ctx.out / "length_vs_documented.csv", index=False)

    by_group = flag_table(con, "f", '"group"', flags)
    by_model = flag_table(con, "f", "coalesce(summary_model, 'unknown')", flags)
    by_group.to_csv(ctx.out / "flags_by_group.csv", index=False)
    by_model.to_csv(ctx.out / "flags_by_model.csv", index=False)

    # per-field status (what share of each field is empty / placeholder / N/A statement / truncated)
    fields = [f for f in PROSE_FIELDS] + ["summary_title", "summary_authors", "field_subfield", "type_of_paper"]
    status_rows = []
    n_all = con.sql("select count(*) from f").fetchone()[0]
    for f in fields:
        trunc = f"sum(trunc__{f}::int)" if f in PROSE_FIELDS else "0"
        r = con.sql(f"select sum((status__{f}='empty')::int), sum((status__{f} in ('placeholder','not_provided'))::int),"
                    f" sum((status__{f}='na_statement')::int), {trunc} from f").fetchone()
        status_rows.append({"field": f, "optional": f in OPTIONAL_FIELDS, "empty": r[0] / n_all,
                            "placeholder": r[1] / n_all, "na_statement": r[2] / n_all, "truncated": (r[3] or 0) / n_all})
    pd.DataFrame(status_rows).to_csv(ctx.out / "field_status.csv", index=False)

    # paper table (LaTeX): overall rate with CI + per-group rates
    allg = by_group[by_group.group == "all"].copy()
    groups = sorted(g for g in by_group.group.unique() if g != "all")
    allg["sev"] = allg.severity.map(SEVERITY_ORDER)
    allg = allg.sort_values(["sev", "rate"], ascending=[True, False])
    tex = pd.DataFrame({"Flag": allg.label, "Severity": allg.severity,
                        "All (95% CI)": [f"{r.rate:.2%} [{r.lo:.2%}, {r.hi:.2%}]" for r in allg.itertuples()]})
    for g in groups:
        gg = by_group[by_group.group == g].set_index("flag")
        tex[g] = [f"{gg.loc[k, 'rate']:.2%}" for k in allg.flag]
    (ctx.out / "flags_table.tex").write_text(tex.to_latex(index=False, escape=True))

    # figure: per-paper defects only (no info flags), dropping flags at 0 in every group
    style()
    import textwrap
    nonzero = set(by_group[(by_group.group != "all") & (by_group.k > 0)].flag)
    show = [k for k in allg[allg.severity != "info"].flag if k in nonzero]
    labels = [textwrap.fill(dict((k, lab) for k, lab, *_ in flags)[k], 34) for k in show]
    series, lo, hi = {}, {}, {}
    for g in groups:
        gg = by_group[by_group.group == g].set_index("flag")
        series[g] = [gg.loc[k, "rate"] for k in show]
        lo[g] = [gg.loc[k, "lo"] for k in show]
        hi[g] = [gg.loc[k, "hi"] for k in show]
    fig, ax = plt.subplots(figsize=(6.8, 0.45 * max(len(show), 1) + 1.2))
    if show:
        grouped_bars(ax, labels, series, horizontal=True, fmt_pct=True, lo=lo, hi=hi)
        ax.legend(loc="lower right")
    ax.set_xlabel("share of papers (95% Wilson interval)")
    fig_paths = save(fig, ctx.out / "figures", "quality_flags")

    # flagged papers: one row per paper with every non-info flag it has (worst severity first), with links
    sel = [(k, lab, sev, sql) for k, lab, _, sev, sql in flags if sev != "info"]
    union = " union all ".join(f"select file, \"row\", '{k}' as flag, {SEVERITY_ORDER[sev]} as sev from f where {sql}"
                               for k, _, sev, sql in sel)
    flagged = con.sql(f"""select file, "row", string_agg(flag, ', ' order by sev, flag) as flags, min(sev) as sev,
                          count(*) as n_flags from ({union}) group by 1, 2 order by sev, n_flags desc, file, "row"
                       """).df()
    flagged = flagged.merge(lo_df[["file", "row", "len_fields"]], on=["file", "row"], how="left")
    flagged["severity"] = flagged["sev"].map({v: k for k, v in SEVERITY_ORDER.items()})
    flagged = with_paper_info(ctx, flagged)
    flagged.drop(columns=["sev"]).to_csv(ctx.out / "flagged_papers.csv", index=False)
    max_rows = int(ctx.section("ws1.flagged").get("max_rows", 500))

    crit = allg[allg.severity.isin(["critical", "serious"])]
    any_crit = con.sql("select count(*) from f where " + " or ".join(
        f"({sql})" for k, _, _, sev, sql in flags if sev == "critical")).fetchone()[0]
    p, plo, phi = wilson(any_crit, n_all)
    fmt_rows = allg.drop(columns=["sev"]).to_dict("records")
    ctx.publish("Data quality flags", [
        {"type": "kpis", "headline": True, "items": [
            {"label": "papers checked", "value": n_all, "fmt": "int"},
            {"label": "with a critical flag", "value": p, "fmt": "pct1", "status": "bad" if p > 0.01 else "warn",
             "hint": f"95% CI {plo:.2%}-{phi:.2%}"},
            *[{"label": r.label[:42], "value": r.rate, "fmt": "pct1",
               "hint": f"{r.k:,} papers; CI {r.lo:.2%}-{r.hi:.2%}"} for r in crit.head(4).itertuples()],
        ]},
        {"type": "table", "title": "All flags (all groups)", "hint": "Wilson 95% intervals; severity = how much the "
         "flag can mislead a downstream user",
         "columns": [{"key": "label", "label": "flag"}, {"key": "category", "label": "category"},
                     {"key": "severity", "label": "severity"}, {"key": "k", "label": "papers", "fmt": "int"},
                     {"key": "rate", "label": "rate", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}], "rows": fmt_rows},
        {"type": "bar", "title": "Flag rates by group", "hint": "critical, serious and minor flags; whiskers = 95% CI",
         "horizontal": True, "fmt": "pct", "labels": labels,
         "series": [{"name": g, "data": series[g], "lo": lo[g], "hi": hi[g]} for g in groups]},
        {"type": "bar", "title": "Field status", "hint": "share of papers per field", "horizontal": True,
         "stacked": True, "fmt": "pct", "labels": [r["field"] for r in status_rows],
         "series": [{"name": s, "data": [r[s] for r in status_rows]}
                    for s in ("empty", "placeholder", "na_statement", "truncated")]},
        paper_table("Flagged papers", flagged, [("severity", "worst", None), ("flags", "flags", None),
                                                ("len_fields", "length detail", None)],
                    hint="Every paper with a critical, serious or minor flag; info flags are not listed. "
                         "Counts per flag are in the table above.", max_rows=max_rows),
        {"type": "table", "title": "Summary length: documented target vs delivered (words)",
         "hint": "targets from the dataset README; a dataset-level finding, not a per-paper defect",
         "columns": [{"key": "field", "label": "field"}, {"key": "target", "label": "README target"},
                     {"key": "p10", "label": "p10", "fmt": "int"}, {"key": "median", "label": "median", "fmt": "int"},
                     {"key": "p90", "label": "p90", "fmt": "int"}, {"key": "max", "label": "max", "fmt": "int"},
                     {"key": "in_target", "label": "within target", "fmt": "pct1"},
                     {"key": "median_vs_target", "label": "median / lower bound", "fmt": "float"}],
         "rows": spec_rows},
        {"type": "image", "title": "Paper figure", "src": fig_paths["png"], "alt": "flag rates by group"},
    ], summary="One row per flag; the same list produces the LaTeX table (flags_table.tex).")
    return {"papers": n_all, "critical_any": any_crit, "flags": len(flags), "flagged_papers": len(flagged),
            "length_outliers": int(lo_df["len_short"].sum() + lo_df["len_long"].sum())}
