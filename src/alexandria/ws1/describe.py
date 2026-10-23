"""Stage `describe` (reduce): what is in the dataset. Distributions of discipline, year, language, open-access
status, input type, summarizer model, field lengths, citations and retractions, per group; a main-text figure and
the numbers the datasheet (`datasheet` stage) cites.

Metadata is reused as released (OpenAlex language / OA / citations / retraction / type); discipline comes from the
summary's own `field_subfield` (the release has no OpenAlex topics), mapped onto SCI-JEPA's 10 domains.
"""
from __future__ import annotations

import duckdb
import numpy as np

from ..domains import DOMAINS
from ..figures import color_for, grouped_bars, plt, save, style
from ..runtime import Context, stage
from ..schema import PROSE_FIELDS

INPUT_TYPES = ["metadata_only", "abstract_only", "short_text", "full_text", "unknown"]


def _share_table(con, col: str, groups: list[str], order: list[str] | None = None, top: int | None = None,
                 where: str = "true") -> tuple[list[str], dict[str, list[float]], dict[str, list[int]]]:
    rows = con.sql(f'select "group", coalesce(cast({col} as varchar), \'(none)\') v, count(*) n from f '
                   f"where {where} group by 1, 2").fetchall()
    totals: dict[str, int] = {}
    counts: dict[tuple[str, str], int] = {}
    overall: dict[str, int] = {}
    for g, v, n in rows:
        totals[g] = totals.get(g, 0) + n
        counts[(g, v)] = n
        overall[v] = overall.get(v, 0) + n
    labels = order or [v for v, _ in sorted(overall.items(), key=lambda kv: -kv[1])]
    if top and len(labels) > top:
        labels = labels[:top]
    labels = [v for v in labels if v in overall]
    share = {g: [counts.get((g, v), 0) / totals[g] if totals.get(g) else 0 for v in labels] for g in groups}
    raw = {g: [counts.get((g, v), 0) for v in labels] for g in groups}
    return labels, share, raw


@stage("ws1", "describe", needs=("features",), config_keys=("ws1.describe",),
       help="dataset description: discipline, year, language, OA, input type, model, lengths, citations")
def run(ctx: Context) -> dict:
    cfgd = ctx.section("ws1.describe")
    con = duckdb.connect()
    parts = str(ctx.input_dir("features") / "parts" / "*.parquet")
    con.sql(f"create view f as select * from read_parquet('{parts}', union_by_name=true)")
    groups = [g for (g,) in con.sql('select distinct "group" from f order by 1').fetchall()]
    n_by_group = dict(con.sql('select "group", count(*) from f group by 1').fetchall())
    n = sum(n_by_group.values())
    stats: dict = {"papers": n, "papers_by_group": n_by_group}

    dom_l, dom_s, dom_n = _share_table(con, "domain", groups, order=DOMAINS)
    in_l, in_s, _ = _share_table(con, "input_type", groups, order=INPUT_TYPES)
    lang_l, lang_s, _ = _share_table(con, "oa_language", groups, top=8)
    slang_l, slang_s, _ = _share_table(con, "summary_lang", groups, top=6)
    oa_l, oa_s, _ = _share_table(con, "oa_is_open_access", groups, order=["true", "false", "(none)"])
    model_l, _, model_n = _share_table(con, "summary_model", groups)
    type_l, type_s, _ = _share_table(con, "lower(split_part(type_of_paper, ' ', 1))", groups, top=8)
    stats.update(domains=dict(zip(dom_l, [sum(dom_n[g][i] for g in groups) for i in range(len(dom_l))])),
                 summary_models={m: sum(model_n[g][i] for g in groups) for i, m in enumerate(model_l)})

    # discipline labels: how free-text is the taxonomy?
    lab = con.sql("select count(distinct field), count(distinct field_clean), sum(field_non_english::int) from f").fetchone()
    top_fields = con.sql(f"select field_clean, count(*) n from f group by 1 order by 2 desc limit {int(cfgd.get('top_fields', 40))}").fetchall()
    stats.update(field_labels_raw=lab[0], field_labels_clean=lab[1], field_labels_non_english_papers=lab[2])

    # year
    ymin = int(cfgd.get("year_min", 1990))
    years = con.sql('select "group", year, count(*) from f where year is not null group by 1, 2 order by 2').fetchall()
    yr = sorted({y for _, y, _ in years if y >= ymin})
    year_series = {g: [0] * len(yr) for g in groups}
    for g, y, c in years:
        if y >= ymin:
            year_series[g][yr.index(y)] = c
    stats["year_missing"] = con.sql("select count(*) from f where year is null").fetchone()[0]
    stats["year_before_min"] = con.sql(f"select count(*) from f where year < {ymin}").fetchone()[0]
    stats["year_source"] = dict(con.sql("select coalesce(year_source,'(none)'), count(*) from f group by 1").fetchall())

    # lengths per field (words): median / IQR / p95
    len_rows = []
    for fld in PROSE_FIELDS:
        q = con.sql(f"select quantile_cont(words__{fld}, [0.25, 0.5, 0.75, 0.95]) from f "
                    f"where status__{fld} = 'ok'").fetchone()[0]
        if q:
            len_rows.append({"field": fld, "p25": q[0], "median": q[1], "p75": q[2], "p95": q[3]})
    tok = con.sql("select quantile_cont(tokens_in, [0.05, 0.5, 0.95]), quantile_cont(tokens_out, [0.05, 0.5, 0.95]),"
                  " sum(tokens_in), sum(tokens_out) from f").fetchone()
    stats.update(tokens_in_q=tok[0], tokens_out_q=tok[1], tokens_in_total=tok[2], tokens_out_total=tok[3],
                 field_words=len_rows)

    # citations, retractions, OpenAlex coverage
    cit = con.sql("select quantile_cont(oa_cited_by_count, [0.5, 0.9, 0.99]), max(oa_cited_by_count),"
                  " sum((oa_cited_by_count = 0)::int), count(oa_cited_by_count) from f").fetchone()
    stats.update(citations_q=cit[0], citations_max=cit[1], citations_zero=cit[2], citations_known=cit[3],
                 retracted=con.sql("select sum(coalesce(oa_is_retracted, false)::int) from f").fetchone()[0],
                 openalex_match=dict(con.sql('select "group", avg(coalesce(has_openalex_match, false)::int) from f group by 1').fetchall()),
                 has_text=dict(con.sql('select "group", avg(has_text::int) from f group by 1').fetchall()),
                 input_truncated=con.sql("select sum(input_truncated::int) from f").fetchone()[0])
    edges = [0, 1, 3, 10, 30, 100, 300, 1000, 10 ** 9]
    cit_labels = ["0", "1-2", "3-9", "10-29", "30-99", "100-299", "300-999", "1000+"]
    cit_series = {}
    for g in groups:
        vals = np.array([v for (v,) in con.sql(f"select oa_cited_by_count from f where \"group\" = '{g}' "
                                               "and oa_cited_by_count is not null").fetchall()], dtype=float)
        h, _ = np.histogram(vals, bins=edges)
        cit_series[g] = (h / max(len(vals), 1)).tolist()
    ctx.write_json("describe.json", stats)

    # ---------------------------------------------------------------- main-text figure (2 x 3)
    style()
    short = {"EESS & Engineering": "EESS / Eng.", "Quantitative Biology & Medicine": "Bio / Med.",
             "Economics & Finance": "Econ. / Fin.", "Computer Science": "Comp. Sci."}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    grouped_bars(axes[0, 0], [short.get(d, d) for d in dom_l], dom_s, horizontal=True, fmt_pct=True)
    axes[0, 0].set_title("discipline (domain)")
    for g in groups:
        axes[0, 1].plot(yr, year_series[g], lw=1.2, color=color_for(g, groups), label=g)
    axes[0, 1].set_title("papers per year")
    grouped_bars(axes[0, 2], [t.replace("_", " ") for t in in_l], in_s, fmt_pct=True)
    axes[0, 2].set_title("input to the summarizer")
    grouped_bars(axes[1, 0], cit_labels, cit_series, fmt_pct=True)
    axes[1, 0].tick_params(axis="x", labelrotation=45)
    axes[1, 0].set_title("citations (OpenAlex-matched)")
    lr = len_rows
    axes[1, 1].barh(range(len(lr)), [r["median"] for r in lr], color="#8a8984",   # one overall series: neutral ink
                    xerr=[[r["median"] - r["p25"] for r in lr], [r["p75"] - r["median"] for r in lr]],
                    error_kw=dict(lw=0.7, ecolor="#52514e"), height=0.6)
    axes[1, 1].set_yticks(range(len(lr)), [r["field"].replace("_", " ")[:24] for r in lr])
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="y", visible=False)
    axes[1, 1].set_title("words per field, all groups (median, IQR)")
    grouped_bars(axes[1, 2], [v if v != "(none)" else "no match" for v in lang_l], lang_s, fmt_pct=True)
    axes[1, 2].set_title("paper language (OpenAlex)")
    for ax in axes.flat:                                   # one shared legend instead of one per panel
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels_ = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(groups), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_paths = save(fig, ctx.out / "figures", "dataset_overview")

    def bar(title, labels, series, hint="", horizontal=False, fmt="pct"):
        return {"type": "bar", "title": title, "hint": hint, "labels": labels, "horizontal": horizontal, "fmt": fmt,
                "series": [{"name": g, "data": series[g]} for g in groups]}

    ctx.publish("Dataset description", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": n, "fmt": "int"},
            *[{"label": g, "value": c, "fmt": "int"} for g, c in sorted(n_by_group.items())],
            {"label": "summarizer models", "value": len(model_l), "fmt": "int", "hint": ", ".join(model_l)},
            {"label": "distinct field labels", "value": lab[0], "fmt": "int",
             "hint": f"{lab[1]:,} after merging spelling variants"},
            {"label": "retracted (OpenAlex)", "value": stats["retracted"], "fmt": "int"},
        ]},
        bar("Discipline (canonical domain)", dom_l, dom_s, "summary field_subfield -> SCI-JEPA domains", True),
        {"type": "line", "title": "Papers per year", "hint": f"source_year > oa_year > arXiv id; "
         f"{stats['year_missing']:,} papers without a year", "labels": yr, "fmt": "int",
         "series": [{"name": g, "data": year_series[g]} for g in groups]},
        bar("Input to the summarizer", in_l, in_s, "from summary_tokens_in minus the prompt overhead"),
        bar("Paper language (OpenAlex, reused)", lang_l, lang_s),
        bar("Summary language (fastText lid.176)", slang_l, slang_s),
        bar("Open access (OpenAlex, reused)", oa_l, oa_s),
        bar("Paper type (summary)", type_l, type_s, horizontal=True),
        bar("Citations (OpenAlex-matched papers)", cit_labels, cit_series),
        {"type": "table", "title": "Words per field", "hint": "non-empty values only",
         "columns": [{"key": "field", "label": "field"}, *[{"key": k, "label": k, "fmt": "int"}
                                                           for k in ("p25", "median", "p75", "p95")]], "rows": len_rows},
        {"type": "table", "title": "Most common field labels (spelling variants merged)",
         "columns": [{"key": "field", "label": "field"}, {"key": "n", "label": "papers", "fmt": "int"}],
         "rows": [{"field": a, "n": b} for a, b in top_fields]},
        {"type": "image", "title": "Main-text figure", "src": fig_paths["png"], "alt": "dataset overview"},
    ], summary="Distributions per group; numbers for the datasheet are in describe.json.")
    return {"papers": n, "groups": n_by_group}
