"""Per-paper tables for the dashboard and the PDF: counts and links instead of text excerpts.

    tbl = with_paper_info(ctx, df)          # df has file, row (+ any columns) -> adds paper_id, title, group, links
    panel = paper_table("Flagged papers", tbl, [("flags", "flags", None), ("n", "sentences", "int")])

The `paper` column shows the title linked to the paper's first public page; `links` lists every page the dataset's
link rules found (arXiv, PMC, DOI, OpenAlex, ...). Without a link the dataset location (file:row) is shown, so every
row can still be traced back to its record.
"""
from __future__ import annotations

import pandas as pd

from .links import first_url, markdown_links
from .runtime import Context

INFO_COLUMNS = ["file", "row", "paper_id", "title", "group", "links"]


def features_info(ctx: Context, ws: str = "ws1") -> pd.DataFrame:
    """paper_id, title, group and links of every paper, from the `features` stage."""
    import duckdb
    try:
        parts = ctx.parts("features", ws)
    except FileNotFoundError:                   # `features` not run yet: tables still show the dataset location
        parts = []
    if not parts:
        return pd.DataFrame(columns=INFO_COLUMNS)
    files = [str(p) for p in parts]
    have = set(duckdb.sql(f"select * from read_parquet({files!r}, union_by_name=true) limit 0").columns)
    cols = ", ".join(f'"{c}"' if c in have else f'null as "{c}"' for c in INFO_COLUMNS)
    return duckdb.sql(f"select {cols} from read_parquet({files!r}, union_by_name=true)").df()


def with_paper_info(ctx: Context, df: pd.DataFrame, ws: str = "ws1") -> pd.DataFrame:
    info = features_info(ctx, ws)
    drop = [c for c in ("paper_id", "title", "group", "links") if c in df.columns]
    return df.drop(columns=drop).merge(info, on=["file", "row"], how="left")


def paper_cell(r) -> str:
    label = (r.get("title") or r.get("paper_id") or f"{r.get('file')}:{r.get('row')}")
    label = str(label).replace("[", "(").replace("]", ")")[:110]
    url = first_url(r.get("links"))
    return f"[{label}]({url})" if url else label


def paper_table(title: str, df: pd.DataFrame, extra: list[tuple[str, str, str | None]], hint: str | None = None,
                limit: int = 50, max_rows: int = 500) -> dict:
    """Table panel: paper (linked title), group, the extra columns, all links, dataset location."""
    rows = []
    for r in df.head(max_rows).to_dict("records"):
        rows.append({"paper": paper_cell(r), "group": r.get("group"),
                     **{k: r.get(k) for k, _, _ in extra},
                     "links": markdown_links(r.get("links")) or "–", "where": f"{r.get('file')}:{r.get('row')}"})
    cols = [{"key": "paper", "label": "paper", "fmt": "md"}, {"key": "group", "label": "group"}] + \
        [{"key": k, "label": lab, **({"fmt": f} if f else {})} for k, lab, f in extra] + \
        [{"key": "links", "label": "links", "fmt": "md"}, {"key": "where", "label": "dataset file:row"}]
    more = f" Showing {len(rows):,} of {len(df):,}; the full list is in the stage's CSV." if len(df) > len(rows) else ""
    return {"type": "table", "title": f"{title} ({len(df):,})", "hint": ((hint or "") + more).strip() or None,
            "columns": cols, "rows": rows, "limit": limit, "wide": True}
