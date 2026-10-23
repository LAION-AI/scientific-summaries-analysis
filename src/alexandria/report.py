"""The `report` stage of every workstream: the whole WS result as one PDF, built from what its stages published.

Reads each stage's manifest.json (status, timing, SLURM job) and dashboard.json (the same panels the dashboard
shows) and lays them out on A4 pages with matplotlib only (no LaTeX / extra PDF libraries): a title page with the
dataset and the stage status table, then one section per stage with its KPIs, tables (first rows; paper links are
clickable), bar / line charts redrawn from the panel data and paper figures. Re-run it after any stage:
    alexandria <ws> report  ->  <run root>/<ws>/report/<WS>_report_<dataset>_<profile>.pdf
A workstream registers it with `register_report("ws2", "WS2: ...")`; the dashboard links the PDF in its header.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from .runtime import STAGES, Context, stage

A4 = (8.27, 11.69)
MARGIN = 0.06                       # figure fraction, left/right
TOP, BOTTOM = 0.95, 0.05
MAX_TABLE_ROWS = 22               # ordinary tables: first rows only
PAGE_ROWS = 46                    # rows of one table per page
MAX_PAPER_ROWS = 500              # per-paper tables (with links): every row, over several pages


def fmt(v: Any, how: str | None) -> str:
    """Format a value as the dashboard does (fmt keys of PANELS.md)."""
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    try:
        if how == "int":
            return f"{int(round(float(v))):,}"
        if how in ("pct1", "pct"):
            return f"{float(v):.1%}"
        if how == "float3":
            return f"{float(v):.3f}"
        if how == "float":
            return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(map(str, v))
    return str(v)


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def md_plain(s: Any) -> tuple[str, str | None]:
    """Inline markdown -> (plain text, first link URL or None)."""
    s = str(s or "")
    m = _MD_LINK.search(s)
    return _MD_LINK.sub(r"\1", s).replace("**", "").replace("`", ""), (m.group(2) if m else None)


def clip(s: Any, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


class Doc:
    """Top-to-bottom flow layout on A4 matplotlib figures; a new page starts when a block does not fit."""

    def __init__(self, path: Path, title: str, preview_dir: Path | None = None):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        self.plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        self.pdf = PdfPages(path, metadata={"Title": title, "Creator": "alexandria ws1 report"})
        self.title = title
        self.fig = None
        self.y = TOP
        self.pages = 0
        self.preview_dir = preview_dir           # low-resolution PNG of every page (layout checks, no poppler needed)
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- pages
    def new_page(self) -> None:
        self.close_page()
        self.fig = self.plt.figure(figsize=A4)
        self.pages += 1
        self.fig.text(MARGIN, 0.975, self.title, fontsize=7, color="#666", parse_math=False)
        self.fig.text(1 - MARGIN, 0.975, f"page {self.pages}", fontsize=7, color="#666", ha="right")
        self.y = TOP

    def close_page(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            if self.preview_dir:
                self.fig.savefig(self.preview_dir / f"page_{self.pages:03d}.png", dpi=70)
            self.plt.close(self.fig)
            self.fig = None

    def need(self, h: float) -> None:
        if self.fig is None or self.y - h < BOTTOM:
            self.new_page()

    def close(self) -> None:
        self.close_page()
        self.pdf.close()

    # ---------------------------------------------------------------- blocks
    def text(self, s: str, size: float = 9, weight: str = "normal", color: str = "#111", width: int | None = None,
             gap: float = 0.004) -> None:
        # characters per line from the font size (DejaVu Sans averages ~0.55 em per character)
        width = width or int((1 - 2 * MARGIN) * A4[0] * 72 / (size * 0.58))
        lines = []
        for para in str(s).split("\n"):
            lines.extend(textwrap.wrap(para, width) or [""])
        line_h = size / 72 / A4[1] * 1.35
        for ln in lines:
            self.need(line_h)
            self.fig.text(MARGIN, self.y - line_h, ln, fontsize=size, weight=weight, color=color, parse_math=False)
            self.y -= line_h
        self.y -= gap

    def heading(self, s: str, level: int = 1) -> None:
        size = {0: 18, 1: 13, 2: 10.5}[level]
        self.need(0.06 if level < 2 else 0.04)            # keep a heading with what follows it
        self.y -= 0.008
        self.text(s, size=size, weight="bold")

    def table(self, columns: list[dict], rows: list[dict], title: str | None = None,
              max_rows: int = MAX_TABLE_ROWS) -> None:
        """A table; rows beyond one page continue on the next pages (up to max_rows)."""
        if not rows:
            return
        shown = rows[:max_rows]
        first_chunk = True
        for start in range(0, len(shown), PAGE_ROWS):
            self._table_chunk(columns[:8], shown[start:start + PAGE_ROWS], title if first_chunk else None)
            first_chunk = False
        if len(rows) > len(shown):
            self.text(f"({len(rows) - len(shown)} more rows in the stage output / dashboard)", size=7, color="#666")
        self.y -= 0.008

    def _table_chunk(self, cols: list[dict], shown: list[dict], title: str | None) -> None:
        cells, urls = [], {}
        for ri, r in enumerate(shown):
            row = []
            for ci, c in enumerate(cols):
                v = r.get(c["key"])
                if c.get("fmt") == "md":
                    v, url = md_plain(v)
                    if url:
                        urls[(ri + 1, ci)] = url
                row.append(clip(fmt(v, None if c.get("fmt") == "md" else c.get("fmt")), 46 if ci == 0 else 22))
            cells.append(row)
        head = [clip(c.get("label", c["key"]), 22) for c in cols]
        row_h = 0.0165
        h = row_h * (len(shown) + 1)
        self.need(h + (0.022 if title else 0) + 0.012)
        if title:
            self.text(title, size=9, weight="bold")
        ax = self.fig.add_axes((MARGIN, self.y - h, 1 - 2 * MARGIN, h))
        ax.axis("off")
        first = max(0.18, min(0.45, max(len(x[0]) for x in [head, *cells]) / 120))
        widths = [first] + [(1 - first) / max(len(cols) - 1, 1)] * (len(cols) - 1)
        t = ax.table(cellText=cells, colLabels=head, colWidths=widths, loc="upper left", cellLoc="left",
                     bbox=(0, 0, 1, 1))
        t.auto_set_font_size(False)
        t.set_fontsize(6.5)
        for (r, ci), cell in t.get_celld().items():
            cell.get_text().set_parse_math(False)
            if (r, ci) in urls:                              # clickable in the PDF
                cell.get_text().set_url(urls[(r, ci)])
                cell.get_text().set_color("#1d5fb4")
            cell.set_linewidth(0.3)
            cell.set_edgecolor("#bbb")
            if r == 0:
                cell.set_facecolor("#eceae4")
                cell.get_text().set_weight("bold")
        self.y -= h + 0.004

    def chart(self, panel: dict) -> None:
        labels = [clip(x, 40) for x in panel.get("labels") or []]
        series = panel.get("series") or []
        if not labels or not series:
            return
        horizontal = bool(panel.get("horizontal")) or panel.get("type") == "bar" and len(labels) > 8
        h = min(0.42, 0.06 + 0.022 * len(labels)) if horizontal else 0.22
        self.need(h + 0.03)
        self.text(panel.get("title", ""), size=9, weight="bold")
        ax = self.fig.add_axes((MARGIN + (0.25 if horizontal else 0.05), self.y - h,
                                1 - 2 * MARGIN - (0.25 if horizontal else 0.05), h - 0.01))
        import numpy as np
        pct = panel.get("fmt") in ("pct", "pct1")
        x = np.arange(len(labels))
        if panel.get("type") == "line":
            for s in series:
                ax.plot(labels, [np.nan if v is None else v for v in s["data"]], label=s.get("name"))
        else:
            stacked = bool(panel.get("stacked"))
            width = 0.8 if stacked else 0.8 / len(series)
            base = np.zeros(len(labels))
            for i, s in enumerate(series):
                vals = np.array([0 if v is None else v for v in s["data"]], dtype=float)
                pos = x if stacked else x - 0.4 + width * (i + 0.5)
                err = None
                if not stacked and s.get("lo") and s.get("hi"):
                    lo = np.array([np.nan if v is None else v for v in s["lo"]], dtype=float)
                    hi = np.array([np.nan if v is None else v for v in s["hi"]], dtype=float)
                    err = np.nan_to_num(np.vstack([vals - lo, hi - vals]).clip(min=0))
                if horizontal:
                    ax.barh(pos, vals, height=width, left=base if stacked else None, label=s.get("name"), xerr=err,
                            error_kw={"lw": 0.6})
                else:
                    ax.bar(pos, vals, width=width, bottom=base if stacked else None, label=s.get("name"), yerr=err,
                           error_kw={"lw": 0.6})
                if stacked:
                    base += vals
            if horizontal:
                ax.set_yticks(x, labels, fontsize=6.5)
                ax.invert_yaxis()
            else:
                ax.set_xticks(x, labels, fontsize=6.5, rotation=45 if len(labels) > 6 else 0, ha="right"
                              if len(labels) > 6 else "center")
        if pct:
            from matplotlib.ticker import PercentFormatter
            (ax.xaxis if horizontal and panel.get("type") != "line" else ax.yaxis).set_major_formatter(PercentFormatter(1.0))
        ax.tick_params(labelsize=6.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if len(series) > 1:
            ax.legend(fontsize=6.5, frameon=False)
        self.y -= h + (0.05 if not horizontal and len(labels) > 6 else 0.015)

    def image(self, path: Path, title: str = "") -> None:
        if not path.exists():
            return
        img = self.plt.imread(str(path))
        w = 1 - 2 * MARGIN
        h = min(w * img.shape[0] / img.shape[1] * A4[0] / A4[1], 0.5)
        self.need(h + 0.03)
        if title:
            self.text(title, size=9, weight="bold")
        ax = self.fig.add_axes((MARGIN, self.y - h, w, h))
        ax.imshow(img)
        ax.axis("off")
        self.y -= h + 0.012


def render_panel(doc: Doc, p: dict, stage_dir: Path) -> None:
    t = p.get("type")
    if t == "kpis":
        items = p.get("items") or []
        doc.table([{"key": "label", "label": "measure"}, {"key": "value", "label": "value"},
                   {"key": "hint", "label": "note"}],
                  [{"label": it.get("label"), "value": fmt(it.get("value"), it.get("fmt")), "hint": it.get("hint") or ""}
                   for it in items])
    elif t == "table":
        is_paper = any(c.get("key") == "paper" and c.get("fmt") == "md" for c in p.get("columns") or [])
        doc.table(p.get("columns") or [], p.get("rows") or [], p.get("title"),
                  max_rows=MAX_PAPER_ROWS if is_paper else MAX_TABLE_ROWS)
        if p.get("hint"):
            doc.text(p["hint"], size=7, color="#666")
    elif t in ("bar", "line"):
        doc.chart(p)
    elif t == "image":
        src = Path(p.get("src", ""))
        doc.image(src if src.is_absolute() else stage_dir / src, p.get("title", ""))
    elif t == "markdown":
        doc.text(md_plain(p.get("text", ""))[0], size=8.5)



def build_report(ctx: Context, ws: str, title: str, subtitle: str = "") -> dict:
    names = [n for (w, n) in STAGES if w == ws and n != "report"]
    info = []
    for n in names:
        d = ctx.cfg.stage_dir(ws, n)
        m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
        b = json.loads((d / "dashboard.json").read_text()) if (d / "dashboard.json").exists() else {}
        info.append((n, d, m, b))
    ds = ctx.cfg.dataset
    path = ctx.out / f"{ws.upper()}_report_{ds.name}_{ctx.cfg.profile}.pdf"
    for old in ctx.out.glob("*.pdf"):                        # one current PDF per stage folder
        if old != path:
            old.unlink()
    doc = Doc(path, f"Alexandria ICML 2027 · {title} · {ds.name} · {ctx.cfg.profile}", ctx.out / "pages")
    doc.new_page()
    doc.y -= 0.05
    doc.heading(f"Alexandria ICML 2027 — {title}", 0)
    if subtitle:
        doc.text(subtitle, size=11)
    doc.text(f"profile: {ctx.cfg.profile} · generated {dt.datetime.now():%Y-%m-%d %H:%M}", size=8, color="#555")
    doc.y -= 0.01
    doc.heading("Dataset", 2)
    doc.table([{"key": "k", "label": "item"}, {"key": "v", "label": "value"}], [
        {"k": "name", "v": ds.name}, {"k": "title", "v": ds.title or "–"},
        {"k": "groups", "v": ", ".join(s.group for s in ctx.cfg.data.sources)},
        {"k": "data root", "v": str(ctx.cfg.site.data_root)}, {"k": "revision", "v": ctx.cfg.data.revision or "–"},
        {"k": "results", "v": str(ctx.cfg.run_root())}])
    if ds.description:
        doc.text(ds.description, size=8, color="#333")
    doc.table([{"key": "stage", "label": "stage"}, {"key": "status", "label": "status"},
               {"key": "finished", "label": "finished"}, {"key": "seconds", "label": "seconds", "fmt": "int"},
               {"key": "job", "label": "SLURM job"}, {"key": "summary", "label": "summary"}],
              [{"stage": n, "status": m.get("status", "not run"), "finished": (m.get("finished") or "")[:16],
                "seconds": m.get("seconds"), "job": m.get("slurm_job") or "", "summary": b.get("summary", "")}
               for n, _, m, b in info], "Stages")
    for n, d, m, b in info:
        if not b:
            continue
        doc.new_page()
        doc.heading(f"{b.get('title') or n}  ·  {ws}/{n}", 1)
        doc.text(f"status {m.get('status', '?')} · updated {b.get('updated', '?')} · {b.get('summary', '')}",
                 size=7.5, color="#555")
        for p in b.get("panels") or []:
            try:
                render_panel(doc, p, d)
            except Exception as e:                      # noqa: BLE001  (one odd panel must not lose the report)
                doc.text(f"[panel '{p.get('title', p.get('type'))}' could not be drawn: {type(e).__name__}: {e}]",
                         size=7, color="#9f1239")
    pages = doc.pages
    doc.close()
    ctx.publish(f"{title} report (PDF)", [{"type": "markdown", "text":
                f"PDF with every {ws.upper()} stage ({pages} pages): `{path}`. "
                "Also linked in the dashboard header."}], summary=f"{pages} pages")
    return {"pdf": str(path), "pages": pages}


def register_report(ws: str, title: str, subtitle: str = "") -> None:
    """Register `<ws> report`; import this last in the workstream package so it lists after every other stage."""
    @stage(ws, "report", help=f"the whole {ws.upper()} result as one PDF (dataset, status, KPIs, tables, charts)")
    def run(ctx: Context) -> dict:
        return build_report(ctx, ws, title, subtitle)
