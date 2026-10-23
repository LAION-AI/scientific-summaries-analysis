"""Paper figures (matplotlib): one consistent style, PDF for LaTeX + PNG for the dashboard.

Colours: the validated categorical palette (dataviz reference instance), assigned to groups in a FIXED order so a
group keeps its colour in every figure. Thin bars, recessive grid, no dual axes, legend only for >= 2 series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_SLOT = {"arxiv/new": 0, "arxiv/old": 1, "arxiv/titled": 2, "pubmed": 3, "bethgelab": 4, "web-papers": 6}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def color_for(name: str, order: list[str]) -> str:
    if name in GROUP_SLOT:
        return PALETTE[GROUP_SLOT[name]]
    free = [i for i in range(len(PALETTE)) if i not in GROUP_SLOT.values()]
    idx = [n for n in order if n not in GROUP_SLOT].index(name)
    return PALETTE[(free + list(range(len(PALETTE))))[idx % len(PALETTE)]]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": f"figures/{name}.pdf", "png": f"figures/{name}.png"}


def grouped_bars(ax, labels: list[str], series: dict[str, list[float]], horizontal: bool = False,
                 fmt_pct: bool = False, lo: dict | None = None, hi: dict | None = None) -> None:
    """Thin grouped bars with optional 95% whiskers; legend only for >= 2 series."""
    import numpy as np
    names = list(series)
    n = len(names)
    width = 0.8 / max(n, 1)
    x = np.arange(len(labels))
    for i, name in enumerate(names):
        pos = x - 0.4 + width * (i + 0.5)
        vals = np.asarray(series[name], dtype=float)
        err = None
        if lo and hi and name in lo:
            err = np.vstack([vals - np.asarray(lo[name], float), np.asarray(hi[name], float) - vals])
            err = np.clip(np.nan_to_num(err), 0, None)        # float rounding at 0% / 100% can go slightly negative
        kw = dict(color=color_for(name, names), edgecolor=SURFACE, linewidth=0.8, label=name)
        if horizontal:
            ax.barh(pos, vals, height=width * 0.9, xerr=err, error_kw=dict(lw=0.7, ecolor=INK2), **kw)
        else:
            ax.bar(pos, vals, width=width * 0.9, yerr=err, error_kw=dict(lw=0.7, ecolor=INK2), **kw)
    if horizontal:
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(x, labels, rotation=30 if max(map(len, labels or [""])) > 8 else 0, ha="right")
        ax.grid(axis="x", visible=False)
    if fmt_pct:
        from matplotlib.ticker import PercentFormatter
        (ax.xaxis if horizontal else ax.yaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))
    if n >= 2:
        ax.legend(loc="best")
