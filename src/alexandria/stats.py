"""Small statistics helpers with intervals, so every reported rate carries uncertainty."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Proportion with a Wilson score 95% interval -> (p, lo, hi). n=0 -> (nan, nan, nan)."""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return (float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def rate_row(k: int, n: int) -> dict:
    p, lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": p, "lo": lo, "hi": hi}


# ---------------------------------------------------------------- paired, cluster bootstrap (WS2 / WS3)
def cluster_bootstrap(df, value: str, cluster: str, by: list[str], n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05):
    """Mean of `value` per group in `by` with a 95% CI from a bootstrap that resamples whole clusters (papers or
    items), the same resample for every group, so differences between groups are paired.
    -> DataFrame: by..., n, clusters, mean, lo, hi.   Also returns the bootstrap matrix {group: means[n_boot]}."""
    import pandas as pd
    clusters = sorted(df[cluster].unique())
    idx = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    counts = np.zeros((n_boot, len(clusters)))
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=len(clusters))
    rows, boot = [], {}
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        ci = g[cluster].map(idx).to_numpy()
        s = np.bincount(ci, weights=g[value].to_numpy(dtype=float), minlength=len(clusters))
        n = np.bincount(ci, minlength=len(clusters)).astype(float)
        bs = (counts @ s) / np.maximum(counts @ n, 1e-12)
        boot[key] = bs
        rows.append({**dict(zip(by, key, strict=True)), "n": len(g), "clusters": int((n > 0).sum()),
                     "mean": float(g[value].mean()), "lo": float(np.quantile(bs, alpha / 2)),
                     "hi": float(np.quantile(bs, 1 - alpha / 2))})
    return pd.DataFrame(rows), boot


def paired_difference(boot: dict, a: tuple, b: tuple, point_a: float, point_b: float, alpha: float = 0.05) -> dict:
    """Difference a - b from paired bootstrap draws: estimate, 95% percentile CI, and a two-sided p-value from the
    bootstrap standard error (Wald, z = diff / SE). A count-based bootstrap p-value cannot go below 1 / n_boot, which
    after Holm over many comparisons would hide clear effects."""
    d = boot[a] - boot[b]
    diff = point_a - point_b
    se = float(np.std(d, ddof=1))
    p = math.erfc(abs(diff) / (se * math.sqrt(2))) if se > 0 else (0.0 if diff != 0 else 1.0)
    return {"diff": diff, "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2)),
            "se": se, "p": float(min(1.0, p))}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
