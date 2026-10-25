"""X1 shared pieces: gold-paper metrics, rank fusion, unit -> paper aggregation, BM25 / FAISS helpers, encoder."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import Context

REPRESENTATIONS = ("fields", "whole", "claims", "chunks")
RETRIEVERS = ("bm25", "dense", "dense_pq48", "rrf")


def setting(ctx: Context, key: str, default: Any) -> Any:
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"x1_{key}", default)
    return default


def stage_table(ctx: Context, stage: str, name: str) -> pd.DataFrame:
    p = ctx.cfg.stage_dir("x1", stage) / name
    if not p.exists():
        raise FileNotFoundError(f"x1/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


# ---------------------------------------------------------------- metrics (pure, unit-tested)
def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    """Share of the gold papers found in the top k (LitSearch's recall@k; one gold -> hit@k)."""
    return len(set(ranked[:k]) & gold) / len(gold) if gold else float("nan")


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, p in enumerate(ranked):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at(ranked: list[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else float("nan")


def first_gold_rank(ranked: list[str], gold: set[str]) -> int | None:
    """1-based rank of the first gold paper, None if not retrieved."""
    for i, p in enumerate(ranked):
        if p in gold:
            return i + 1
    return None


def to_papers(unit_ids: np.ndarray, scores: np.ndarray, unit_paper: np.ndarray, depth: int) -> tuple[list[str], list[float]]:
    """Unit hits -> paper ranking: a paper's score is its best unit's score (max); ties keep unit order."""
    best: dict[str, float] = {}
    for u, s in zip(unit_ids, scores, strict=True):
        if u < 0:
            continue
        p = unit_paper[u]
        if p not in best:
            best[p] = float(s)
    order = sorted(best.items(), key=lambda kv: -kv[1])[:depth]
    return [p for p, _ in order], [s for _, s in order]


def rrf(rankings: list[list[str]], k: int = 60, depth: int = 100) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009): fuses ranks, not scores."""
    score: dict[str, float] = {}
    for r in rankings:
        for i, p in enumerate(r):
            score[p] = score.get(p, 0.0) + 1.0 / (k + i + 1)
    return [p for p, _ in sorted(score.items(), key=lambda kv: -kv[1])][:depth]


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- BM25 (bm25s, as in the release) and FAISS
def bm25_tokenize(texts: list[str]):
    import bm25s
    import Stemmer
    return bm25s.tokenize(texts, stemmer=Stemmer.Stemmer("english"), show_progress=False)


def pq_params(n: int, cfg: dict) -> tuple[int, int]:
    """(nlist, nprobe) scaled from the release: same vectors per inverted list and the same probed share."""
    nlist = max(1, int(round(n / int(cfg.get("vectors_per_list", 5300)))))
    nprobe = max(1, int(round(nlist * float(cfg.get("nprobe_share", 32 / 4096)))))
    return nlist, nprobe


def build_pq(vecs: np.ndarray, nlist: int, m: int = 48, bits: int = 8):
    """IVF-PQ index with inner product, as the release (IVF{nlist},PQ48)."""
    import faiss
    d = vecs.shape[1]
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, m, bits, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    return index


# ---------------------------------------------------------------- encoder
def local_snapshot(ctx: Context, spec: dict) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(spec["hf"], revision=spec.get("revision"), cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                             local_files_only=True)


def load_encoder(ctx: Context, name: str, device: str = "cuda", batch_size: int = 32):
    """Dense encoder from the local snapshot in fp32 (exact vectors; the release's PQ48 codes are the lossy part)."""
    from ..llm import embedder_spec
    from ..skill.retrieval import Encoder
    spec = embedder_spec(ctx.cfg, name)
    spec = {**spec, "hf": local_snapshot(ctx, spec), "revision": None, "batch_size": batch_size}
    enc = Encoder(spec, "cpu")                          # fp32 weights
    enc.model.to(device)
    enc.device = device
    return enc


def out_dir(ctx: Context, stage: str) -> Path:
    return ctx.cfg.stage_dir("x1", stage)
