"""X6 shared pieces: settings, model architecture numbers (parameters, KV-cache bytes per token), FLOPs, cost and
energy, and a bootstrap for ratios of sums.

FLOPs follow the standard dense-transformer estimate 2 * N * tokens (prefill + decode; the attention term is ignored,
which slightly favours long contexts, i.e. understates the savings of short ones). Energy is an estimate:
GPU-seconds x TDP x slice share (MIG slices do not report their own power).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..runtime import Context


def setting(ctx: Context, key: str, default: Any) -> Any:
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"x6_{key}", default)
    return default


def local_snapshot(ctx: Context, spec: dict) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(spec["hf"], revision=spec.get("revision"), cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                             local_files_only=True)


def arch(ctx: Context, spec: dict) -> dict:
    """Parameters (from models.yaml size_b) and KV-cache bytes per token (bf16 KV) from the model's config.json."""
    cfg = json.loads((Path(local_snapshot(ctx, spec)) / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    layers = int(tc["num_hidden_layers"])
    heads = int(tc["num_attention_heads"])
    kv = int(tc.get("num_key_value_heads", heads))
    head_dim = int(tc.get("head_dim") or tc["hidden_size"] // heads)
    return {"params": float(spec.get("size_b", 0)) * 1e9, "layers": layers, "kv_heads": kv, "head_dim": head_dim,
            "kv_bytes_per_token": 2 * layers * kv * head_dim * 2}


def flops(params: float, tokens: float) -> float:
    return 2.0 * params * tokens


def cost(gpu_seconds: float, cfg: dict) -> dict:
    share = float(cfg.get("slice_fraction", 2 / 7))
    usd = gpu_seconds / 3600 * float(cfg.get("usd_per_gpu_hour", 2.0)) * share
    wh = gpu_seconds * float(cfg.get("gpu_tdp_watts", 400)) * share / 3600
    return {"usd": usd, "wh": wh}


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a 95% bootstrap CI over items (e.g. GPU-seconds per correct answer)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if not len(num) or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def iso_tokens(curve: pd.DataFrame, target: float) -> float | None:
    """Smallest token count at which an accuracy-vs-tokens curve (columns tokens, acc; sorted by tokens) reaches
    `target`, by linear interpolation between the measured points; None if it never does."""
    c = curve.sort_values("tokens")
    prev = None
    for r in c.itertuples():
        if r.acc >= target:
            if prev is None or r.acc == prev.acc:
                return float(r.tokens)
            f = (target - prev.acc) / (r.acc - prev.acc)
            return float(prev.tokens + f * (r.tokens - prev.tokens))
        prev = r
    return None


def stage_dir(ctx: Context, ws: str, stage: str) -> Path:
    return ctx.cfg.stage_dir(ws, stage)
