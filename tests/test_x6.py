"""X6 (token & compute efficiency): the CPU helpers: FLOPs / cost, ratio bootstrap, iso-accuracy interpolation,
context packing to a token budget, training checkpoint steps, per-text storage cost, iso tables."""
import numpy as np
import pandas as pd

from alexandria.x6.common import cost, flops, iso_tokens, ratio_ci
from alexandria.x6.rag import pack
from alexandria.x6.results import iso_inference
from alexandria.x6.storage import text_cost
from alexandria.x6.train import budget_steps


class WordTok:
    """Whitespace tokenizer with the HF call / decode interface the helpers use."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids):
        return " ".join(ids)


def test_flops_and_cost():
    assert flops(7e9, 1000) == 1.4e13
    c = cost(3600, {"usd_per_gpu_hour": 2.0, "slice_fraction": 0.25, "gpu_tdp_watts": 400})
    assert abs(c["usd"] - 0.5) < 1e-9 and abs(c["wh"] - 100) < 1e-9


def test_ratio_ci():
    est, lo, hi = ratio_ci(np.ones(100), np.r_[np.ones(50), np.zeros(50)], n_boot=500)
    assert abs(est - 2.0) < 1e-9 and lo < 2.0 < hi
    assert np.isnan(ratio_ci(np.ones(3), np.zeros(3))[0])


def test_iso_tokens():
    curve = pd.DataFrame({"tokens": [500, 1000, 2000], "acc": [0.2, 0.4, 0.6]})
    assert iso_tokens(curve, 0.5) == 1500
    assert iso_tokens(curve, 0.1) == 500
    assert iso_tokens(curve, 0.9) is None


def test_pack_respects_budget():
    units = [("[1] A", " ".join(["a"] * 40), "p1"), ("[2] B", " ".join(["b"] * 40), "p2"),
             ("[3] C", " ".join(["c"] * 40), "p3")]
    text, n, papers = pack(WordTok(), units, 100)
    assert n <= 100 and papers == ["p1", "p2"]          # third unit: 100 - 84 = 16 < 32 tokens left, dropped
    text, n, papers = pack(WordTok(), units, 120)
    assert n == 120 and papers == ["p1", "p2", "p3"]    # third unit cut to the remaining 36 tokens
    assert pack(WordTok(), [], 100) == ("", 0, [])


def test_budget_steps():
    s = budget_steps(167970, [0.25, 0.5, 1.0, 2.0], 8192)
    assert s == {0.25: 6, 0.5: 11, 1.0: 21, 2.0: 42}


def test_text_cost():
    c = text_cost(WordTok(), "one two three")
    assert c["tokens"] == 3 and c["bytes"] == 13 and c["gzip_bytes"] > 0
    assert text_cost(WordTok(), "") == {"bytes": 0, "gzip_bytes": 0, "tokens": 0}


def test_iso_inference():
    tbl = pd.DataFrame([
        {"model": "m", "arm": "ku_clean", "accuracy": 0.5, "context_tokens": 1000, "prompt_tokens": 1100,
         "gpu_s_per_1k_correct": 10.0},
        {"model": "m", "arm": "full_text", "accuracy": 0.6, "context_tokens": 8000, "prompt_tokens": 8100,
         "gpu_s_per_1k_correct": 50.0},
        {"model": "m", "arm": "ft@1000", "accuracy": 0.3, "context_tokens": 1000, "prompt_tokens": 1100,
         "gpu_s_per_1k_correct": 12.0},
        {"model": "m", "arm": "ft@4000", "accuracy": 0.5, "context_tokens": 4000, "prompt_tokens": 4100,
         "gpu_s_per_1k_correct": 30.0}])
    rng = np.random.default_rng(0)
    boot = {("m", "ku_clean"): 0.5 + rng.normal(0, .01, 200), ("m", "full_text"): 0.6 + rng.normal(0, .01, 200)}
    iso, share = iso_inference(tbl, boot, [0.8, 1.0])
    r = iso[0]
    assert r["ft_tokens_to_match_ku"] == 4000 and r["iso_ratio"] == 4.0
    assert abs(r["diff"] + 0.1) < 1e-9 and abs(r["gpu_ratio"] - 5.0) < 1e-9
    assert share[0]["ku_clean_reaches"] and not share[1]["ku_clean_reaches"]
