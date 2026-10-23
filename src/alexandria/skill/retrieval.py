"""Retrieval primitives: BM25 (scipy sparse), a dense encoder (transformers), reciprocal rank fusion.

BM25: Robertson & Zaragoza (2009) with Lucene's idf, k1 = 0.9, b = 0.4 (Pyserini defaults); the document-term
matrix stores the per-term BM25 weight, so a query is one sparse column sum. Dense: CLS pooling, L2-normalised
(BAAI/bge-m3 dense mode). Hybrid: reciprocal rank fusion (Cormack et al., 2009), k = 60, over the top `depth` of each.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how in into is it
its may might of on or our that the their them then there these they this those to was we were what when where which
while who why will with would you your than such also between both each more most other some""".split())


def tokenize(text: str | None) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, vocab: dict[str, int], weights, k1: float = 0.9, b: float = 0.4):
        self.vocab, self.W, self.k1, self.b = vocab, weights, k1, b        # W: csc matrix docs x terms

    @classmethod
    def build(cls, texts: list[str], k1: float = 0.9, b: float = 0.4) -> BM25:
        from scipy import sparse
        vocab: dict[str, int] = {}
        rows, cols, tfs, lens = [], [], [], []
        for i, t in enumerate(texts):
            c = Counter(tokenize(t))
            lens.append(sum(c.values()))
            for term, n in c.items():
                rows.append(i)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(n)
        n_docs = len(texts)
        tf = sparse.csr_matrix((np.asarray(tfs, np.float32), (rows, cols)), shape=(n_docs, len(vocab)))
        df = np.bincount(np.asarray(cols, np.int64), minlength=len(vocab)).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(lens, np.float32)
        norm = k1 * (1 - b + b * dl / max(float(dl.mean()) if n_docs else 1.0, 1e-9))
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        W = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=(n_docs, len(vocab)))
        return cls(vocab, W, k1, b)

    def scores(self, query: str) -> np.ndarray:
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.W.shape[0], np.float32)
        return np.asarray(self.W[:, ids].sum(axis=1)).ravel()

    def save(self, path: Path) -> None:
        from scipy import sparse
        sparse.save_npz(path.with_suffix(".npz"), self.W)
        path.with_suffix(".vocab.json").write_text(json.dumps({"vocab": self.vocab, "k1": self.k1, "b": self.b}))

    @classmethod
    def load(cls, path: Path) -> BM25:
        from scipy import sparse
        meta = json.loads(path.with_suffix(".vocab.json").read_text())
        return cls(meta["vocab"], sparse.load_npz(path.with_suffix(".npz")).tocsc(), meta["k1"], meta["b"])


class Encoder:
    """Dense encoder: transformers AutoModel, CLS or mean pooling, L2-normalised, fp16 on GPU."""

    def __init__(self, spec: dict, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision"))
        self.model = AutoModel.from_pretrained(spec["hf"], revision=spec.get("revision"),
                                               torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device).eval()
        self.pooling = spec.get("pooling", "cls")
        self.max_length = int(spec.get("max_length", 512))
        self.batch_size = int(spec.get("batch_size", 64))

    def encode(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        out = []
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))          # length-sorted batches
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                enc = self.tok([texts[i] or " " for i in idx], padding=True, truncation=True,
                               max_length=max_length or self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                if self.pooling == "mean":
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    v = h[:, 0]
                v = torch.nn.functional.normalize(v.float(), dim=-1)
                out.append((idx, v.cpu().numpy()))
        res = np.zeros((len(texts), out[0][1].shape[1] if out else 0), np.float32)
        for idx, v in out:
            res[idx] = v
        return res


def rrf(rankings: list[np.ndarray], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion of several ranked id lists -> {id: fused score}."""
    fused: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (k + rank + 1)
    return fused


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()) if mask is not None else len(s))
    if k <= 0:
        return np.array([], np.int64)
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return idx[np.isfinite(s[idx]) & (s[idx] > (-np.inf))]


def approx_tokens(text: str) -> int:
    """Token estimate for budgets when no tokenizer is at hand (~4 characters per token for English)."""
    return math.ceil(len(text or "") / 4)
