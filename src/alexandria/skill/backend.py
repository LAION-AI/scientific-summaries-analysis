"""The retrieval backend behind every skill tool (agent harness, MCP server).

    be = Backend(index_dir, encoder_spec, cache_path)
    be.call("search", {"query": "...", "field": "key_results", "k": 5}, view="summary")
    be.call("get_paper", {"id": "a1b2c3d4e5f6", "fields": ["executive_summary"]}, view="summary")
    be.call("search_fulltext", {"query": "...", "k": 5}, view="fulltext")

Views: `summary` (field-aware search over each paper's KU, i.e. its 18-field summary and claims), `fulltext` (chunks
of the source text). A view can be restricted to papers that have full text (S-OA arm). Every call is
answered from the cache when the same (index version, view, tool, arguments) was seen before, and stored otherwise,
so a run can be replayed exactly and the cache can be released with the results.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .retrieval import BM25, Encoder, approx_tokens, rrf, top_k

FIELD_GROUPS = {          # tool field name -> summary units' field values
    "executive_summary": ["executive_summary"],
    "key_results": ["key_results"],
    "methods": ["methodological_details", "procedures_architectures"],
    "limitations": ["contradictions_limitations"],
    "claims": ["claims"],
}
PAPER_FIELDS = ["executive_summary", "research_question_hypothesis", "methodological_details", "key_results",
                "interpretation_implications", "contradictions_limitations", "claims"]


class Backend:
    def __init__(self, index_dir: Path, encoder_spec: dict | None, cache_path: Path | None = None,
                 device: str | None = None, depth: int = 100, hit_chars: int = 1200, max_k: int = 10,
                 dense: bool = True):
        self.dir = Path(index_dir)
        self.units = pd.read_parquet(self.dir / "units.parquet")
        self.version = json.loads((self.dir / "index.json").read_text())["version"]
        self.kind_idx = {k: np.flatnonzero(self.units["kind"].to_numpy() == k) for k in self.units["kind"].unique()}
        self.bm25 = {k: BM25.load(self.dir / f"bm25_{k}") for k in self.kind_idx if (self.dir / f"bm25_{k}.npz").exists()}
        self.emb = {k: np.load(self.dir / f"emb_{k}.npy", mmap_mode="r") for k in self.kind_idx
                    if dense and (self.dir / f"emb_{k}.npy").exists()}
        self.encoder = Encoder(encoder_spec, device) if (dense and encoder_spec and self.emb) else None
        self.depth, self.hit_chars, self.max_k = depth, hit_chars, max_k
        self.papers = self.units.drop_duplicates("paper_key").set_index("paper_key")[["paper_id", "title", "has_text"]]
        self.db = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(cache_path), timeout=60)
            self.db.execute("pragma journal_mode=wal")
            self.db.execute("create table if not exists cache (key text primary key, tool text, args text, "
                            "result text, papers text)")
        self.hits = self.misses = 0

    # ---------------------------------------------------------------- public
    def call(self, tool: str, args: dict, view: str, only_full_text: bool = False) -> tuple[str, list[str]]:
        """-> (text result for the model, paper keys it contains). Cached."""
        key = hashlib.sha256(json.dumps([self.version, view, only_full_text, tool, args], sort_keys=True).encode()).hexdigest()
        if self.db is not None:
            row = self.db.execute("select result, papers from cache where key = ?", (key,)).fetchone()
            if row:
                self.hits += 1
                return row[0], json.loads(row[1])
        self.misses += 1
        try:
            if tool in ("search", "search_fulltext"):
                kind = "fulltext" if tool == "search_fulltext" else view
                text, papers = self._search(str(args.get("query", "")), kind, str(args.get("field", "any")),
                                            int(args.get("k", 5) or 5), only_full_text)
            elif tool == "get_paper":
                text, papers = self._get_paper(str(args.get("id", "")), args.get("fields"), view)
            else:
                text, papers = f"Unknown tool {tool!r}.", []
        except (TypeError, ValueError) as e:
            text, papers = f"Invalid arguments for {tool}: {e}", []
        if self.db is not None:
            self.db.execute("insert or replace into cache values (?, ?, ?, ?, ?)",
                            (key, tool, json.dumps(args, sort_keys=True), text, json.dumps(papers)))
            self.db.commit()
        return text, papers

    def retrieve(self, query: str, kinds: list[str], k: int, only_full_text: bool = False) -> list[int]:
        """Hybrid top-k unit rows over several kinds (plain RAG arm)."""
        fused: dict[int, float] = {}
        for kind in kinds:
            for i, s in self._rank(query, kind, None, only_full_text).items():
                fused[i] = max(fused.get(i, 0.0), s)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]

    def render(self, rows: list[int]) -> str:
        out = []
        for n, i in enumerate(rows, 1):
            u = self.units.iloc[i]
            out.append(f"[{n}] {u['title'] or u['paper_id']} (id: {u['paper_key']}; {u['kind']}"
                       f"{'/' + u['field'] if u['field'] else ''})\n{str(u['text'])[:self.hit_chars]}")
        return "\n\n".join(out)

    # ---------------------------------------------------------------- internals
    def _mask(self, kind: str, field: str, only_full_text: bool) -> np.ndarray:
        idx = self.kind_idx.get(kind, np.array([], np.int64))
        m = np.ones(len(idx), bool)
        if kind == "summary" and field != "any":
            if field not in FIELD_GROUPS:
                raise ValueError(f"field must be one of {['any', *FIELD_GROUPS]}")
            m &= np.isin(self.units["field"].to_numpy()[idx].astype(str), FIELD_GROUPS[field])
        if only_full_text:
            m &= self.units["has_text"].to_numpy()[idx].astype(bool)
        return m

    def _rank(self, query: str, kind: str, field: str | None, only_full_text: bool) -> dict[int, float]:
        if kind not in self.kind_idx:
            return {}
        idx = self.kind_idx[kind]
        mask = self._mask(kind, field or "any", only_full_text)
        lists = []
        if kind in self.bm25:
            lists.append(top_k(self.bm25[kind].scores(query), self.depth, mask))
        if self.encoder is not None and kind in self.emb:
            q = self.encoder.encode([query], max_length=256)[0]
            lists.append(top_k(np.asarray(self.emb[kind] @ q, np.float32), self.depth, mask))
        fused = rrf(lists)
        return {int(idx[i]): s for i, s in fused.items()}

    def _search(self, query: str, kind: str, field: str, k: int, only_full_text: bool) -> tuple[str, list[str]]:
        if not query.strip():
            return "Empty query.", []
        k = max(1, min(k, self.max_k))
        ranked = sorted(self._rank(query, kind, field, only_full_text).items(), key=lambda kv: -kv[1])[:k]
        rows = [i for i, _ in ranked]
        if not rows:
            return "No results.", []
        return self.render(rows), list(dict.fromkeys(self.units.iloc[rows]["paper_key"]))

    def _get_paper(self, pid: str, fields, view: str) -> tuple[str, list[str]]:
        pk = pid.strip()
        if pk not in self.papers.index:                       # also accept the dataset's paper_id
            m = self.papers.index[self.papers["paper_id"].astype(str) == pk]
            if not len(m):
                return f"No paper with id {pid!r}.", []
            pk = m[0]
        u = self.units[self.units["paper_key"] == pk]
        head = f"{self.papers.loc[pk, 'title'] or self.papers.loc[pk, 'paper_id']} (id: {pk})"
        if view == "fulltext":
            body = "\n\n".join(u[u["kind"] == "fulltext"]["text"].head(3))
        else:
            want = [f for f in (fields or ["executive_summary", "key_results"]) if f in PAPER_FIELDS] or ["executive_summary"]
            s = u[(u["kind"] == "summary")]
            body = "\n\n".join(f"## {f}\n" + "\n".join(s[s["field"] == f]["text"]) for f in want
                               if (s["field"] == f).any())
        return f"{head}\n{body[: self.hit_chars * 4] or '(no content in this view)'}", [pk]


def result_tokens(text: str, tok=None) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"]) if tok is not None else approx_tokens(text)
