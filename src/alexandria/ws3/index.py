"""Stage `index` (WS3, GPU for the dense encoder): the retrieval index behind the Alexandria skill.

Corpus: every paper of the profile (debug: the WS1 sample; full: the release, or `ws3.index.max_papers`) plus the
linked gold papers of the benchmark items (ws3.gold; in the full profile they are already in it), minus a blocklist
(papers that define a benchmark, e.g. HLE arXiv:2501.14249, GPQA arXiv:2311.12022; leakage control). In the debug
profile the corpus is small (sample + gold papers), so retrieval there is easier than over the full release.
Units (rows of units.parquet), one index per kind:
  summary   the paper's KU (its 18-field summary): one unit per WS1-cleaned field, one per claim (description + evidence)
  fulltext  source text in `chunk_words`-word chunks with `chunk_overlap` overlap (papers with text only)
Per kind: BM25 (bm25_<kind>.npz + vocab) and dense embeddings (emb_<kind>.npy, float16, L2-normalised).
index.json carries a version hash: the tool-response cache is keyed on it.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..data import fetch_release_records, input_shards, iter_batches
from ..llm import embedder_spec
from ..runtime import Context, log, stage
from ..schema import SUMMARY_FIELDS, load_summary
from ..skill.retrieval import BM25, Encoder
from ..ws1.faith.common import paper_key, source_text

SUMMARY_UNIT_FIELDS = ["executive_summary", "research_question_hypothesis", "methodological_details",
                       "procedures_architectures", "key_results", "interpretation_implications",
                       "contradictions_limitations"]


def chunk_words(text: str, size: int, overlap: int, max_words: int) -> list[str]:
    w = (text or "").split()[:max_words]
    step = max(size - overlap, 1)
    return [" ".join(w[i:i + size]) for i in range(0, len(w), step) if len(w[i:i + size]) >= 20]


def paper_units(rec: dict, cfg: dict) -> list[dict]:
    s = load_summary(rec)["summary"]
    out = []
    for f in SUMMARY_UNIT_FIELDS:
        if s.get(f):
            out.append({"kind": "summary", "field": f, "text": str(s[f]).strip()})
    for c in s.get("claims") or []:
        t = (c.get("description") or "").strip()
        if c.get("supporting_evidence"):
            t += f" Evidence: {str(c['supporting_evidence']).strip()}"
        if t:
            out.append({"kind": "summary", "field": "claims", "text": t})
    src = source_text(rec)
    for ch in chunk_words(src, int(cfg.get("chunk_words", 300)), int(cfg.get("chunk_overlap", 50)),
                          int(cfg.get("max_words_per_paper", 20000))):
        out.append({"kind": "fulltext", "field": "", "text": ch})
    return out


@stage("ws3", "index", needs=("benchmarks",), resources="gpu", config_keys=("ws3.index",),
       help="retrieval index: KU (summary) fields + claims, full-text chunks; BM25 + bge-m3 per kind")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws3.index")
    block = {str(x) for x in cfg.get("blocklist_arxiv", [])}
    cols = ["paper_id", "source_title", "oa_title", "oa_doi", "source_doi", "text_raw", "text_sanitized",
            *SUMMARY_FIELDS]
    rows, seen, n_blocked = [], set(), 0
    max_papers = cfg.get("max_papers")

    def add(rec: dict, gold: bool) -> None:
        nonlocal n_blocked
        pk = paper_key(rec["_file"], rec["_row"])
        if pk in seen:
            return
        ids = " ".join(str(rec.get(k) or "") for k in ("paper_id", "oa_doi", "source_doi")).lower()
        if any(b in ids for b in block):
            n_blocked += 1
            return
        seen.add(pk)
        title = " ".join(str(rec.get("oa_title") or rec.get("summary_title") or rec.get("source_title")
                             or "").split())[:200]
        has_text = bool(source_text(rec).strip())
        for u in paper_units(rec, cfg):
            rows.append({**u, "paper_key": pk, "file": rec["_file"], "row": rec["_row"], "paper_id": rec.get("paper_id"),
                         "title": title, "has_text": has_text, "gold": gold})

    for sh in input_shards(ctx.cfg):
        for tbl in iter_batches(ctx.cfg, sh, cols, batch_size=512):
            for rec in tbl.to_pylist():
                add(rec, False)
            if max_papers and len(seen) >= int(max_papers):
                break
    n_corpus = len(seen)
    items = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
    g = items.dropna(subset=["gold_file", "gold_row"])
    gold_keys = sorted({(f, int(r)) for f, r in zip(g["gold_file"], g["gold_row"], strict=True)})
    missing = [k for k in gold_keys if paper_key(*k) not in seen]
    log(f"index: {n_corpus} corpus papers; {len(gold_keys)} gold papers, {len(missing)} added from the release")
    for rec in fetch_release_records(ctx.cfg, missing, cols).values():
        add(rec, True)
    n_papers = len(seen)
    units = pd.DataFrame(rows)
    units.insert(0, "uid", range(len(units)))
    units.to_parquet(ctx.out / "units.parquet", index=False)
    enc = Encoder(embedder_spec(ctx.cfg, cfg.get("embedder", "bge-m3")))
    sizes = {}
    for kind, g in units.groupby("kind", sort=False):
        texts = g["text"].tolist()
        log(f"index: {kind}: {len(texts)} units: BM25")
        BM25.build(texts, float(cfg.get("k1", 0.9)), float(cfg.get("b", 0.4))).save(ctx.out / f"bm25_{kind}")
        log(f"index: {kind}: dense embeddings")
        np.save(ctx.out / f"emb_{kind}.npy", enc.encode(texts, int(cfg.get("max_length", 512))).astype(np.float16))
        sizes[kind] = len(texts)
    version = hashlib.sha256(json.dumps([sizes, ctx.cfg.fingerprint("ws3.index"), n_papers]).encode()).hexdigest()[:12]
    ctx.write_json("index.json", {"version": version, "papers": n_papers, "gold_added": n_papers - n_corpus,
                                  "blocked": n_blocked, "units": sizes,
                                  "embedder": cfg.get("embedder", "bge-m3")})
    ctx.publish("WS3 retrieval index", [
        {"type": "kpis", "items": [
            {"label": "papers indexed", "value": n_papers, "fmt": "int"},
            {"label": "gold papers added", "value": n_papers - n_corpus, "fmt": "int",
             "hint": "benchmark gold papers outside the profile's corpus (linked by DOI / title)"},
            {"label": "blocked (benchmark papers)", "value": n_blocked, "fmt": "int"},
            *[{"label": f"{k} units", "value": v, "fmt": "int"} for k, v in sizes.items()],
            {"label": "index version", "value": version}]}],
        summary=f"{n_papers} papers, {sum(sizes.values())} units")
    return {"papers": n_papers, "units": sizes, "version": version}
