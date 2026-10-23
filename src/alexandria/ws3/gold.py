"""Gold-paper linking for WS3: which dataset row is the paper a benchmark item is about.

Retrieval arms can only help when the gold paper is in the corpus, and the gold-paper-retrieved rate is only defined
when we know which row it is. Benchmarks name their papers by DOI (LitQA2 `sources`), PMID (PubMedQA; title + DOI
fetched from NCBI on the login node) or corpus document (SciFact: title from its corpus.jsonl). Each item gets
`gold_refs` = [{"doi": ..., "title": ...}, ...]; `link` matches them against an id lookup of every release row:
  1. DOI (source_doi / oa_doi, normalised), else 2. normalised title (>= `min_title_words` words, exact).
Among several matching rows, rows with full text win, then the first (file, row). Unmatched items keep
gold_paper_key = None and are reported as "gold not in corpus".

The lookup (file, row, doi, title_norm, has_text for the whole release: metadata columns only) is built once per
dataset and cached under site.assets.
"""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..data import iter_batches, release_shards
from ..runtime import log
from ..ws1.faith.common import paper_key

_DOI = re.compile(r"10\.\d{4,9}/[^\s'\"\]\[,]+", re.I)
_WORD = re.compile(r"[a-z0-9]+")


def norm_doi(s) -> str | None:
    m = _DOI.search(str(s or ""))
    return m.group(0).lower().rstrip(".;") if m else None


def norm_title(s, min_words: int = 5) -> str | None:
    w = _WORD.findall(unicodedata.normalize("NFKC", str(s or "")).lower())
    return " ".join(w) if len(w) >= min_words else None


def dois_in(refs) -> list[str]:
    """DOIs in a string, or in a list / array of strings (LitQA2 `sources` is either, depending on the file)."""
    text = refs if isinstance(refs, str) else " ".join(map(str, refs)) if refs is not None else ""
    return sorted({d.lower().rstrip(".;") for d in _DOI.findall(text)})


def _shard_ids(cfg, sh, min_words: int) -> list[dict]:
    rows = []
    for tbl in iter_batches(cfg, sh, ["source_doi", "oa_doi", "source_title", "oa_title", "text_length"], 4096):
        for r in tbl.to_pylist():
            base = {"file": r["_file"], "row": int(r["_row"]), "has_text": bool(r.get("text_length") or 0)}
            for d in {norm_doi(r.get("source_doi")), norm_doi(r.get("oa_doi"))} - {None}:
                rows.append({**base, "kind": "doi", "key": d})
            for t in {norm_title(r.get("source_title"), min_words), norm_title(r.get("oa_title"), min_words)} - {None}:
                rows.append({**base, "kind": "title", "key": t})
    return rows


def id_lookup(cfg, min_words: int = 5, workers: int = 16) -> pd.DataFrame:
    """(kind, key) -> rows for the whole release; cached as a parquet asset."""
    path = cfg.site.assets / f"id_lookup_{cfg.dataset.name}_w{min_words}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    shards = release_shards(cfg)
    log(f"gold: building the id lookup over {len(shards)} release shards (once; cached at {path})")
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda s: _shard_ids(cfg, s, min_words), shards))
    df = pd.DataFrame([r for p in parts for r in p], columns=["file", "row", "has_text", "kind", "key"])
    df = df.sort_values(["kind", "key", "has_text", "file", "row"], ascending=[True, True, False, True, True])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return df


def link(items: pd.DataFrame, lookup: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Fill gold_paper_key / gold_file / gold_row / gold_match for items with gold_refs (others untouched)."""
    best = lookup.drop_duplicates(["kind", "key"])            # sorted: full text first
    by = {(r.kind, r.key): (r.file, int(r.row)) for r in best.itertuples()}
    out = items.copy()
    for c in ("gold_file", "gold_row", "gold_match"):
        if c not in out:
            out[c] = None
    for i, refs in out["gold_refs"].items():
        if out.at[i, "gold_paper_key"] or not isinstance(refs, (list, tuple)) or not len(refs):
            continue
        for ref in refs:
            hit, how = None, None
            if norm_doi(ref.get("doi")):
                hit, how = by.get(("doi", norm_doi(ref.get("doi")))), "doi"
            if hit is None and norm_title(ref.get("title"), min_words):
                hit, how = by.get(("title", norm_title(ref.get("title"), min_words))), "title"
            if hit is not None:
                out.at[i, "gold_paper_key"], out.at[i, "gold_file"], out.at[i, "gold_row"] = paper_key(*hit), hit[0], hit[1]
                out.at[i, "gold_match"] = how
                break
    return out
