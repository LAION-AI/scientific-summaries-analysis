"""Stage `corpus` (X1): the search corpus and its units, one table per representation the release indexes.

Papers = every linked gold paper of the evaluated queries (+ the WS3 QA-bench gold papers, for the WS3 diagnosis)
+ `distractors` random release papers from `distractor_shards` random local shards (deterministic). Units:
  fields  one unit per KU field (WS1-cleaned: title + the 13 prose fields + claims), as the release's 19 per-field
          indices (summary_authors, field_subfield, type_of_paper left out: no topical content)
  whole   the whole KU concatenated (the release's `concatenated_summary` index)
  claims  one unit per claim (description + supporting evidence; the release's claims index)
  chunks  500-word segments of the paper's full text (the release's chunk index is built from the text, not the KU)
Rows are read per parquet row group (only the groups holding the wanted rows), which keeps gold papers spread over
hundreds of shards cheap.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data import release_shards
from ..runtime import Context, log, stage
from ..schema import PROSE_FIELDS, SUMMARY_FIELDS, load_summary
from ..ws1.faith.common import paper_key, source_text
from .common import setting, stage_table

COLUMNS = ["paper_id", *SUMMARY_FIELDS, "text_raw", "text_sanitized"]


def read_rows(cfg, keys: list[tuple[str, int]], columns: list[str]) -> dict[tuple[str, int], dict]:
    """(release file, row) -> record, reading only the row groups that contain the wanted rows."""
    paths = {s.rel: s.path for s in release_shards(cfg)}
    by: dict[str, set[int]] = {}
    for f, r in keys:
        by.setdefault(f, set()).add(int(r))
    out = {}
    for f, rows in by.items():
        pf = pq.ParquetFile(paths[f])
        cols = [c for c in columns if c in pf.schema_arrow.names]
        starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
        for g in sorted({int(np.searchsorted(starts, r, side="right") - 1) for r in rows}):
            t = pf.read_row_group(g, columns=cols).to_pylist()
            for r in rows:
                if starts[g] <= r < starts[g + 1]:
                    out[(f, r)] = t[r - starts[g]]
    return out


def distractor_keys(cfg, n: int, n_shards: int, seed: int, exclude: set[tuple[str, int]]) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    shards = [s for s in release_shards(cfg)]
    pick = rng.sample(shards, min(n_shards, len(shards)))
    per = -(-n // max(len(pick), 1))
    keys = []
    for s in pick:
        m = pq.ParquetFile(s.path).metadata.num_rows
        keys += [(s.rel, r) for r in rng.sample(range(m), min(per, m)) if (s.rel, r) not in exclude]
    return keys[:n]


def units_of(rec: dict, chunk_words: int, max_chunks: int) -> dict[str, list[tuple[str, str]]]:
    s = load_summary(rec)["summary"]
    fields = []
    if s.get("summary_title"):
        fields.append(("summary_title", str(s["summary_title"])))
    fields += [(f, str(s[f])) for f in PROSE_FIELDS if s.get(f)]
    claims = [(c.get("description") or "") + (" " + c["supporting_evidence"] if c.get("supporting_evidence") else "")
              for c in s.get("claims") or [] if c.get("description")]
    if claims:
        fields.append(("claims", " ".join(claims)))
    whole = "\n".join(t for _, t in fields)
    words = source_text(rec).split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)][:max_chunks]
    return {"fields": fields, "whole": [("whole", whole)] if whole else [],
            "claims": [("claim", c) for c in claims], "chunks": [("chunk", c) for c in chunks if len(c.split()) >= 50]}


@stage("x1", "corpus", needs=("benchmarks",), config_keys=("x1.corpus",),
       help="search corpus: linked gold papers + random release distractors; units per representation")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x1.corpus")
    q = stage_table(ctx, "benchmarks", "queries.parquet")
    gold = {(f, int(r)) for fs, rs in zip(q["gold_files"], q["gold_rows"], strict=True) for f, r in zip(fs, rs, strict=True)}
    ws3 = ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet"
    n_ws3 = 0
    if ws3.exists():
        it = pd.read_parquet(ws3)
        it = it[(it["bench"] == "qabench-open") & it["gold_file"].notna()]
        add = {(f, int(r)) for f, r in zip(it["gold_file"], it["gold_row"], strict=True)}
        n_ws3 = len(add - gold)
        gold |= add
    n_d = int(setting(ctx, "distractors", cfg.get("distractors", 200000)))
    dis = distractor_keys(ctx.cfg, n_d, int(setting(ctx, "distractor_shards", cfg.get("distractor_shards", 400))),
                          int(cfg.get("seed", 0)), gold)
    keys = sorted(gold) + dis
    log(f"corpus: {len(gold)} gold papers + {len(dis)} distractors")
    recs = read_rows(ctx.cfg, keys, COLUMNS)
    rows, papers = [], []
    for f, r in keys:
        rec = recs.get((f, r))
        if rec is None:
            continue
        pk = paper_key(f, r)
        u = units_of(rec, int(cfg.get("chunk_words", 500)), int(cfg.get("max_chunks_per_paper", 40)))
        papers.append({"paper_key": pk, "file": f, "row": r, "paper_id": rec.get("paper_id"), "gold": (f, r) in gold,
                       "has_text": bool(u["chunks"]), **{f"n_{k}": len(v) for k, v in u.items()}})
        for rep, us in u.items():
            rows += [{"rep": rep, "paper_key": pk, "field": fld, "text": t} for fld, t in us]
    units = pd.DataFrame(rows)
    pp = pd.DataFrame(papers)
    pp.to_parquet(ctx.out / "papers.parquet", index=False)
    for rep, g in units.groupby("rep"):
        g.reset_index(drop=True).to_parquet(ctx.out / f"units_{rep}.parquet", index=False)
    stats = units.groupby("rep").agg(units=("text", "size"), papers=("paper_key", "nunique"),
                                     words=("text", lambda s: float(s.str.split().str.len().median()))).reset_index()
    ctx.publish("X1 search corpus", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(pp), "fmt": "int"},
            {"label": "gold papers", "value": int(pp["gold"].sum()), "fmt": "int", "hint": f"incl. {n_ws3} WS3 QA-bench gold papers"},
            {"label": "distractors", "value": int((~pp["gold"]).sum()), "fmt": "int"},
            {"label": "papers with full text (chunks)", "value": int(pp["has_text"].sum()), "fmt": "int"}]},
        {"type": "table", "title": "Units per representation", "rows": stats.to_dict("records"),
         "hint": "fields / whole / claims come from the KU (summary); chunks are 500-word segments of the paper text",
         "columns": [{"key": "rep", "label": "representation"}, {"key": "units", "label": "units", "fmt": "int"},
                     {"key": "papers", "label": "papers", "fmt": "int"}, {"key": "words", "label": "median words", "fmt": "int"}]},
    ], summary=f"{len(pp)} papers ({int(pp['gold'].sum())} gold), {len(units)} units")
    return {"papers": len(pp), "units": len(units)}
