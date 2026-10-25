"""Stage `benchmarks` (X1) and the login-node `fetch`: the queries and their gold papers, linked to release rows.

  litsearch  LitSearch (Ajith et al., 2024): 597 literature-search queries; gold = corpusids of its corpus, linked by
             exact normalised title (corpus_clean titles) to release rows.
  citeme     CiteME (Press et al., 2024): citation excerpts; gold = the cited paper (title). Gated on the Hub: needs
             HF_TOKEN with accepted terms (the project website no longer serves it); listed as not run otherwise.
  scifact    SciFact dev claims (Wadden et al., 2020); gold = the evidence documents (corpus.jsonl titles).
Linking reuses the WS3 id lookup over every local release row (DOI, else exact title >= `min_title_words` words). Only
queries with at least one linked gold paper are evaluated (the share is reported per benchmark): a gold paper that
is not in the release cannot be retrieved from it.

    python -c "from alexandria.config import load_config; from alexandria.x1.benchmarks import fetch; fetch(load_config('x1','debug'))"
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ..runtime import Context, log, stage
from ..ws1.faith.common import paper_key
from ..ws3.gold import id_lookup, norm_doi, norm_title
from .common import setting

LITSEARCH_CORPUS = [f"corpus_clean/full-0000{i}-of-00006.parquet" for i in range(6)]


def _hub(cfg) -> str:
    return str(cfg.site.hf_home / "hub")


def fetch(cfg) -> dict[str, str]:
    """Login node: download LitSearch (queries + corpus titles) and, with HF_TOKEN, CiteME."""
    from huggingface_hub import hf_hub_download
    b, out = cfg.section("x1").get("benchmarks", {}), {}
    ls = b.get("litsearch", {})
    for f in ["query/full-00000-of-00001.parquet", *LITSEARCH_CORPUS]:
        hf_hub_download(ls["repo"], f, repo_type="dataset", revision=ls["revision"], cache_dir=_hub(cfg))
    out["litsearch"] = "ok"
    c = b.get("citeme", {})
    try:
        hf_hub_download(c["repo"], c["file"], repo_type="dataset", revision=c["revision"], cache_dir=_hub(cfg),
                        token=os.environ.get("HF_TOKEN"))
        out["citeme"] = "ok"
    except Exception as e:                                    # noqa: BLE001  (gated)
        out["citeme"] = f"not available ({type(e).__name__}): gated, needs HF_TOKEN with accepted terms"
    log(f"x1 fetch: {out}")
    return out


def _local(cfg, repo: str, rev: str, f: str):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, f, repo_type="dataset", revision=rev, cache_dir=_hub(cfg), local_files_only=True)


def load_litsearch(cfg) -> list[dict]:
    import pyarrow.parquet as pq
    ls = cfg.section("x1")["benchmarks"]["litsearch"]
    q = pd.read_parquet(_local(cfg, ls["repo"], ls["revision"], "query/full-00000-of-00001.parquet"))
    titles = {}
    for f in LITSEARCH_CORPUS:
        t = pq.read_table(_local(cfg, ls["repo"], ls["revision"], f), columns=["corpusid", "title"]).to_pandas()
        titles.update(dict(zip(t["corpusid"].astype(str), t["title"], strict=True)))
    return [{"bench": "litsearch", "qid": f"litsearch:{i}", "query": r.query, "category": r.query_set,
             "gold_refs": [{"title": titles.get(str(c))} for c in r.corpusids]} for i, r in enumerate(q.itertuples())]


def load_citeme(cfg) -> list[dict]:
    c = cfg.section("x1")["benchmarks"]["citeme"]
    df = pd.read_csv(_local(cfg, c["repo"], c["revision"], c["file"]), sep="\t")
    qcol = next(x for x in ("excerpt", "text", "query") if x in df.columns)
    tcol = next(x for x in ("target_paper_title", "title", "target_title") if x in df.columns)
    return [{"bench": "citeme", "qid": f"citeme:{i}", "query": str(r[qcol]), "category": "citation",
             "gold_refs": [{"title": r[tcol]}]} for i, r in df.iterrows()]


def load_scifact(cfg) -> list[dict]:
    root = cfg.site.store
    d = os.path.join(str(root), "benchmarks", "scifact")
    titles = {str(x["doc_id"]): x["title"] for x in map(json.loads, open(os.path.join(d, "corpus.jsonl")))}
    out = []
    for line in open(os.path.join(d, "claims_dev.jsonl")):
        c = json.loads(line)
        gold = [titles[k] for k in (c.get("evidence") or {}) if k in titles]
        if gold:                                               # claims with evidence documents (BEIR SciFact protocol)
            out.append({"bench": "scifact", "qid": f"scifact:{c['id']}", "query": c["claim"], "category": "claim",
                        "gold_refs": [{"title": t} for t in gold]})
    return out


def link_all(items: list[dict], lookup: pd.DataFrame, min_words: int) -> pd.DataFrame:
    """Every gold reference of every query -> release rows (all of them, not only the first)."""
    best = lookup.drop_duplicates(["kind", "key"])             # sorted: full-text rows first
    by = {(r.kind, r.key): (r.file, int(r.row)) for r in best.itertuples()}
    rows = []
    for it in items:
        keys, files, rws = [], [], []
        for ref in it["gold_refs"]:
            hit = by.get(("doi", norm_doi(ref.get("doi")))) if ref.get("doi") else None
            if hit is None and ref.get("title"):
                hit = by.get(("title", norm_title(ref.get("title"), min_words)))
            if hit is not None and paper_key(*hit) not in keys:
                keys.append(paper_key(*hit))
                files.append(hit[0])
                rws.append(hit[1])
        rows.append({**{k: v for k, v in it.items() if k != "gold_refs"},
                     "gold_titles": [r.get("title") for r in it["gold_refs"]], "n_gold": len(it["gold_refs"]),
                     "gold_keys": keys, "gold_files": files, "gold_rows": rws, "n_linked": len(keys)})
    return pd.DataFrame(rows)


@stage("x1", "benchmarks", config_keys=("x1.benchmarks",),
       help="LitSearch / CiteME / SciFact queries with gold papers linked to release rows (DOI / exact title)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x1.benchmarks")
    loaders = {"litsearch": load_litsearch, "citeme": load_citeme, "scifact": load_scifact}
    items, status = [], []
    for name in cfg.get("names", list(loaders)):
        try:
            its = loaders[name](ctx.cfg)
        except (FileNotFoundError, OSError, StopIteration) as e:
            status.append({"bench": name, "queries": 0, "linked": 0, "evaluated": 0,
                           "status": "not run: gated, needs HF_TOKEN" if name == "citeme" else f"not available ({type(e).__name__})"})
            continue
        items += its
    mw = int(cfg.get("min_title_words", 5))
    q = link_all(items, id_lookup(ctx.cfg, mw), mw) if items else pd.DataFrame()
    n_max = setting(ctx, "max_queries", cfg.get("max_queries"))
    parts = []
    for name, g in (q.groupby("bench", sort=False) if len(q) else []):
        lk = g[g["n_linked"] > 0]
        ev = lk.sample(n=min(len(lk), int(n_max)), random_state=int(cfg.get("seed", 0))) if n_max else lk
        parts.append(ev)
        status.append({"bench": name, "queries": len(g), "linked": len(lk), "evaluated": len(ev),
                       "gold_papers_linked": int(lk["n_linked"].sum()), "status": "ok"})
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ev.to_parquet(ctx.out / "queries.parquet", index=False)
    q.to_parquet(ctx.out / "all_queries.parquet", index=False)
    ctx.publish("X1 benchmarks", [
        {"type": "kpis", "items": [
            {"label": "queries evaluated", "value": len(ev), "fmt": "int"},
            {"label": "gold papers in the corpus", "value": int(ev["n_linked"].sum()) if len(ev) else 0, "fmt": "int"}]},
        {"type": "table", "title": "Queries per benchmark and gold-paper linkage", "rows": status,
         "hint": "only queries whose gold paper is a release row (DOI / exact title) can be evaluated",
         "columns": [{"key": "bench", "label": "benchmark"}, {"key": "queries", "label": "queries", "fmt": "int"},
                     {"key": "linked", "label": "gold linked", "fmt": "int"}, {"key": "evaluated", "label": "evaluated", "fmt": "int"},
                     {"key": "status", "label": "status"}]},
    ], summary=f"{len(ev)} queries with a linked gold paper")
    return {"evaluated": len(ev), "status": status}
