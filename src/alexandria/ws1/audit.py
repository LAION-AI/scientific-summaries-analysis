"""Stage `audit` (WS1, reduce): the numbers and method details the paper must state correctly (section 5 of the
plan: dataset and paper consistency fixes).

Always reads the WHOLE local release (arXiv + PubMed) for the light metadata columns (ids, DOIs, OpenAlex id,
summary model, token counts, text length) — cheap, and needed for exact counts — and a seeded sample of shards for
heavy columns (summary fields, raw / sanitized text). Measures:
  * arXiv coverage: rows per arXiv folder, rows with an arXiv id (paper_id or an arXiv DOI), distinct arXiv ids,
    versions, rows without an id (title-matched folders), duplicate ids, and the KU-Fixes cross-subset dedup
    (method = DOI / title+year / title-only clusters; title-only = possible false positives) -> why "99.8% of
    3,011,619" would be ~110% of arXiv;
  * index arithmetic: FAISS IVF-PQ48 bytes per vector (48) vs float32 (3,072) = 64x, index sizes from vector counts;
  * tokens: summary_tokens_out per summary (the summariser's own count) -> total tokens vs the published "~60B",
    and per-field word / token statistics (reference tokenizer) on the sample;
  * model numbers for the paper's tables: parameters and bf16 / fp32 memory (assets/model_params.json, fetched
    on a login node);
  * method details: summariser model(s), input type (full text vs abstract only, from summary_tokens_in as in
    `features`), what "sanitized" removes (raw vs sanitized text on the sample), chunk source (chunk counts per
    subset vs papers with full text), decontamination hits (13-gram overlap of summaries with every eval set),
    and a seeded random sample of rows for the embedding visualisation (instead of "the first 500k").
Plus a checklist of the paper's statements that do not match its tables, with the measured evidence.
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import iter_batches, release_shards
from ..runtime import Context, log, stage
from ..schema import PROSE_FIELDS, SUMMARY_FIELDS

LIGHT = ["paper_id", "subset", "summary_model", "summary_tokens_in", "summary_tokens_out", "text_length",
         "source_doi", "oa_doi", "openalex_id", "oa_is_open_access", "oa_language", "match_method"]
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(v\d+)?", re.I)
_SANITIZE_PATTERNS = {"urls": r"https?://\S+", "emails": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "latex_commands": r"\\[a-zA-Z]+\{",
                      "citation_brackets": r"\[\d+(?:[,–-]\s*\d+)*\]", "control_chars": r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
                      "hyphenated_linebreaks": r"\w-\n\w", "page_headers_numbers": r"\n\s*\d{1,4}\s*\n"}


def arxiv_id(paper_id, *dois) -> tuple[str | None, str | None]:
    """(arXiv id without version, version) from paper_id or an arXiv DOI; (None, None) if there is none."""
    m = _ARXIV_ID.match(str(paper_id or "").strip())
    if m:
        return m.group(1).lower(), m.group(2)
    for d in dois:
        m = _ARXIV_DOI.search(str(d or ""))
        if m:
            return m.group(1).lower(), m.group(2)
    return None, None


def _light(cfg, sh) -> pd.DataFrame:
    frames = [tbl.to_pandas() for tbl in iter_batches(cfg, sh, LIGHT, batch_size=8192)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pq_math(vectors: int, dim: int = 768, pq_bytes: int = 48, nlist: int = 8192) -> dict:
    flat = vectors * dim * 4
    ivfpq = vectors * (pq_bytes + 8) + nlist * dim * 4            # codes + 64-bit ids + coarse centroids
    return {"vectors": vectors, "float32_GB": flat / 1e9, "ivfpq_GB": ivfpq / 1e9,
            "compression_codes_only": dim * 4 / pq_bytes, "compression_with_ids": flat / ivfpq}


@stage("ws1", "audit", config_keys=("ws1.audit", "ws1.checks.input_type_tokens", "ws1.checks.prompt_overhead_tokens"),
       help="paper consistency audit: arXiv coverage + dedup, index arithmetic, token stats, model sizes, method details")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws1.audit")
    settings = ctx.cfg.profile_settings if ctx.cfg.profile == "debug" else {}
    shards = release_shards(ctx.cfg)
    log(f"audit: light columns of {len(shards)} release shards")
    with ThreadPoolExecutor(int(cfg.get("workers", 16))) as ex:
        light = pd.concat(list(ex.map(lambda s: _light(ctx.cfg, s), shards)), ignore_index=True)
    ids = [arxiv_id(p, a, b) for p, a, b in zip(light["paper_id"], light["source_doi"], light["oa_doi"], strict=True)]
    light["arxiv_id"] = [i for i, _ in ids]
    light["arxiv_version"] = [v for _, v in ids]
    ax = light[light["_group"].str.startswith("arxiv")]

    # ---- arXiv coverage and dedup
    folder = ax.groupby("_group").agg(rows=("paper_id", "size"), with_arxiv_id=("arxiv_id", lambda s: int(s.notna().sum())),
                                      with_openalex=("openalex_id", lambda s: int(s.notna().sum()))).reset_index()
    distinct = int(ax["arxiv_id"].nunique())
    dup_ids = int((ax["arxiv_id"].dropna().value_counts() > 1).sum())
    dup_rows = int(ax["arxiv_id"].dropna().duplicated().sum())
    versions = int(ax["arxiv_version"].notna().sum())
    oa_dup = int(ax["openalex_id"].dropna().duplicated().sum())
    no_id = int(ax["arxiv_id"].isna().sum())
    published = cfg.get("arxiv_claim", {"share": 0.998, "of": 3011619})
    claimed = published["share"] * published["of"]
    dup_tbl = []
    dpath = Path(ctx.cfg.site.ku_fixes or "") / "duplicates.parquet"
    if ctx.cfg.site.ku_fixes and dpath.exists():
        d = pd.read_parquet(dpath, columns=["folder", "tier", "cluster_id", "match_keys", "keep"])
        d["method"] = d["match_keys"].map(lambda m: "+".join(sorted(m)) if m is not None else "?")
        dup_tbl = d.groupby(["tier", "method"]).agg(rows=("cluster_id", "size"), clusters=("cluster_id", "nunique"),
                                                    dropped=("keep", lambda s: int((~s).sum()))).reset_index() \
            .sort_values("rows", ascending=False).to_dict("records")
        cross = d.groupby("cluster_id")["folder"].nunique()
        cross_clusters = int((cross > 1).sum())
    else:
        cross_clusters = None
    arxiv_rows = len(ax)
    arxiv_rows_published = int(cfg.get("arxiv_rows_published", 3322974))

    # ---- tokens
    tok_out = pd.to_numeric(light["summary_tokens_out"], errors="coerce")
    per_summary = float(tok_out.mean())
    n_total = int(cfg.get("papers_published", 22087912))
    tokens_claim = float(cfg.get("tokens_claim", 60e9))
    tin = pd.to_numeric(light["summary_tokens_in"], errors="coerce")
    bins = ctx.section("ws1.checks").get("input_type_tokens", {})
    over = int(ctx.section("ws1.checks").get("prompt_overhead_tokens", 1950))
    content = tin - over
    itype = pd.Series("unknown", index=light.index)
    for name, lo in sorted(bins.items(), key=lambda kv: kv[1]):
        itype[content >= lo] = name
    light["input_type"] = itype
    inp = light.groupby(["_group", "input_type"]).size().unstack(fill_value=0)
    models = light["summary_model"].value_counts().to_dict()

    # ---- sample of shards for heavy columns
    rng = random.Random(int(cfg.get("seed", 0)))
    n_sh = int(settings.get("audit_sample_shards", cfg.get("sample_shards", 200)))
    sample_shards = rng.sample(shards, min(n_sh, len(shards)))
    field_rows, san = [], []
    tok = None
    try:
        import os

        from transformers import AutoTokenizer
        os.environ.setdefault("HF_HOME", str(ctx.cfg.site.hf_home))
        m = ctx.cfg.raw.get("models", {}).get(cfg.get("tokenizer", "qwen2.5-7b"), {})
        tok = AutoTokenizer.from_pretrained(m["hf"], revision=m.get("revision")) if m else None
    except Exception as e:                                             # noqa: BLE001  (offline / missing: words only)
        log(f"audit: no reference tokenizer ({type(e).__name__}); word counts only")
    max_rows = int(settings.get("audit_rows_per_shard", cfg.get("rows_per_shard", 200)))
    kus_for_decontam = []
    for sh in sample_shards:
        for tbl in iter_batches(ctx.cfg, sh, [*SUMMARY_FIELDS, "text_raw", "text_sanitized"], batch_size=max_rows):
            df = tbl.to_pandas()
            for f in SUMMARY_FIELDS:
                v = df[f].fillna("").astype(str)
                w = v.str.split().str.len()
                rec = {"field": f, "words": w.tolist()}
                if tok is not None:
                    rec["tokens"] = [len(x) for x in tok(v.tolist(), add_special_tokens=False)["input_ids"]]
                field_rows.append(rec)
            kus_for_decontam += [" ".join(str(r[f] or "") for f in PROSE_FIELDS + ["claims"]) for r in df.to_dict("records")]
            both = df[(df["text_raw"].fillna("") != "") & (df["text_sanitized"].fillna("") != "")]
            for r in both.head(20).to_dict("records"):
                raw, cl = r["text_raw"], r["text_sanitized"]
                san.append({"len_ratio": len(cl) / max(len(raw), 1),
                            **{f"{k}_raw": len(re.findall(p, raw)) for k, p in _SANITIZE_PATTERNS.items()},
                            **{f"{k}_sanitized": len(re.findall(p, cl)) for k, p in _SANITIZE_PATTERNS.items()}})
            break                                                       # first batch of each sampled shard
    fstat = []
    for f in SUMMARY_FIELDS:
        recs = [r for r in field_rows if r["field"] == f]
        w = np.concatenate([r["words"] for r in recs]) if recs else np.array([])
        t = np.concatenate([r["tokens"] for r in recs]) if recs and "tokens" in recs[0] else np.array([])
        fstat.append({"field": f, "papers": int(len(w)), "words_median": float(np.median(w)) if len(w) else None,
                      "words_mean": float(w.mean()) if len(w) else None,
                      "tokens_median": float(np.median(t)) if len(t) else None, "tokens_mean": float(t.mean()) if len(t) else None})
    fstat = pd.DataFrame(fstat)
    sum_tokens_ref = float(fstat["tokens_mean"].sum()) if fstat["tokens_mean"].notna().any() else None
    sant = pd.DataFrame(san)
    san_rows = []
    if len(sant):
        for k in _SANITIZE_PATTERNS:
            r_, s_ = sant[f"{k}_raw"].sum(), sant[f"{k}_sanitized"].sum()
            san_rows.append({"pattern": k, "raw": int(r_), "sanitized": int(s_), "removed_share": float(1 - s_ / r_) if r_ else None})
        san_rows.append({"pattern": "length ratio sanitized / raw (median)", "raw": None, "sanitized": None,
                         "removed_share": float(1 - sant["len_ratio"].median())})

    # ---- decontamination hits of the summaries against every eval set (13-gram)
    from ..ws4.common import ngrams
    from ..ws4.data import EVAL_FILES
    ev = {}
    from huggingface_hub import hf_hub_download
    for name, (repo, f, cols) in EVAL_FILES.items():
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=str(ctx.cfg.site.hf_home / "hub"), local_files_only=True)
        except Exception:                                              # noqa: BLE001  (gated / not fetched)
            continue
        df = pd.read_parquet(p) if f.endswith(".parquet") else pd.read_json(p, lines=True) if f.endswith(".jsonl") else pd.read_csv(p)
        ev[name] = [" ".join(map(str, r)) for r in df[[c for c in cols if c in df.columns]].itertuples(index=False)]
    bench = Path(ctx.cfg.site.store) / "benchmarks"
    if (bench / "litqa2").exists():
        lq = pd.read_parquet(next((bench / "litqa2").glob("*.parquet")))
        ev["litqa2"] = (lq["question"].astype(str) + " " + lq["ideal"].astype(str)).tolist()
    index: dict[tuple, str] = {}
    for name, texts in ev.items():
        for t in texts:
            for g in ngrams(t, 13):
                index.setdefault(g, name)
    hits: dict[str, int] = {}
    for t in kus_for_decontam:
        found = {index[g] for g in ngrams(t, 13) if g in index}
        for name in found:
            hits[name] = hits.get(name, 0) + 1
    decon = [{"eval_set": k, "eval_items": len(v), "summaries_checked": len(kus_for_decontam), "summaries_hit": hits.get(k, 0),
              "share": hits.get(k, 0) / max(len(kus_for_decontam), 1)} for k, v in ev.items()]

    # ---- index arithmetic (search metadata) and chunk source
    meta_p = Path(ctx.cfg.site.data_root) / "search" / "faiss" / "faiss_metadata.json"
    idx_rows, chunk_rows = [], []
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        for kind, m in meta.items():
            pm = pq_math(int(m["total_vectors"]), int(m.get("dim", 768)), int(m.get("pq_bytes", 48)), int(m.get("nlist", 8192)))
            idx_rows.append({"index": f"FAISS {kind}", **pm})
            if kind == "chunks":
                fulltext = cfg.get("fulltext_papers_published", {"arxiv": 3322974, "bethgelab": 334706, "pubmed": 270000,
                                                                 "web-papers": 1390000})
                papers = cfg.get("papers_published_per_subset", {"arxiv": 3322974, "bethgelab": 334706, "pubmed": 1732185,
                                                                 "web-papers": 16698047})
                for s in m["subsets"]:
                    chunk_rows.append({"subset": s["subset"], "chunks": s["vectors"],
                                       "per_paper_with_full_text": s["vectors"] / fulltext.get(s["subset"], np.nan),
                                       "per_summary": s["vectors"] / papers.get(s["subset"], np.nan)})
        summ_vec = int(cfg.get("faiss_summary_vectors", 21796509))
        pm = pq_math(summ_vec, 768, 48, 8192)
        idx_rows.append({"index": "FAISS per summary field (each of 19)", **pm})
        idx_rows.append({"index": "FAISS 19 summary fields total", **{k: (v * 19 if "GB" in k or k == "vectors" else v)
                                                                       for k, v in pm.items()}})

    # ---- model numbers for the paper's tables
    mp = Path(ctx.cfg.site.assets) / "model_params.json"
    model_rows = []
    if mp.exists():
        for name, m in json.loads(mp.read_text()).items():
            if m.get("params"):
                model_rows.append({"model": name, "params_B": m["params"] / 1e9, "bf16_GB": m["params"] * 2 / 1e9,
                                   "bf16_GiB": m["params"] * 2 / 2**30, "fp32_GB": m["params"] * 4 / 1e9,
                                   "stored_dtype": ", ".join(m.get("by_dtype") or {})})

    # ---- seeded random sample for the embedding visualisation
    n_viz = int(cfg.get("viz_sample", 500000))
    viz = light[["_file", "_row", "paper_id", "_group"]].sample(n=min(n_viz, len(light)), random_state=int(cfg.get("seed", 0)))
    viz.to_parquet(ctx.out / "viz_sample.parquet", index=False)

    # ---- outputs
    facts = {"arxiv_rows_local": arxiv_rows, "arxiv_rows_published": arxiv_rows_published,
             "arxiv_rows_with_id": int(ax["arxiv_id"].notna().sum()), "arxiv_rows_without_id": no_id,
             "distinct_arxiv_ids": distinct, "duplicate_arxiv_ids": dup_ids, "duplicate_arxiv_id_rows": dup_rows,
             "rows_with_explicit_version": versions, "duplicate_openalex_ids_in_arxiv": oa_dup,
             "claimed_coverage_papers": claimed, "claimed_share_of_arxiv_rows": arxiv_rows_published / published["of"],
             "cross_subset_dup_clusters": cross_clusters, "tokens_out_mean_per_summary": per_summary,
             "tokens_total_extrapolated": per_summary * n_total, "tokens_claim": tokens_claim,
             "tokens_claim_per_summary": tokens_claim / n_total, "sum_field_tokens_ref_mean": sum_tokens_ref,
             "pq48_compression": 768 * 4 / 48, "summary_models": models}
    ctx.write_json("audit.json", facts)
    for name, df in (("arxiv_folders", folder), ("dedup_methods", pd.DataFrame(dup_tbl)), ("field_tokens", fstat),
                     ("input_type", inp.reset_index()), ("sanitized", pd.DataFrame(san_rows)), ("decontam", pd.DataFrame(decon)),
                     ("indices", pd.DataFrame(idx_rows)), ("chunks", pd.DataFrame(chunk_rows)), ("models", pd.DataFrame(model_rows))):
        df.to_csv(ctx.out / f"{name}.csv", index=False)
    checklist = paper_checklist(facts, fstat, pd.DataFrame(chunk_rows), pd.DataFrame(model_rows), inp)
    pd.DataFrame(checklist).to_csv(ctx.out / "paper_checklist.csv", index=False)

    ratio = arxiv_rows_published / published["of"]
    ctx.publish("WS1 paper consistency audit", [
        {"type": "kpis", "items": [
            {"label": "arXiv rows vs '3,011,619' arXiv papers", "value": ratio, "fmt": "pct1", "status": "warn",
             "hint": f"{arxiv_rows_published:,} rows; 99.8% of 3,011,619 = {claimed:,.0f}"},
            {"label": "distinct arXiv ids in arXiv rows", "value": distinct, "fmt": "int",
             "hint": f"{no_id:,} arXiv rows have no arXiv id (title-matched folders)"},
            {"label": "tokens per summary (summariser count)", "value": per_summary, "fmt": "int",
             "hint": f"'~60B' implies {tokens_claim / n_total:,.0f}"},
            {"label": "PQ48 compression of 768-d float32", "value": 768 * 4 / 48, "fmt": "int", "hint": "64x, not 16x"}]},
        {"type": "table", "title": "Paper fixes: statement, measured evidence, action", "rows": checklist,
         "columns": [{"key": "item", "label": "item"}, {"key": "evidence", "label": "measured"}, {"key": "action", "label": "action"},
                     {"key": "status", "label": "status"}]},
        {"type": "table", "title": "arXiv folders: rows, rows with an arXiv id, OpenAlex matches", "rows": folder.to_dict("records"),
         "columns": [{"key": "_group", "label": "folder"}, {"key": "rows", "label": "rows", "fmt": "int"},
                     {"key": "with_arxiv_id", "label": "with arXiv id", "fmt": "int"},
                     {"key": "with_openalex", "label": "OpenAlex match", "fmt": "int"}]},
        {"type": "table", "title": "Cross-subset dedup (KU-Fixes): method and counts", "rows": dup_tbl,
         "hint": "title_only clusters are possible false positives", "columns": [
             {"key": "tier", "label": "tier"}, {"key": "method", "label": "match keys"}, {"key": "rows", "label": "rows", "fmt": "int"},
             {"key": "clusters", "label": "clusters", "fmt": "int"}, {"key": "dropped", "label": "rows dropped", "fmt": "int"}]},
        {"type": "table", "title": "Per-field length (sample): words and reference tokens", "rows": fstat.to_dict("records"),
         "columns": [{"key": "field", "label": "field"}, {"key": "words_median", "label": "words (median)", "fmt": "int"},
                     {"key": "tokens_median", "label": "tokens (median)", "fmt": "int"}, {"key": "tokens_mean", "label": "tokens (mean)", "fmt": "int"}]},
        {"type": "table", "title": "Index sizes and PQ48 arithmetic", "rows": idx_rows, "columns": [
            {"key": "index", "label": "index"}, {"key": "vectors", "label": "vectors", "fmt": "int"},
            {"key": "float32_GB", "label": "flat float32 GB", "fmt": "float"}, {"key": "ivfpq_GB", "label": "IVF-PQ48 GB", "fmt": "float"},
            {"key": "compression_codes_only", "label": "compression (codes)", "fmt": "int"}]},
        {"type": "table", "title": "Chunk source: chunks per paper with full text vs per summary", "rows": chunk_rows,
         "hint": "~10-13 chunks per full-text paper and << 1 per summary for web-papers: chunks come from full text",
         "columns": [{"key": "subset", "label": "subset"}, {"key": "chunks", "label": "chunks", "fmt": "int"},
                     {"key": "per_paper_with_full_text", "label": "per full-text paper", "fmt": "float"},
                     {"key": "per_summary", "label": "per summary", "fmt": "float"}]},
        {"type": "table", "title": "What 'sanitized' removes (raw vs sanitized text, sample)", "rows": san_rows,
         "columns": [{"key": "pattern", "label": "pattern"}, {"key": "raw", "label": "in raw", "fmt": "int"},
                     {"key": "sanitized", "label": "in sanitized", "fmt": "int"}, {"key": "removed_share", "label": "removed", "fmt": "pct1"}]},
        {"type": "table", "title": "Decontamination: summaries sharing a 13-gram with an eval set (sample)", "rows": decon,
         "columns": [{"key": "eval_set", "label": "eval set"}, {"key": "eval_items", "label": "eval items", "fmt": "int"},
                     {"key": "summaries_hit", "label": "summaries hit", "fmt": "int"}, {"key": "share", "label": "share", "fmt": "pct1"}]},
        {"type": "table", "title": "Model sizes for the paper's tables", "rows": model_rows, "columns": [
            {"key": "model", "label": "model"}, {"key": "params_B", "label": "parameters (B)", "fmt": "float"},
            {"key": "bf16_GB", "label": "bf16 GB", "fmt": "float"}, {"key": "fp32_GB", "label": "fp32 GB", "fmt": "float"},
            {"key": "stored_dtype", "label": "stored as"}]},
    ], summary=f"arXiv rows = {ratio:.0%} of the cited arXiv total; {per_summary:,.0f} tokens / summary; PQ48 = 64x")
    return {k: v for k, v in facts.items() if not isinstance(v, dict)}


def paper_checklist(f: dict, fstat: pd.DataFrame, chunks: pd.DataFrame, models: pd.DataFrame, inp: pd.DataFrame) -> list[dict]:
    """Section 5 items with the measured evidence and the action for the paper text."""
    def mdl(name):
        r = models[models["model"] == name] if len(models) else models
        return (f"{float(r['params_B'].iloc[0]):.2f}B params, {float(r['bf16_GB'].iloc[0]):.2f} GB bf16, "
                f"stored {r['stored_dtype'].iloc[0]} ({float(r['fp32_GB'].iloc[0]):.2f} GB)") if len(r) else "n/a"
    ft = inp.sum().to_dict() if len(inp) else {}
    return [
        {"item": "arXiv coverage '99.8% of 3,011,619'", "status": "fix numbers",
         "evidence": f"{f['arxiv_rows_published']:,} arXiv rows = {f['claimed_share_of_arxiv_rows']:.0%} of 3,011,619; "
                     f"{f['distinct_arxiv_ids']:,} distinct arXiv ids; {f['arxiv_rows_without_id']:,} rows without an id "
                     f"(old/titled folders, title-matched); {f['duplicate_arxiv_id_rows']:,} duplicate-id rows",
         "action": "report distinct papers, explain folders / versions / dedup and title-match false positives"},
        {"item": "18 vs 19 fields; 44 columns", "status": "fix text", "evidence": "18 LLM fields + concatenated_summary = 19 indexed; 44 = 7 + 18 + 3 + 13 + 3",
         "action": "add the full 44-column schema table (resources/canonical_schema.json)"},
        {"item": "22,087,912 papers vs 21.8M FAISS vs 21.4M BM25", "status": "explain", "evidence": "see `reconcile` rows",
         "action": "state per index which rows are excluded (titled subset, empty fields)"},
        {"item": "Table 2 index sizes / two chunk and two claim indices", "status": "fix table",
         "evidence": "BM25 + FAISS each for chunks and claims (4 indices); FAISS sizes from vectors x 56 B + centroids",
         "action": "list all indices with sizes incl. claims; recompute the ~345 GB total"},
        {"item": "PQ48 compression", "status": "fix number", "evidence": f"768 x 4 B / 48 B = {f['pq48_compression']:.0f}x", "action": "64x, not 16x"},
        {"item": "~60B tokens", "status": "fix number",
         "evidence": f"{f['tokens_out_mean_per_summary']:,.0f} output tokens per summary (summariser count) x 22.09M = "
                     f"{f['tokens_total_extrapolated'] / 1e9:,.1f}B; '60B' implies {f['tokens_claim_per_summary']:,.0f} per summary",
         "action": "report real per-field token statistics (field_tokens.csv)"},
        {"item": "Table 6 memory 9.884 GB / Table 5 params", "status": "fix tables",
         "evidence": f"Llama-3.2-1B: {mdl('meta-llama/Llama-3.2-1B')}; Gemma-2-2B: {mdl('google/gemma-2-2b')}",
         "action": "use parameter counts from the checkpoints; memory = params x bytes of the dtype actually loaded"},
        {"item": "'knowledge-intensive benchmarks'", "status": "fix text", "evidence": "only MMLU in that table", "action": "singular, or add benchmarks (WS4 eval set)"},
        {"item": "'commonsense trade-off'", "status": "fix text", "evidence": "driven by BoolQ alone, below chance", "action": "drop the claim or show it without BoolQ"},
        {"item": "Table 4 text / prompt mentions 'summaries' in every condition", "status": "replace", "evidence": "WS3 results",
         "action": "replace Table 4 with the WS3 main table"},
        {"item": "HLE is 'multiple choice'", "status": "fix text", "evidence": "HLE mixes exact-match and multiple-choice questions",
         "action": "describe as mixed exact-match / multiple-choice, text-only subset"},
        {"item": "RRF 'fuses scores'", "status": "fix text", "evidence": "RRF sums 1/(k + rank)", "action": "'fuses ranks'"},
        {"item": "summariser model(s), prompt, cost, failure rate", "status": "add", "evidence": f"models: {f['summary_models']}; see `provenance`",
         "action": "state model, prompt (publish it), cost and failure rate from `provenance`"},
        {"item": "full text vs abstract only", "status": "add", "evidence": f"input types (local arXiv + PubMed): {ft}", "action": "report per subset (input_type.csv)"},
        {"item": "what 'sanitized' means", "status": "add", "evidence": "sanitized.csv: patterns removed vs raw", "action": "define the sanitisation steps"},
        {"item": "decontamination hit counts", "status": "add", "evidence": "decontam.csv (13-gram, sample)", "action": "report hits per eval set; decontaminate before training"},
        {"item": "500-word chunks from full text or summaries", "status": "add",
         "evidence": "; ".join(f"{r.subset}: {r.per_paper_with_full_text:.1f} per full-text paper" for r in chunks.itertuples()) if len(chunks) else "n/a",
         "action": "state: chunks are cut from full text"},
        {"item": "embedding visualisation 'first 500k'", "status": "fixed", "evidence": "viz_sample.parquet: seeded random sample",
         "action": "redraw the figure from the random sample"},
        {"item": "per-record licence; § 60d UrhG vs CC-BY; opt-out; 'legal defensibility'", "status": "see `licenses`",
         "evidence": "licence per record from OpenAlex / arXiv; takedown process in docs/TAKEDOWN.md",
         "action": "add a licence column; reconcile non-commercial TDM with CC-BY; remove 'establishes legal defensibility'; lawyer review"},
    ]
