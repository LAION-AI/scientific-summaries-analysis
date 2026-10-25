"""Stages `index` and `search` (X1, GPU for the encoder; search itself on CPU, as the released indices are used).

index   Per representation: BM25 (bm25s, English stemmer: the release's settings), dense gte-modernbert vectors in
        fp32 (exact; L2-normalised, inner product), and an IVF-PQ48 FAISS index trained on the same vectors with the
        release's vectors-per-list and probed share (summaries: IVF4096,PQ48 over 21.8M, nprobe 32).
search  Every query x representation x retriever (bm25 / dense flat / dense PQ48 / RRF of bm25 + dense): units ->
        papers (max score), top `paper_depth` papers stored. Also: PQ48 vs flat agreement (unit recall@10/100 of the
        exact top-k), and timing: query encoding, single-query search latency (p50 / p95 over `latency_queries`),
        and batch throughput (QPS), on the job's CPU cores (search) and GPU slice (encoding).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ..runtime import Context, log, stage
from .common import REPRESENTATIONS, bm25_tokenize, build_pq, load_encoder, out_dir, pq_params, rrf, stage_table, to_papers


def _seconds(fn, *args, **kw) -> float:
    t = time.perf_counter()
    fn(*args, **kw)
    return time.perf_counter() - t


def _hardware() -> dict:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)), "gpu": gpu,
            "gres": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")}


@stage("x1", "index", needs=("corpus",), resources="gpu", config_keys=("x1.index",),
       help="per representation: bm25s index, exact gte-modernbert vectors, IVF-PQ48 index (release settings)")
def run_index(ctx: Context) -> dict:
    import bm25s
    import faiss
    cfg = ctx.section("x1.index")
    enc = load_encoder(ctx, cfg.get("embedder", "gte-modernbert"), batch_size=int(cfg.get("batch_size", 32)))
    b = cfg.get("bm25", {})
    stats = []
    for rep in REPRESENTATIONS:
        p = out_dir(ctx, "corpus") / f"units_{rep}.parquet"
        if not p.exists():
            continue
        texts = pd.read_parquet(p, columns=["text"])["text"].tolist()
        t0 = time.time()
        bm = bm25s.BM25(k1=float(b.get("k1", 1.5)), b=float(b.get("b", 0.75)), method=b.get("method", "lucene"))
        bm.index(bm25_tokenize(texts), show_progress=False)
        bm.save(str(ctx.out / f"bm25_{rep}"))
        t_bm = time.time() - t0
        t0 = time.time()
        vecs = enc.encode(texts, int(cfg.get("max_length", {}).get(rep, 512))).astype(np.float32)
        t_enc = time.time() - t0
        np.save(ctx.out / f"emb_{rep}.npy", vecs)
        nlist, nprobe = pq_params(len(vecs), cfg)
        t0 = time.time()
        pq = build_pq(vecs, nlist, int(cfg.get("pq_m", 48)), int(cfg.get("pq_bits", 8)))
        faiss.write_index(pq, str(ctx.out / f"pq48_{rep}.faiss"))
        t_pq = time.time() - t0
        stats.append({"rep": rep, "units": len(texts), "nlist": nlist, "nprobe": nprobe,
                      "flat_mb": vecs.nbytes / 1e6, "pq_mb": os.path.getsize(ctx.out / f"pq48_{rep}.faiss") / 1e6,
                      "bm25_s": t_bm, "encode_s": t_enc, "pq_train_s": t_pq, "units_per_s": len(texts) / max(t_enc, 1e-9)})
        log(f"index: {rep}: {len(texts)} units, nlist {nlist}, encode {t_enc:.0f}s")
    ctx.write_json("index.json", {"reps": stats, "hardware": _hardware()})
    ctx.publish("X1 indices", [
        {"type": "table", "title": "Indices per representation", "rows": stats,
         "hint": "IVF-PQ48 scaled from the release: same vectors per inverted list, same probed share; 768-d float32 = "
                 "3,072 bytes per vector vs 48-byte PQ codes (64x)",
         "columns": [{"key": "rep", "label": "representation"}, {"key": "units", "label": "units", "fmt": "int"},
                     {"key": "nlist", "label": "IVF lists", "fmt": "int"}, {"key": "nprobe", "label": "nprobe", "fmt": "int"},
                     {"key": "flat_mb", "label": "flat MB", "fmt": "float"}, {"key": "pq_mb", "label": "PQ48 MB", "fmt": "float"},
                     {"key": "units_per_s", "label": "encoded units / s", "fmt": "float"}]}],
        summary=f"{len(stats)} representations indexed")
    return {"reps": [s["rep"] for s in stats]}


@stage("x1", "search", needs=("index", "benchmarks"), resources="gpu", config_keys=("x1.search",),
       help="every query x representation x retriever (BM25 / dense / PQ48 / RRF): paper rankings, PQ loss, latency, QPS")
def run_search(ctx: Context) -> dict:
    import bm25s
    import faiss
    cfg, icfg = ctx.section("x1.search"), ctx.section("x1.index")
    faiss.omp_set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    q = stage_table(ctx, "benchmarks", "queries.parquet")[["bench", "qid", "query"]]
    ws3 = ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet"
    if ws3.exists():
        it = pd.read_parquet(ws3)
        it = it[(it["bench"] == "qabench-open") & it["gold_paper_key"].notna()]
        q = pd.concat([q, pd.DataFrame({"bench": "ws3-qabench", "qid": it["item_id"], "query": it["question"]})], ignore_index=True)
    enc = load_encoder(ctx, icfg.get("embedder", "gte-modernbert"))
    queries = q["query"].tolist()
    t0 = time.time()
    qv = enc.encode(queries, 256).astype(np.float32)
    enc_batch_s = time.time() - t0
    lat_n = min(int(cfg.get("latency_queries", 100)), len(queries))
    t_single = []
    for s in queries[:lat_n]:
        t1 = time.perf_counter()
        enc.encode([s], 256)
        t_single.append(time.perf_counter() - t1)
    depth, pdepth, kk = int(cfg.get("unit_depth", 1000)), int(cfg.get("paper_depth", 100)), int(cfg.get("rrf_k", 60))
    idx_dir = out_dir(ctx, "index")
    rank_rows, timing, pq_rows = [], [{"rep": "-", "retriever": "query encoding", "p50_ms": 1000 * np.median(t_single),
                                       "p95_ms": 1000 * np.percentile(t_single, 95), "qps_batch": len(queries) / enc_batch_s}], []
    for rep in [r for r in ("fields", "whole", "claims", "chunks") if (idx_dir / f"emb_{r}.npy").exists()]:
        unit_paper = pd.read_parquet(out_dir(ctx, "corpus") / f"units_{rep}.parquet", columns=["paper_key"])["paper_key"].to_numpy()
        vecs = np.load(idx_dir / f"emb_{rep}.npy")
        d = min(depth, len(vecs))
        bm = bm25s.BM25.load(str(idx_dir / f"bm25_{rep}"))
        flat = faiss.IndexFlatIP(vecs.shape[1])
        flat.add(vecs)
        pq = faiss.read_index(str(idx_dir / f"pq48_{rep}.faiss"))
        pq.nprobe = pq_params(len(vecs), icfg)[1]
        res = {}
        t0 = time.time()
        bi, bs = bm.retrieve(bm25_tokenize(queries), k=d, show_progress=False, n_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        tb = time.time() - t0
        t0 = time.time()
        fs, fi = flat.search(qv, d)
        tf = time.time() - t0
        t0 = time.time()
        ps, pi = pq.search(qv, d)
        tp = time.time() - t0
        res["bm25"] = [to_papers(bi[i], bs[i], unit_paper, pdepth)[0] for i in range(len(queries))]
        res["dense"] = [to_papers(fi[i], fs[i], unit_paper, pdepth)[0] for i in range(len(queries))]
        res["dense_pq48"] = [to_papers(pi[i], ps[i], unit_paper, pdepth)[0] for i in range(len(queries))]
        res["rrf"] = [rrf([res["bm25"][i], res["dense"][i]], kk, pdepth) for i in range(len(queries))]
        for name, ranks in res.items():
            rank_rows += [{"bench": b_, "qid": qid, "rep": rep, "retriever": name, "ranked": r}
                          for b_, qid, r in zip(q["bench"], q["qid"], ranks, strict=True)]
        for k in (10, 100):
            kk_ = min(k, d)
            pq_rows.append({"rep": rep, "k": k, "unit_recall_vs_flat": float(np.mean(
                [len(set(pi[i, :kk_]) & set(fi[i, :kk_])) / kk_ for i in range(len(queries))]))})
        single = {
            "bm25": [_seconds(bm.retrieve, bm25_tokenize([queries[i]]), k=d, show_progress=False, n_threads=1)
                     for i in range(lat_n)],
            "dense": [_seconds(flat.search, qv[i:i + 1], d) for i in range(lat_n)],
            "dense_pq48": [_seconds(pq.search, qv[i:i + 1], d) for i in range(lat_n)]}
        for name, tot in (("bm25", tb), ("dense", tf), ("dense_pq48", tp)):
            timing.append({"rep": rep, "retriever": name, "units": len(vecs), "p50_ms": 1000 * np.median(single[name]),
                           "p95_ms": 1000 * np.percentile(single[name], 95), "qps_batch": len(queries) / max(tot, 1e-9)})
        log(f"search: {rep}: {len(queries)} queries done")
    pd.DataFrame(rank_rows).to_parquet(ctx.out / "rankings.parquet", index=False)
    pd.DataFrame(timing).to_csv(ctx.out / "timing.csv", index=False)
    pd.DataFrame(pq_rows).to_csv(ctx.out / "pq_vs_flat.csv", index=False)
    ctx.write_json("hardware.json", _hardware())
    ctx.publish("X1 search runs", [{"type": "kpis", "items": [
        {"label": "queries", "value": len(queries), "fmt": "int"},
        {"label": "rankings", "value": len(rank_rows), "fmt": "int"},
        {"label": "hardware", "value": json.dumps(_hardware())}]}], summary=f"{len(queries)} queries x {len(rank_rows) // max(len(queries), 1)} runs")
    return {"queries": len(queries)}
