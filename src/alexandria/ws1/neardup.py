"""Stages `neardup-sign` (map) and `neardup` (reduce): near-duplicate SUMMARIES via MinHash-LSH.

KU-Fixes already clusters paper-level duplicates by identifiers (paper_id, DOI, title+year). What it cannot see is
two DIFFERENT papers whose summaries are near-identical: templated or boilerplate generations, or the same paper under
unrelated identifiers. This stage finds both and separates them:
    known_duplicate   every member of the cluster is in one KU-Fixes duplicate cluster (already known)
    new               otherwise (the new finding: near-identical summaries that identifier matching misses)

neardup-sign: word k-shingles (textutil.tokens) of one field -> MinHash (datasketch, fixed seed) -> `bands` LSH band
              hashes (uint64) + the signature (uint32) for verification. One parquet per input shard.
neardup:      DuckDB over the parts: (band, hash) buckets with 2..max_bucket members -> candidate pairs -> keep pairs
              whose estimated Jaccard (share of equal signature slots) >= jaccard_threshold -> connected components
              (scipy, over candidate pairs only) -> clusters.parquet. Memory stays bounded at ~5M documents: the
              signatures never leave DuckDB; Python only sees the verified pairs.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pyarrow as pa

from .. import textutil as tu
from ..config import Config
from ..data import Shard, input_shards, iter_batches
from ..papers import paper_table, with_paper_info
from ..runtime import Context, log, stage
from ..stats import rate_row

SEED = 1


# ---------------------------------------------------------------- pure helpers (unit-tested)
def shingles(text: str | None, k: int) -> set[bytes]:
    """Word k-shingles of the lower-cased token stream; a text shorter than k is one shingle; empty -> empty set."""
    toks = tu.tokens(text)
    if not toks:
        return set()
    if len(toks) < k:
        return {" ".join(toks).encode()}
    return {" ".join(toks[i:i + k]).encode() for i in range(len(toks) - k + 1)}


def band_hashes(signature: np.ndarray, bands: int) -> list[int]:
    """Split a signature into `bands` equal rows and hash each band to an unsigned 64-bit integer."""
    if len(signature) % bands:
        raise ValueError(f"num_perm {len(signature)} is not divisible by bands {bands}")
    sig = np.ascontiguousarray(signature, dtype=np.uint32)
    r = len(sig) // bands
    return [int.from_bytes(hashlib.blake2b(sig[b * r:(b + 1) * r].tobytes(), digest_size=8).digest(), "little")
            for b in range(bands)]


def estimated_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def connected_components(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edges (left[i], right[i]) over arbitrary integer node ids -> (nodes, component label per node).
    Only nodes that appear in an edge are returned; labels are 0..C-1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    if left.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    nodes, inv = np.unique(np.concatenate([left, right]), return_inverse=True)
    n = nodes.size
    m = coo_matrix((np.ones(left.size, np.int8), (inv[:left.size], inv[left.size:])), shape=(n, n))
    _, labels = cc(m, directed=False)
    return nodes, labels.astype(np.int64)


# ---------------------------------------------------------------- map: signatures
def sign_shard(cfg: Config, shard: Shard) -> pa.Table:
    from datasketch import MinHash
    nd = cfg.raw["ws1"].get("neardup") or {}
    field = nd.get("field", "executive_summary")
    k, num_perm, bands = int(nd.get("shingle_words", 5)), int(nd.get("num_perm", 128)), int(nd.get("bands", 16))
    files, rows, groups, pids, nsh, sigs, bh = [], [], [], [], [], [], []
    for tbl in iter_batches(cfg, shard, ["paper_id", field], batch_size=1024):
        recs = tbl.to_pylist()
        sets = [shingles(r[field], k) for r in recs]
        keep = [i for i, s in enumerate(sets) if s]
        mhs = MinHash.bulk([sets[i] for i in keep], num_perm=num_perm, seed=SEED) if keep else []
        for i, mh in zip(keep, mhs, strict=True):
            r = recs[i]
            sig = mh.hashvalues.astype(np.uint32)
            files.append(r["_file"]), rows.append(r["_row"]), groups.append(r["_group"])
            pids.append(r["paper_id"]), nsh.append(len(sets[i]))
            sigs.append(sig), bh.append(band_hashes(sig, bands))
    return pa.table({
        "file": pa.array(files, pa.string()), "row": pa.array(rows, pa.int64()),
        "group": pa.array(groups, pa.string()), "paper_id": pa.array(pids, pa.string()),
        "n_shingles": pa.array(nsh, pa.int32()),
        "bands": pa.array(bh, pa.list_(pa.uint64())),
        "signature": pa.array([s.tolist() for s in sigs], pa.list_(pa.uint32())),
    })


@stage("ws1", "neardup-sign", kind="map", resources="cpu", config_keys=("ws1.neardup",),
       help="MinHash signatures + LSH band hashes of one summary field")
def run_sign(ctx: Context) -> dict:
    return ctx.map_shards(sign_shard)


# ---------------------------------------------------------------- reduce: clusters
@stage("ws1", "neardup", kind="reduce", needs=("neardup-sign",), resources="cpu", config_keys=("ws1.neardup",),
       help="LSH candidate pairs -> verified near-duplicate summary clusters, split known vs new")
def run_clusters(ctx: Context) -> dict:
    import duckdb
    nd = ctx.section("ws1.neardup")
    field = nd.get("field", "executive_summary")
    thr, max_bucket = float(nd.get("jaccard_threshold", 0.8)), int(nd.get("max_bucket", 500))
    parts = ctx.parts("neardup-sign")
    if not parts:
        raise FileNotFoundError("neardup-sign has no parts; run it first")
    glob = str(ctx.input_dir("neardup-sign") / "parts" / "*.parquet")
    con = duckdb.connect()
    con.execute(f"SET temp_directory='{ctx.out / 'duckdb_tmp'}'")
    con.execute(f"""CREATE TABLE sig AS SELECT row_number() OVER (ORDER BY file, row) - 1 AS doc,
                    file, row, "group", paper_id, signature, bands FROM read_parquet('{glob}')""")
    n_docs = con.execute("SELECT count(*) FROM sig").fetchone()[0]
    con.execute("""CREATE TABLE b AS SELECT doc, unnest(bands) AS h, generate_subscripts(bands, 1) AS bi FROM sig""")
    con.execute(f"""CREATE TABLE buckets AS SELECT bi, h FROM b GROUP BY bi, h
                    HAVING count(*) BETWEEN 2 AND {max_bucket}""")
    n_big = con.execute(f"SELECT count(*) FROM (SELECT 1 FROM b GROUP BY bi, h HAVING count(*) > {max_bucket})").fetchone()[0]
    con.execute("""CREATE TABLE cand AS SELECT DISTINCT x.doc AS d1, y.doc AS d2
                   FROM b x JOIN buckets USING (bi, h) JOIN b y ON x.bi = y.bi AND x.h = y.h AND x.doc < y.doc""")
    n_cand = con.execute("SELECT count(*) FROM cand").fetchone()[0]
    con.execute("""CREATE TABLE pairs AS SELECT d1, d2,
                     list_sum(list_transform(list_zip(s1.signature, s2.signature), lambda z: (z[1] = z[2])::INT))
                       / len(s1.signature) AS jaccard
                   FROM cand JOIN sig s1 ON s1.doc = d1 JOIN sig s2 ON s2.doc = d2""")
    con.execute(f"DELETE FROM pairs WHERE jaccard < {thr}")
    arr = con.execute("SELECT d1, d2 FROM pairs").fetchnumpy()
    d1, d2 = np.asarray(arr["d1"], np.int64), np.asarray(arr["d2"], np.int64)
    nodes, labels = connected_components(d1, d2)
    log(f"neardup: {n_docs} docs, {n_cand} candidate pairs, {len(d1)} verified, {len(nodes)} docs in clusters")
    con.register("comp", pa.table({"doc": pa.array(nodes, pa.int64()), "cluster_id": pa.array(labels, pa.int64())}))
    ku = ctx.cfg.site.ku_fixes
    if ku is not None and (ku / "duplicates.parquet").exists():
        dup_src = f"read_parquet('{ku / 'duplicates.parquet'}')"
    else:                                          # dataset without KU-Fixes: no known paper duplicates
        dup_src = ("(select null::varchar as file, null::bigint as row_index, null::bigint as cluster_id, "
                   "null::varchar as tier where false)")
    con.execute(f"""CREATE TABLE clusters AS
        SELECT s.file, s.row, s."group", s.paper_id, c.cluster_id,
               count(*) OVER (PARTITION BY c.cluster_id) AS cluster_size, d.cluster_id AS ku_cluster_id, d.tier AS ku_tier
        FROM comp c JOIN sig s USING (doc)
        LEFT JOIN {dup_src} d ON d.file = s.file AND d.row_index = s.row""")
    con.execute("""CREATE TABLE cinfo AS SELECT cluster_id, any_value(cluster_size) AS cluster_size,
                     count(DISTINCT paper_id) AS n_papers, count(DISTINCT "group") AS n_groups,
                     CASE WHEN count(ku_cluster_id) = count(*) AND count(DISTINCT ku_cluster_id) = 1
                          THEN 'known_duplicate' ELSE 'new' END AS category
                   FROM clusters GROUP BY cluster_id""")
    con.execute(f"COPY (SELECT c.*, i.category FROM clusters c JOIN cinfo i USING (cluster_id) ORDER BY cluster_id, file, row) "
                f"TO '{ctx.out / 'clusters.parquet'}' (FORMAT parquet)")
    con.execute(f"""COPY (SELECT s1.file AS file1, s1.row AS row1, s2.file AS file2, s2.row AS row2, p.jaccard
                    FROM pairs p JOIN sig s1 ON s1.doc = p.d1 JOIN sig s2 ON s2.doc = p.d2
                    ORDER BY p.jaccard DESC LIMIT 10000) TO '{ctx.out / 'pairs_sample.parquet'}' (FORMAT parquet)""")

    tot = con.execute("""SELECT count(*), count(DISTINCT cluster_id),
                           sum((category = 'new')::INT), count(DISTINCT CASE WHEN category = 'new' THEN cluster_id END)
                         FROM clusters JOIN cinfo USING (cluster_id)""").fetchone()
    in_cl, n_cl, in_new, n_new = (int(x or 0) for x in tot)
    by_group = con.execute("""SELECT g, n, coalesce(k, 0), coalesce(kn, 0) FROM
        (SELECT "group" g, count(*) n FROM sig GROUP BY 1)
        LEFT JOIN (SELECT "group" g, count(*) k, sum((category = 'new')::INT) kn
                   FROM clusters JOIN cinfo USING (cluster_id) GROUP BY 1) USING (g) ORDER BY g""").fetchall()
    largest = [dict(zip(["cluster_id", "cluster_size", "n_papers", "n_groups", "category"], r, strict=True))
               for r in con.execute("SELECT cluster_id, cluster_size, n_papers, n_groups, category FROM cinfo "
                                    "ORDER BY cluster_size DESC, cluster_id LIMIT 50").fetchall()]
    clustered = con.execute("""SELECT c.file, c.row, c.cluster_id, i.cluster_size, i.category FROM clusters c
                               JOIN cinfo i USING (cluster_id)
                               ORDER BY i.category DESC, i.cluster_size DESC, c.cluster_id, c.file, c.row""").df()
    clustered = with_paper_info(ctx, clustered)
    clustered.to_csv(ctx.out / "clustered_papers.csv", index=False)

    group_rows = []
    for g, n, k, kn in by_group:
        rr, rn = rate_row(k, n), rate_row(kn, n)
        group_rows.append({"group": g, "docs": n, "in_clusters": k, "rate": rr["rate"], "lo": rr["lo"], "hi": rr["hi"],
                           "in_new_clusters": kn, "rate_new": rn["rate"], "lo_new": rn["lo"], "hi_new": rn["hi"]})
    result = {"docs_signed": n_docs, "candidate_pairs": n_cand, "verified_pairs": len(d1),
              "oversized_buckets_skipped": n_big, "docs_in_clusters": in_cl, "clusters": n_cl,
              "docs_in_new_clusters": in_new, "new_clusters": n_new, "by_group": group_rows,
              "field": field, "jaccard_threshold": thr}
    ctx.write_json("summary.json", result)
    share = in_cl / n_docs if n_docs else 0.0
    ctx.publish(f"Near-duplicate summaries ({field})", [
        {"type": "kpis", "items": [
            {"label": "documents signed", "value": n_docs, "fmt": "int"},
            {"label": "in near-dup clusters", "value": in_cl, "fmt": "int"},
            {"label": "share of corpus", "value": share, "fmt": "pct1",
             "status": "good" if share < 0.01 else ("warn" if share < 0.05 else "bad")},
            {"label": "clusters", "value": n_cl, "fmt": "int"},
            {"label": "new (not KU-Fixes duplicates)", "value": (in_new / in_cl) if in_cl else 0.0, "fmt": "pct1",
             "hint": "share of clustered docs whose cluster is not one KU-Fixes duplicate cluster"},
            {"label": "candidate pairs", "value": n_cand, "fmt": "int",
             "hint": f"verified at Jaccard >= {thr}: {len(d1)}; buckets > {max_bucket} skipped: {n_big}"}]},
        {"type": "bar", "title": "Share of documents in near-duplicate clusters, by group", "fmt": "pct1",
         "hint": "95% Wilson intervals", "labels": [r["group"] for r in group_rows],
         "series": [{"name": "all clusters", "data": [r["rate"] for r in group_rows],
                     "lo": [r["lo"] for r in group_rows], "hi": [r["hi"] for r in group_rows]},
                    {"name": "new clusters", "data": [r["rate_new"] for r in group_rows],
                     "lo": [r["lo_new"] for r in group_rows], "hi": [r["hi_new"] for r in group_rows]}]},
        {"type": "table", "title": "Largest clusters", "columns": [
            {"key": "cluster_id", "label": "cluster", "fmt": "int"}, {"key": "cluster_size", "label": "docs", "fmt": "int"},
            {"key": "n_papers", "label": "paper ids", "fmt": "int"}, {"key": "n_groups", "label": "groups", "fmt": "int"},
            {"key": "category", "label": "category"}], "rows": largest},
        paper_table("Papers in near-duplicate clusters", clustered,
                    [("cluster_id", "cluster", "int"), ("cluster_size", "docs in cluster", "int"),
                     ("category", "category", None)],
                    hint="'new' = not one KU-Fixes duplicate cluster (different papers with near-identical summaries)"),
    ], summary=f"{in_cl:,} of {n_docs:,} summaries ({share:.2%}) fall in {n_cl:,} near-duplicate clusters; "
               f"{n_new:,} clusters are not explained by KU-Fixes identifier duplicates.")
    return {k: v for k, v in result.items() if k != "by_group"}
