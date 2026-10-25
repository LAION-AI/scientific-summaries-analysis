"""Input data: shard discovery, type-unified streaming reads, and the small debug sample.

Every batch read through `iter_batches` carries three provenance columns, so any result row can be joined back to the
release and to the KU-Fixes index:
    _file   repo-relative path of the original shard, e.g. "data/arxiv/new_00300.parquet"
    _row    row index inside that original shard
    _group  reporting group from the config, e.g. "arxiv/new"

Every dataset is read into ONE canonical schema (resources/canonical_schema.json: the 44 columns of the
Scientific-Summaries release, as unified by KU-Fixes). A dataset whose columns are named differently maps them in
`dataset.columns` (canonical name -> its name); canonical columns a dataset does not have become typed nulls, and the
checks that need them report "n/a". Shards disagree on types (null-typed OpenAlex columns, token counts as double or
int64, numbers stored as strings), so every batch is cast to the canonical types. Columns outside the canonical
schema keep their own type.
"""
from __future__ import annotations

import glob
import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import Config
from .runtime import atomic_write_json, log

_TYPES = {"large_string": pa.large_string(), "string": pa.string(), "int64": pa.int64(),
          "double": pa.float64(), "bool": pa.bool_()}
PROVENANCE = ["_file", "_row", "_group"]


@dataclass(frozen=True)
class Shard:
    group: str
    rel: str          # repo-relative path of the ORIGINAL release shard (join key)
    path: Path        # file actually read (the release shard, or its debug sample copy)
    size: int


CANONICAL_SCHEMA = Path(__file__).parent / "resources" / "canonical_schema.json"


@lru_cache(maxsize=4)
def _unified(path: str) -> pa.Schema:
    cols = json.loads(Path(path).read_text())["columns"]
    return pa.schema([(c["name"], _TYPES[c["type"]]) for c in cols])


def unified_schema(cfg: Config | None = None) -> pa.Schema:
    """The canonical schema every dataset is read into."""
    return _unified(str(CANONICAL_SCHEMA))


def source_name(cfg: Config, canonical: str, available: set[str]) -> str | None:
    """The column that holds `canonical` in a shard: the canonical name itself (debug samples and datasets that use
    it), else the dataset's mapped name, else None (column absent)."""
    if canonical in available:
        return canonical
    mapped = cfg.dataset.columns.get(canonical)
    return mapped if mapped in available else None


def cast_batch(batch: pa.RecordBatch | pa.Table, schema: pa.Schema, columns: list[str] | None = None) -> pa.Table:
    """Cast to the unified types; columns missing from the shard become typed nulls."""
    tbl = pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch) else batch
    names = columns or schema.names
    arrays = []
    for name in names:
        if name in PROVENANCE:
            arrays.append(tbl[name])
            continue
        if name not in schema.names:                     # not canonical: keep the dataset's own column as it is
            arrays.append(tbl[name] if name in tbl.column_names else pa.nulls(tbl.num_rows, pa.string()))
            continue
        field = schema.field(name)
        if name not in tbl.column_names:
            arrays.append(pa.nulls(tbl.num_rows, field.type))
            continue
        col = tbl[name]
        if pa.types.is_floating(col.type) and pa.types.is_integer(field.type):
            col = pc.round(col)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            if pa.types.is_boolean(field.type):
                col = pc.equal(pc.utf8_lower(col), "true")
            elif not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
                col = pc.if_else(pc.equal(col, ""), pa.scalar(None, col.type), col)
                if pa.types.is_integer(field.type):
                    col = pc.round(col.cast(pa.float64()))
        arrays.append(col.cast(field.type))
    return pa.Table.from_arrays(arrays, names=names)


def release_shards(cfg: Config) -> list[Shard]:
    out = []
    for src in cfg.data.sources:
        for p in sorted(glob.glob(str(cfg.site.data_root / src.glob))):
            out.append(Shard(src.group, os.path.relpath(p, cfg.site.data_root), Path(p), os.path.getsize(p)))
    return out


def sample_dir(cfg: Config) -> Path:
    return cfg.run_root("debug") / "_sample"


def input_shards(cfg: Config) -> list[Shard]:
    """The shards a stage processes: the whole release (full) or the stored debug sample (debug)."""
    if cfg.profile == "full":
        return release_shards(cfg)
    manifest = sample_dir(cfg) / "sample.json"
    if not manifest.exists():
        raise FileNotFoundError(f"no debug sample at {manifest}; run `alexandria sample` first")
    items = json.loads(manifest.read_text())["shards"]
    return [Shard(i["group"], i["rel"], sample_dir(cfg) / i["file"], i["size"]) for i in items]


def partition(shards: list[Shard], index: int, count: int) -> list[Shard]:
    """Deterministic size-balanced split of the shards over `count` workers (longest-processing-time first)."""
    if count <= 1:
        return list(shards)
    loads = [0] * count
    owner: dict[str, int] = {}
    for s in sorted(shards, key=lambda s: (-s.size, s.rel)):
        w = min(range(count), key=lambda i: (loads[i], i))
        loads[w] += s.size
        owner[s.rel] = w
    return [s for s in shards if owner[s.rel] == index]


OPTOUT_KEYS = ("paper_id", "source_doi", "oa_doi")


@lru_cache(maxsize=4)
def optout_ids(path: str | None) -> frozenset[str]:
    """Normalised ids of the opt-out / takedown list (docs/TAKEDOWN.md): paper ids or DOIs, one per line, # comments."""
    if not path or not os.path.exists(path):
        return frozenset()
    out = set()
    for line in open(path, encoding="utf-8"):
        v = line.split("#", 1)[0].strip().lower()
        if v:
            out.add(v.removeprefix("https://doi.org/").removeprefix("doi:"))
    return frozenset(out)


def optout_mask(tbl: pa.Table, ids: frozenset[str]) -> pa.Array:
    """True for rows to KEEP (none of paper_id / source_doi / oa_doi is on the opt-out list)."""
    keep = [True] * tbl.num_rows
    for c in OPTOUT_KEYS:
        if c in tbl.column_names:
            for i, v in enumerate(tbl[c].to_pylist()):
                if v and str(v).strip().lower().removeprefix("https://doi.org/") in ids:
                    keep[i] = False
    return pa.array(keep)


def iter_batches(cfg: Config, shard: Shard, columns: list[str], batch_size: int = 1024) -> Iterator[pa.Table]:
    """Stream one shard as unified-type tables with the provenance columns attached. Rows on the opt-out list
    (site.optout_file) are dropped; provenance (_file, _row) keeps the release row numbers."""
    schema = unified_schema(cfg)
    pf = pq.ParquetFile(shard.path)
    names = set(pf.schema_arrow.names)
    has_prov = "_file" in names
    optout = optout_ids(str(cfg.site.optout_file) if cfg.site.optout_file else None)
    extra = [c for c in OPTOUT_KEYS if optout and c not in columns]
    src = {c: source_name(cfg, c, names) for c in [*columns, *extra] if c not in PROVENANCE}
    read_cols = sorted({s for s in src.values() if s}) + ([c for c in PROVENANCE if c in names] if has_prov else [])
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        raw = pa.Table.from_batches([batch])
        renamed = pa.table({"_n": pa.nulls(raw.num_rows)} |                # keeps the row count if nothing matches
                           {c: raw[s] for c, s in src.items() if s})        # dataset names -> canonical names
        tbl = cast_batch(renamed, schema, list(src))
        n = tbl.num_rows
        if has_prov:
            for c in PROVENANCE:
                tbl = tbl.append_column(c, raw[c])
        else:
            tbl = tbl.append_column("_file", pa.array([shard.rel] * n, pa.string()))
            tbl = tbl.append_column("_row", pa.array(range(offset, offset + n), pa.int64()))
            tbl = tbl.append_column("_group", pa.array([shard.group] * n, pa.string()))
        offset += n
        if optout:
            tbl = tbl.filter(optout_mask(tbl, optout)).drop_columns(extra)
        yield tbl


def fetch_records(cfg: Config, keys: list[tuple[str, int]], columns: list[str]) -> dict[tuple[str, int], dict]:
    """Read specific records by (release file, row) -> {key: record}. For showing flagged examples; reads only the
    shards involved (from the debug sample when the profile is debug)."""
    by_file: dict[str, set[int]] = {}
    for f, r in keys:
        by_file.setdefault(f, set()).add(int(r))
    shards = {s.rel: s for s in input_shards(cfg)}
    out: dict[tuple[str, int], dict] = {}
    for rel, rows in by_file.items():
        sh = shards.get(rel)
        if sh is None:
            continue
        for tbl in iter_batches(cfg, sh, columns, batch_size=2048):
            hit = [i for i, r in enumerate(tbl["_row"].to_pylist()) if r in rows]
            for rec in tbl.take(pa.array(hit, pa.int64())).to_pylist() if hit else []:
                out[(rel, rec["_row"])] = rec
    return out


def fetch_release_records(cfg: Config, keys: list[tuple[str, int]], columns: list[str]) -> dict[tuple[str, int], dict]:
    """Like fetch_records, but reads the release shards themselves (any profile): for rows outside the debug sample,
    e.g. benchmark gold papers that WS3 adds to its corpus."""
    by_file: dict[str, set[int]] = {}
    for f, r in keys:
        by_file.setdefault(f, set()).add(int(r))
    shards = {s.rel: s for s in release_shards(cfg)}
    out: dict[tuple[str, int], dict] = {}
    for rel, rows in by_file.items():
        sh = shards.get(rel)
        if sh is None:
            continue
        for tbl in iter_batches(cfg, sh, columns, batch_size=2048):
            hit = [i for i, r in enumerate(tbl["_row"].to_pylist()) if r in rows]
            for rec in tbl.take(pa.array(hit, pa.int64())).to_pylist() if hit else []:
                out[(rel, rec["_row"])] = rec
    return out


def make_sample(cfg: Config, files_per_group: int, rows_per_file: int, seed: int = 0) -> Path:
    """Copy a tiny, reproducible, stratified subset of the release (all columns) into the debug sample folder."""
    out = sample_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_group: dict[str, list[Shard]] = {}
    for s in release_shards(cfg):
        by_group.setdefault(s.group, []).append(s)
    schema = unified_schema(cfg)
    items = []
    for group, shards in sorted(by_group.items()):
        # arxiv/titled shards can hold a single row: prefer shards with enough rows
        rows = {s.rel: pq.ParquetFile(s.path).metadata.num_rows for s in shards}
        big = [s for s in shards if rows[s.rel] >= rows_per_file] or shards
        for s in sorted(rng.sample(big, min(files_per_group, len(big))), key=lambda s: s.rel):
            n = rows[s.rel]
            idx = sorted(rng.sample(range(n), min(rows_per_file, n)))
            full = pq.read_table(s.path)
            names = set(full.column_names)
            canon = {c: source_name(cfg, c, names) for c in schema.names}
            tbl = cast_batch(pa.table({c: full[x] for c, x in canon.items() if x}), schema).take(pa.array(idx))
            tbl = tbl.append_column("_file", pa.array([s.rel] * len(idx), pa.string()))
            tbl = tbl.append_column("_row", pa.array(idx, pa.int64()))
            tbl = tbl.append_column("_group", pa.array([group] * len(idx), pa.string()))
            name = s.rel.replace("/", "__")
            pq.write_table(tbl, out / name)
            items.append({"group": group, "rel": s.rel, "file": name, "rows": len(idx),
                          "size": os.path.getsize(out / name)})
            log(f"sample: {s.rel} -> {len(idx)} rows")
    atomic_write_json(out / "sample.json", {"seed": seed, "files_per_group": files_per_group,
                                            "rows_per_file": rows_per_file, "revision": cfg.data.revision,
                                            "shards": items})
    return out
