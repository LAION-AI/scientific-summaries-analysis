"""Helpers shared by the WS1b stages: stable keys, sentence splitting, deterministic sampling, record fetching."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Any

from ...config import Config
from ...data import input_shards, iter_batches
from ...runtime import Context

LABELS = ("supported", "unsupported", "contradicted")


def paper_key(file: str, row: int) -> str:
    """Stable short id of one release row (file + row index), used in every WS1b item id."""
    return hashlib.sha1(f"{file}:{int(row)}".encode()).hexdigest()[:12]


def item_id(pkey: str, field: str, idx: int | None = None) -> str:
    return f"{pkey}:{field}" if idx is None else f"{pkey}:{field}:{idx}"


def stable_rank(seed: int, *parts: Any) -> str:
    """Deterministic pseudo-random order key (same seed + same paper -> same position, on any machine)."""
    return hashlib.sha1(("|".join(map(str, (seed, *parts)))).encode()).hexdigest()


# ---------------------------------------------------------------- sentences
_ABBREV = re.compile(r"\b(?:et al|e\.g|i\.e|cf|vs|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|Tab|approx|resp|Dr|Mr|Ms|no|No|ca)\.$")
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str | None) -> list[str]:
    """Regex sentence splitter (no nltk): splits after . ! ? followed by whitespace and an upper-case / digit start,
    except after common abbreviations (et al., e.g., Fig., Eq., ...). Good enough for LLM-written prose."""
    out: list[str] = []
    for block in (text or "").split("\n"):
        block = block.strip()
        if not block:
            continue
        pieces = _BOUNDARY.split(block)
        buf = ""
        for p in pieces:
            buf = f"{buf} {p}".strip() if buf else p.strip()
            if not _ABBREV.search(buf):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    return [s for s in out if len(s) > 1]


# ---------------------------------------------------------------- sampling
def allocate(counts: dict[Any, int], n: int, min_per: int = 1) -> dict[Any, int]:
    """Proportional allocation of n draws over strata with a floor of min_per per non-empty stratum (capped by the
    stratum size), remaining draws by largest remainder. Never exceeds a stratum's size or n (unless the floors
    alone exceed n, then every non-empty stratum still gets min(min_per, size))."""
    counts = {k: int(v) for k, v in counts.items() if v > 0}
    alloc = {k: min(min_per, v) for k, v in counts.items()}
    left = n - sum(alloc.values())
    if left <= 0:
        return alloc
    room = {k: counts[k] - alloc[k] for k in counts}
    total_room = sum(room.values())
    if total_room <= 0:
        return alloc
    left = min(left, total_room)
    while left > 0:
        cap = {k: r for k, r in room.items() if r > 0}
        if not cap:
            break
        tot = sum(counts[k] for k in cap)
        quota = {k: left * counts[k] / tot for k in cap}
        add = {k: min(cap[k], math.floor(q)) for k, q in quota.items()}
        given = sum(add.values())
        if given < left:                                  # largest remainders get the rest
            for k in sorted(cap, key=lambda k: (-(quota[k] - math.floor(quota[k])), str(k))):
                if given >= left:
                    break
                if add[k] < cap[k]:
                    add[k] += 1
                    given += 1
        if given == 0:
            break
        for k, a in add.items():
            alloc[k] += a
            room[k] -= a
        left -= given
    return alloc


def features_table(ctx: Context, columns: list[str] | None = None):
    """All WS1 `features` parts as one pandas frame (DuckDB, only the requested columns); usable from any WS."""
    import duckdb
    parts = ctx.parts("features", "ws1")
    if not parts:
        raise FileNotFoundError("stage `features` has no parts; run it first")
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    files = [str(p) for p in parts]
    return duckdb.sql(f"select {cols} from read_parquet({files!r}, union_by_name=true)").df()


def stratified_pick(df, n: int, strata: list[str], seed: int, min_per: int = 1, debug_all: bool = False):
    """Deterministic stratified sample of rows of df (needs file, row columns). debug_all -> every row."""
    if df.empty:
        return df
    df = df.copy()
    df["_rank"] = [stable_rank(seed, f, r) for f, r in zip(df["file"], df["row"], strict=True)]
    df = df.sort_values("_rank")
    if debug_all or n >= len(df):
        return df.drop(columns="_rank")
    keys = [c for c in strata if c in df.columns]
    if not keys:
        return df.head(n).drop(columns="_rank")
    df[keys] = df[keys].fillna("NA").astype(str)
    def tup(k):                                       # pandas returns scalars or tuples depending on len(keys)
        return k if isinstance(k, tuple) else (k,)
    groups = [(tup(k), g) for k, g in df.groupby(keys, sort=True)]
    alloc = allocate({k: len(g) for k, g in groups}, n, min_per)
    picked = [g.head(alloc.get(k, 0)) for k, g in groups]
    import pandas as pd
    return pd.concat(picked).sort_values("_rank").drop(columns="_rank")


# ---------------------------------------------------------------- records
def fetch_records(cfg: Config, keys: Iterable[tuple[str, int]], columns: list[str]) -> dict[tuple[str, int], dict]:
    """Read the listed (file, row) records (release or debug sample) with the given columns."""
    wanted: dict[str, set[int]] = {}
    for f, r in keys:
        wanted.setdefault(f, set()).add(int(r))
    shards = {s.rel: s for s in input_shards(cfg)}
    out: dict[tuple[str, int], dict] = {}
    for f, rows in wanted.items():
        sh = shards.get(f)
        if sh is None:
            continue
        for tbl in iter_batches(cfg, sh, columns, batch_size=1024):
            for rec in tbl.to_pylist():
                if rec["_row"] in rows:
                    out[(f, rec["_row"])] = rec
            if len([k for k in out if k[0] == f]) == len(rows):
                break
    return out


HUMAN_OFF = ("Human labelling is switched off (`ws1b.human.enabled: false` in configs/ws1.yaml). "
             "Turn it on with `--set ws1b.human.enabled=true` once annotators are available.")


def human_enabled(ctx: Context) -> bool:
    return bool(ctx.section("ws1b.human").get("enabled", True))


def source_text(rec: dict) -> str:
    return rec.get("text_raw") or rec.get("text_sanitized") or ""


def field_items(rec: dict, field: str) -> list[str]:
    """Annotation / checking items of one field: the field text, or one item per claim description."""
    if field == "claims":
        from ...schema import parse_claims
        claims, _ = parse_claims(rec.get("claims"), [])
        return [c["description"] for c in claims]
    text = (rec.get(field) or "").strip()
    return [text] if text else []
