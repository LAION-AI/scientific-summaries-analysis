"""Stage `features` (map): one pass over the release -> one row per paper with every WS1 check and the metadata the
description needs. No text is stored; `quality` and `describe` aggregate these rows with DuckDB.

Reused, not recomputed: OpenAlex metadata (oa_language, oa_is_retracted, oa_is_open_access, oa_cited_by_count, ...),
the KU-Fixes fault index (title/author placeholders, needs_regeneration, claims JSON faults) and the KU-Fixes
duplicate clusters (paper-level duplicates by DOI / id / title+year). New here: per-field emptiness and placeholder
status, N/A statements, truncation, refusals and markup/key leaks, field lengths vs the prompt spec, summary language,
wrong-paper evidence (title / author / content agreement with the source), input type, and claim structure.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .. import links as links_mod
from .. import textutil as tu
from ..config import Config
from ..data import Shard, iter_batches
from ..domains import clean_label, domain_of, is_non_english_label, split_field
from ..runtime import Context, stage
from ..schema import CLAIM_KEYS, PROSE_FIELDS, SUMMARY_FIELDS, parse_claims

READ_COLUMNS = ["paper_id", "subset", "source_title", "source_authors", "source_year", "source_doi",
                *SUMMARY_FIELDS, "summary_model", "summary_tokens_in", "summary_tokens_out",
                "openalex_id", "oa_doi", "oa_title", "oa_type", "oa_year", "oa_cited_by_count", "oa_language",
                "oa_is_retracted", "oa_is_open_access", "oa_authorships", "match_method", "has_openalex_match",
                "text_length", "text_sanitized", "text_raw"]
_ARXIV_ID = re.compile(r"^(?:arXiv:)?(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


# ---------------------------------------------------------------- helpers (pure; unit-tested)
def arxiv_id(paper_id: str | None, *dois: str | None) -> str | None:
    m = _ARXIV_ID.match((paper_id or "").strip())
    if m:
        return m.group(1)
    for d in dois:
        m = _ARXIV_DOI.search(d or "")
        if m:
            return m.group(1)
    return None


def year_of(rec: dict, aid: str | None) -> tuple[int | None, str | None]:
    """Same priority as the SCI-JEPA split: source_year > oa_year > arXiv id."""
    for key in ("source_year", "oa_year"):
        v = rec.get(key)
        try:
            y = int(float(v))
            if 1600 <= y <= 2100:
                return y, key
        except (TypeError, ValueError):
            pass
    if aid and re.match(r"^\d{4}\.", aid):
        yy = int(aid[:2])
        return (2000 + yy if yy < 90 else 1900 + yy), "arxiv_id"
    if aid and "/" in aid:
        yy = int(aid.split("/")[1][:2])
        return (2000 + yy if yy < 90 else 1900 + yy), "arxiv_id"
    return None, None


def input_type(tokens_in: float | None, overhead: int, bins: dict[str, int]) -> tuple[str, int | None]:
    if tokens_in is None or tokens_in <= 0:
        return "unknown", None
    content = int(tokens_in) - overhead
    label = "metadata_only"
    for name, lo in sorted(bins.items(), key=lambda kv: kv[1]):
        if content >= lo:
            label = name
    return label, content


def openalex_names(raw: str | None) -> set[str]:
    try:
        items = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return set()
    names = []
    for a in items if isinstance(items, list) else []:
        if isinstance(a, dict):
            names.append((a.get("author") or {}).get("display_name") or a.get("raw_author_name") or "")
    return tu.name_tokens("; ".join(n for n in names if n))


def grobid_names(raw: str | None) -> set[str]:
    try:
        items = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return set()
    parts = []
    for p in items if isinstance(items, list) else []:
        if isinstance(p, dict):
            parts += [p.get("first") or "", *(p.get("middle") or []), p.get("last") or ""]
    return tu.name_tokens(" ; ".join(x for x in parts if isinstance(x, str)))


def jaccard(a: set, b: set) -> float | None:
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


@lru_cache(maxsize=1)
def _lang_model(path: str):
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None          # silence the load warning
    return fasttext.load_model(path)


def detect_language(text: str, model_path: str) -> tuple[str | None, float | None]:
    if not text.strip():
        return None, None
    labels, probs = _lang_model(model_path).predict(" ".join(text.split())[:2000], k=1)
    return labels[0].replace("__label__", ""), float(probs[0])


# ---------------------------------------------------------------- KU-Fixes joins
def _ku_lookup(cfg: Config, rel: str) -> tuple[dict, dict]:
    """KU-Fixes fault index and duplicate clusters of one shard; empty for datasets without KU-Fixes."""
    ku = cfg.site.ku_fixes
    if ku is None or not (ku / "ku_fix_index.parquet").exists() or not (ku / "duplicates.parquet").exists():
        return {}, {}
    idx = pq.read_table(cfg.site.ku_fixes / "ku_fix_index.parquet",
                        columns=["file", "row_index", "title_fault", "authors_fault", "needs_regeneration",
                                 "json_faults", "n_blank_summary_fields"],
                        filters=[("file", "=", rel)]).to_pylist()
    dup = pq.read_table(cfg.site.ku_fixes / "duplicates.parquet",
                        columns=["file", "row_index", "tier", "cluster_size", "keep", "keep_strong"],
                        filters=[("file", "=", rel)]).to_pylist()
    return {r["row_index"]: r for r in idx}, {r["row_index"]: r for r in dup}


# ---------------------------------------------------------------- one record
def record_features(r: dict, ku: dict | None, dup: dict | None, cfg_checks: dict, spec: dict,
                    lang_model: str | None, rules: list | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"file": r["_file"], "row": r["_row"], "group": r["_group"],
                           "paper_id": r.get("paper_id"), "subset": r.get("subset"),
                           "title": " ".join(str(r.get("oa_title") or r.get("summary_title") or
                                                 r.get("source_title") or "").split())[:200] or None,
                           "links": links_mod.to_json(links_mod.paper_links(r, rules or []))}
    # ----- identity, time, discipline (metadata reused from the release / OpenAlex)
    aid = arxiv_id(r.get("paper_id"), r.get("source_doi"), r.get("oa_doi"))
    year, year_src = year_of(r, aid)
    coarse, _ = split_field(r.get("field_subfield"))
    out.update(arxiv_id=aid, paper_id_is_path=(r.get("paper_id") or "").startswith("/"),
               year=year, year_source=year_src, field=coarse, field_clean=clean_label(coarse),
               field_non_english=is_non_english_label(coarse), domain=domain_of(r.get("field_subfield")),
               type_of_paper=(r.get("type_of_paper") or "").strip()[:80] or None,
               summary_model=r.get("summary_model"),
               tokens_in=r.get("summary_tokens_in"), tokens_out=r.get("summary_tokens_out"),
               has_openalex_match=r.get("has_openalex_match"), match_method=r.get("match_method"),
               oa_type=r.get("oa_type"), oa_language=r.get("oa_language"),
               oa_is_open_access=r.get("oa_is_open_access"), oa_is_retracted=r.get("oa_is_retracted"),
               oa_cited_by_count=r.get("oa_cited_by_count"), text_length=r.get("text_length"))

    # ----- input actually given to the summarizer
    src = r.get("text_raw") or r.get("text_sanitized") or ""
    out["has_text"] = bool(src.strip())
    itype, content_tokens = input_type(r.get("summary_tokens_in"), int(cfg_checks["prompt_overhead_tokens"]),
                                       cfg_checks["input_type_tokens"])
    out.update(input_type=itype, input_content_tokens=content_tokens,
               input_truncated=bool((r.get("text_length") or 0) >= int(cfg_checks["input_char_cap"])))
    cap = int(cfg_checks["output_token_cap"]) - int(cfg_checks["output_cap_margin"])
    out["output_capped"] = bool((r.get("summary_tokens_out") or 0) >= cap)

    # ----- per-field status, length, truncation, leaks
    n_bad = n_na = n_trunc = 0
    refusal_fields, markup_fields, leaked = [], [], set()
    for f in SUMMARY_FIELDS:
        if f == "claims":
            continue
        v = r.get(f)
        status = tu.value_status(v)
        out[f"status__{f}"] = status
        if f in PROSE_FIELDS:
            wc = tu.word_count(v)
            out[f"words__{f}"] = wc
            trunc = status == "ok" and tu.ends_mid_sentence(v)
            out[f"trunc__{f}"] = trunc
            n_trunc += trunc
            if f in spec and status == "ok":
                lo, hi = spec[f]
                out[f"spec__{f}"] = "short" if wc < lo else ("long" if wc > hi else "ok")
        n_bad += status in ("empty", "placeholder", "not_provided")
        n_na += status == "na_statement"
        if tu.refusal(v):
            refusal_fields.append(f)
        if f in PROSE_FIELDS and tu.markup_leak(v):
            markup_fields.append(f)
        leaked.update(tu.key_leak(v))
    out.update(n_fields_missing=n_bad, n_fields_na=n_na, n_fields_truncated=n_trunc,
               refusal=bool(refusal_fields), refusal_fields=refusal_fields or None,
               markup_leak=bool(markup_fields), markup_fields=markup_fields or None,
               key_leak=sorted(leaked) or None)

    # ----- claims structure
    issues: list[str] = []
    claims, _ = parse_claims(r.get("claims"), issues)
    raw_claims = r.get("claims") or ""
    out.update(n_claims=len(claims),
               claims_bad_json="claims:bad_json" in issues,
               claims_truncated_json=bool(raw_claims.strip()) and "claims:bad_json" in issues
               and not raw_claims.rstrip().endswith("]"),
               claims_off_schema=any(i.startswith("claims:off_schema_key") for i in issues),
               claims_missing_keys=any(i.startswith("claims:missing_keys") for i in issues),
               claims_empty_support=sum(1 for c in claims if not c["supporting_evidence"].strip()),
               claims_empty_implications=sum(1 for c in claims if not c["implications"].strip()))
    for k in CLAIM_KEYS:
        out[f"words__claims_{k}"] = sum(tu.word_count(c[k]) for c in claims)

    # ----- language of the summary (the paper's language is oa_language, reused)
    text_for_lang = " ".join((r.get(f) or "") for f in cfg_checks["lang_fields"])
    lang, prob = detect_language(text_for_lang, lang_model) if lang_model else (None, None)
    out.update(summary_lang=lang, summary_lang_prob=prob, summary_non_latin=tu.non_latin_fraction(text_for_lang),
               summary_non_english=bool(lang and lang != "en" and (prob or 0) >= cfg_checks["lang_min_prob"]))

    # ----- wrong-paper evidence: does the summary describe THIS paper?
    from rapidfuzz import fuzz
    st = (r.get("summary_title") or "").strip()
    refs = [("oa_title", r.get("oa_title")), ("source_title", r.get("source_title"))]
    sims = {k: fuzz.token_set_ratio(st.lower(), v.lower()) for k, v in refs if st and (v or "").strip()}
    out.update(title_sim_oa=sims.get("oa_title"), title_sim_source=sims.get("source_title"),
               title_sim=max(sims.values()) if sims else None)
    sa = tu.name_tokens(r.get("summary_authors"))
    ref_names = openalex_names(r.get("oa_authorships")) or grobid_names(r.get("source_authors"))
    out["author_overlap"] = jaccard(sa, ref_names)
    if src:
        src_vocab = set(tu.tokens(src[:200000]))
        cw = tu.content_words(r.get("executive_summary"))
        out["content_support"] = (len(cw & src_vocab) / len(cw)) if cw else None
    else:
        out["content_support"] = None
    # unreadable source (PDF fonts extracted without a Unicode map): any fluent summary of it is largely invented
    garbled = tu.garbled_fraction(src)
    out["source_garbled_frac"] = garbled
    out["source_garbled"] = garbled is not None and garbled >= cfg_checks["max_source_garbled"]
    t_bad = out["title_sim"] is not None and out["title_sim"] < cfg_checks["min_title_similarity"]
    a_bad = out["author_overlap"] is not None and out["author_overlap"] < cfg_checks["min_author_overlap"]
    c_bad = (out["content_support"] is not None and out["content_support"] < cfg_checks["min_content_support"]
             and not out["source_garbled"])
    out.update(title_mismatch=t_bad, author_mismatch=a_bad, content_mismatch=c_bad,
               # strongest evidence: the summary's content is not in a readable source text, or both title and
               # authors disagree with the reference metadata
               wrong_paper_suspect=bool(c_bad or (t_bad and a_bad)))

    # ----- KU-Fixes (reused)
    out.update(ku_title_fault=(ku or {}).get("title_fault"), ku_authors_fault=(ku or {}).get("authors_fault"),
               ku_needs_regeneration=bool((ku or {}).get("needs_regeneration")),
               ku_json_faults=(ku or {}).get("json_faults"),
               dup_tier=(dup or {}).get("tier"), dup_cluster_size=(dup or {}).get("cluster_size"),
               dup_drop_strong=(dup is not None and dup.get("tier") == "strong" and not dup.get("keep_strong")),
               dup_drop_any=(dup is not None and not dup.get("keep")))
    return out


_LIST_COLS = {"refusal_fields", "markup_fields", "key_leak", "ku_json_faults"}
_FLOAT_COLS = {"title_sim", "title_sim_oa", "title_sim_source", "author_overlap", "content_support", "source_garbled_frac",
               "summary_lang_prob", "summary_non_latin", "tokens_in", "tokens_out", "oa_cited_by_count"}
_INT_COLS = {"year", "input_content_tokens", "text_length", "dup_cluster_size", "row"}
_BOOL_COLS = {"has_openalex_match", "oa_is_open_access", "oa_is_retracted"}


def to_table(rows: list[dict]) -> pa.Table:
    """Rows -> table with a stable schema: every shard gets the same columns and types (all-null columns too)."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if k in _LIST_COLS:
            typ = pa.list_(pa.string())
        elif k in _FLOAT_COLS:
            typ = pa.float64()
        elif k in _INT_COLS or k.startswith(("words__", "n_", "claims_empty")):
            typ = pa.int64()
        elif k in _BOOL_COLS or k.startswith("trunc__") or isinstance(next((v for v in vals if v is not None), ""), bool):
            typ = pa.bool_()
        else:
            typ = pa.string()
        cols[k] = pa.array(vals, type=typ)
    return pa.table(cols)


def process_shard(cfg: Config, shard: Shard) -> pa.Table:
    checks = cfg.raw["ws1"]["checks"]
    spec = {k: tuple(v) for k, v in (cfg.raw["ws1"].get("length_spec_words") or {}).items() if v}
    lang_model = checks.get("lang_model")
    if lang_model and not Path(lang_model).exists():
        raise FileNotFoundError(f"fastText model missing: {lang_model} (see README: assets)")
    ku, dup = _ku_lookup(cfg, shard.rel)
    rules = links_mod.compile_rules(cfg.dataset.links)
    rows = []
    columns = READ_COLUMNS + [c for c in dict.fromkeys(r[1] for r in rules) if c not in READ_COLUMNS]
    for tbl in iter_batches(cfg, shard, columns, batch_size=512):
        for r in tbl.to_pylist():
            feats = record_features(r, ku.get(r["_row"]), dup.get(r["_row"]), checks, spec, lang_model, rules)
            for f in spec:                                   # same columns in every row
                feats.setdefault(f"spec__{f}", None)
            rows.append(feats)
    return to_table(rows)


@stage("ws1", "features", kind="map", needs=(), resources="cpu",
       config_keys=("ws1.checks", "ws1.length_spec_words", "dataset"),
       help="per-paper quality checks + metadata (one pass over the release)")
def run(ctx: Context) -> dict:
    return ctx.map_shards(process_shard)
