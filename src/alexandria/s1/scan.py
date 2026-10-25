"""Stages `pii`, `injection` and `retractions` (S1, CPU).

pii          PII screen of the summaries (each of the 18 KU fields) and of the full text, on the S1 record sample.
             In full text, author contact data (emails, affiliation addresses, social handles) is expected; in a
             summary every hit is a finding. Rates with Wilson 95% intervals per field / subset; a precision
             spot-check packet of (record, field, category) only — never the matched text.
injection    Prompt-injection / instruction-like text that could hijack an agent reading the KUs or full texts
             through the Alexandria skill (override instructions, role / chat-template tags, tool-call JSON,
             invisible or bidi control characters, Unicode tag characters), per source and category.
retractions  Retracted papers (OpenAlex `oa_is_retracted`) in the whole local release (light columns), and how many
             are in the WS2 / WS4 / WS5 paper sets and the WS3 retrieval index.
"""
from __future__ import annotations

import pandas as pd

from ..data import iter_batches, release_shards
from ..runtime import Context, log, stage
from ..schema import SUMMARY_FIELDS
from ..stats import rate_row
from ..ws1.faith.common import paper_key, source_text
from .common import injection_hits, pii_hits, sample_records


@stage("s1", "pii", config_keys=("s1.sample", "s1.pii"),
       help="PII screen of summaries (18 KU fields) and full text; expected author contacts vs PII in summaries")
def run_pii(ctx: Context) -> dict:
    cfg = ctx.section("s1.pii")
    df = sample_records(ctx, [*SUMMARY_FIELDS, "text_raw", "text_sanitized"])
    expected = set(cfg.get("expected_in_fulltext", ["email", "postal_address", "url_social"]))
    rows = []
    for r in df.to_dict("records"):
        for f in SUMMARY_FIELDS:
            for cat, k in pii_hits(r.get(f)).items():
                rows.append({"paper_key": r["paper_key"], "_group": r["_group"], "source": "summary", "field": f,
                             "category": cat, "matches": k, "expected": False})
        for cat, k in pii_hits(source_text(r)).items():
            rows.append({"paper_key": r["paper_key"], "_group": r["_group"], "source": "fulltext", "field": "text",
                         "category": cat, "matches": k, "expected": cat in expected})
    flags = pd.DataFrame(rows, columns=["paper_key", "_group", "source", "field", "category", "matches", "expected"])
    flags.to_parquet(ctx.out / "pii_flags.parquet", index=False)
    n = len(df)
    has_text = int(sum(bool(source_text(r).strip()) for r in df.to_dict("records")))
    summ = flags[flags["source"] == "summary"]
    unexp_ft = flags[(flags["source"] == "fulltext") & ~flags["expected"]]
    by_cat = []
    for (src, cat), g in flags.groupby(["source", "category"]):
        by_cat.append({"source": src, "category": cat, **rate_row(g["paper_key"].nunique(), n if src == "summary" else has_text),
                       "expected": bool(g["expected"].iloc[0])})
    by_field = [{"field": f, **rate_row(g["paper_key"].nunique(), n)} for f, g in summ.groupby("field")]
    spot = flags[~flags["expected"]].sample(n=min(int(cfg.get("spot_check", 50)), int((~flags["expected"]).sum())),
                                            random_state=0)[["paper_key", "source", "field", "category"]]
    spot.assign(true_pii="", notes="").to_csv(ctx.out / "pii_spot_check.csv", index=False)
    s_rate = rate_row(summ["paper_key"].nunique(), n)
    ctx.write_json("pii.json", {"records": n, "with_fulltext": has_text, "summary": s_rate,
                                "fulltext_unexpected": rate_row(unexp_ft["paper_key"].nunique(), has_text)})
    ctx.publish("S1 personal data (PII) screen", [
        {"type": "kpis", "items": [
            {"label": "records screened", "value": n, "fmt": "int"},
            {"label": "summaries with PII", "value": s_rate["rate"], "fmt": "pct1", "status": "warn" if s_rate["k"] else "good",
             "hint": f"{s_rate['k']} of {n}; Wilson 95% CI {s_rate['lo']:.1%}-{s_rate['hi']:.1%}"},
            {"label": "full texts with unexpected PII", "value": rate_row(unexp_ft["paper_key"].nunique(), has_text)["rate"],
             "fmt": "pct1", "hint": "beyond author contacts (phone, IDs, dates of birth, patient ids, IPs)"},
            {"label": "spot-check packet", "value": len(spot), "fmt": "int", "hint": "pii_spot_check.csv (ids only)"}]},
        {"type": "table", "title": "PII by source and category (records with >= 1 match)", "rows": by_cat, "columns": [
            {"key": "source", "label": "source"}, {"key": "category", "label": "category"}, {"key": "k", "label": "records", "fmt": "int"},
            {"key": "rate", "label": "rate", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"},
            {"key": "expected", "label": "expected in full text"}]},
        {"type": "table", "title": "Summaries: PII by KU field", "rows": by_field, "columns": [
            {"key": "field", "label": "field"}, {"key": "k", "label": "records", "fmt": "int"}, {"key": "rate", "label": "rate", "fmt": "pct1"},
            {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"}]},
    ], summary=f"PII in {s_rate['k']} of {n} summaries ({s_rate['rate']:.1%})")
    return {"records": n, "summary_pii": s_rate["k"]}


@stage("s1", "injection", config_keys=("s1.sample", "s1.injection"),
       help="prompt-injection / instruction-like text in KUs and full texts (could hijack agents using the skill)")
def run_injection(ctx: Context) -> dict:
    cfg = ctx.section("s1.injection")
    df = sample_records(ctx, [*SUMMARY_FIELDS, "text_raw", "text_sanitized"])
    rows = []
    for r in df.to_dict("records"):
        for src, txt in (("summary", "\n".join(str(r.get(f) or "") for f in SUMMARY_FIELDS)), ("fulltext", source_text(r))):
            for cat, k in injection_hits(txt).items():
                rows.append({"paper_key": r["paper_key"], "_group": r["_group"], "source": src, "category": cat, "matches": k})
    flags = pd.DataFrame(rows, columns=["paper_key", "_group", "source", "category", "matches"])
    flags.to_parquet(ctx.out / "injection_flags.parquet", index=False)
    n = len(df)
    tbl = [{"source": s, "category": c, **rate_row(g["paper_key"].nunique(), n)} for (s, c), g in flags.groupby(["source", "category"])]
    hard = flags[~flags["category"].isin(["llm_boilerplate"])]
    s_rate = rate_row(hard.loc[hard["source"] == "summary", "paper_key"].nunique(), n)
    f_rate = rate_row(hard.loc[hard["source"] == "fulltext", "paper_key"].nunique(), n)
    spot = flags.sample(n=min(int(cfg.get("spot_check", 50)), len(flags)), random_state=0)[["paper_key", "source", "category"]]
    spot.assign(real_injection="", notes="").to_csv(ctx.out / "injection_spot_check.csv", index=False)
    ctx.write_json("injection.json", {"records": n, "summary": s_rate, "fulltext": f_rate})
    ctx.publish("S1 prompt-injection screen (static)", [
        {"type": "kpis", "items": [
            {"label": "records screened", "value": n, "fmt": "int"},
            {"label": "summaries with injection-like text", "value": s_rate["rate"], "fmt": "pct1", "hint": f"{s_rate['k']} records"},
            {"label": "full texts with injection-like text", "value": f_rate["rate"], "fmt": "pct1", "hint": f"{f_rate['k']} records"}]},
        {"type": "table", "title": "Injection-like patterns by source and category", "rows": tbl,
         "hint": "llm_boilerplate ('as an AI language model') is reported but is not an injection",
         "columns": [{"key": "source", "label": "source"}, {"key": "category", "label": "category"},
                     {"key": "k", "label": "records", "fmt": "int"}, {"key": "rate", "label": "rate", "fmt": "pct1"},
                     {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"}]},
    ], summary=f"injection-like text in {s_rate['k']} summaries / {f_rate['k']} full texts of {n}")
    return {"records": n, "summary_hits": s_rate["k"], "fulltext_hits": f_rate["k"]}


def _retracted(cfg, sh) -> pd.DataFrame:
    rows = []
    for tbl in iter_batches(cfg, sh, ["oa_is_retracted"], batch_size=16384):
        d = tbl.to_pandas()
        rows.append(pd.DataFrame({"_file": d["_file"], "_row": d["_row"], "_group": d["_group"],
                                  "retracted": d["oa_is_retracted"].fillna(False).astype(bool)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@stage("s1", "retractions", config_keys=("s1.retractions",),
       help="retracted papers in the whole local release and in the WS2 / WS4 / WS5 sets and the WS3 index")
def run_retractions(ctx: Context) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    shards = release_shards(ctx.cfg)
    log(f"retractions: {len(shards)} release shards")
    with ThreadPoolExecutor(16) as ex:
        df = pd.concat(list(ex.map(lambda s: _retracted(ctx.cfg, s), shards)), ignore_index=True)
    ret = df[df["retracted"]].copy()
    ret["paper_key"] = [paper_key(f, r) for f, r in zip(ret["_file"], ret["_row"], strict=True)]
    ret[["_file", "_row", "_group", "paper_key"]].to_parquet(ctx.out / "retracted.parquet", index=False)
    keys = set(ret["paper_key"])
    sets = {"WS2 eval papers": ("ws2", "papers", "papers.parquet", "paper_key"),
            "WS3 retrieval index": ("ws3", "index", "units.parquet", "paper_key"),
            "WS4 training papers": ("ws4", "papers", "papers.parquet", "paper_key"),
            "WS5 papers": ("ws5", "papers", "papers.parquet", "paper_key")}
    usage = []
    for name, (ws, st, fn, col) in sets.items():
        p = ctx.cfg.stage_dir(ws, st) / fn
        if p.exists():
            s = set(pd.read_parquet(p, columns=[col])[col].dropna())
            usage.append({"set": name, "papers": len(s), "retracted": len(s & keys)})
    by = df.groupby("_group")["retracted"].agg(["sum", "size"]).reset_index().rename(columns={"_group": "subset"})
    tbl = [{"subset": r["subset"], **rate_row(int(r["sum"]), int(r["size"]))} for r in by.to_dict("records")]
    total = rate_row(len(ret), len(df))
    ctx.write_json("retractions.json", {"total": total, "usage": usage})
    ctx.publish("S1 retracted papers", [
        {"type": "kpis", "items": [
            {"label": "records scanned", "value": len(df), "fmt": "int", "hint": "whole local release (arXiv + PubMed)"},
            {"label": "retracted (OpenAlex)", "value": len(ret), "fmt": "int", "hint": f"{total['rate']:.3%} of records"},
            {"label": "retracted in project sets", "value": int(sum(u["retracted"] for u in usage)), "fmt": "int"}]},
        {"type": "table", "title": "Retracted share per subset", "rows": tbl, "columns": [
            {"key": "subset", "label": "subset"}, {"key": "k", "label": "retracted", "fmt": "int"}, {"key": "n", "label": "records", "fmt": "int"},
            {"key": "rate", "label": "rate", "fmt": "pct1"}]},
        {"type": "table", "title": "Retracted papers in the project's paper sets", "rows": usage, "columns": [
            {"key": "set", "label": "set"}, {"key": "papers", "label": "papers", "fmt": "int"}, {"key": "retracted", "label": "retracted", "fmt": "int"}]},
    ], summary=f"{len(ret)} retracted records ({total['rate']:.3%})")
    return {"retracted": len(ret), "records": len(df)}
