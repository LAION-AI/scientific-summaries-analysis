"""Stage `evidence` (C1, CPU): every measurement the compliance matrix needs, as one flat {key: value} map.

Most keys are read from earlier outputs (S1 PII / guard / dual-use / injection / uplift, WS1 quality / licences /
provenance / datasheet, X6). Three are measured here on the records (debug: the stored sample; full: up to
`c1.max_records`):
  * GDPR Art. 9 proxy: direct patient identifiers in KUs of biomedical papers (medical record / hospital numbers,
    titled patient names, exact dates of birth / admission) and quasi-identifiers (case reports with exact age + sex);
  * GDPR Art. 3 scope: share of papers with at least one author affiliated in the EU / EEA (OpenAlex authorships);
  * representativeness (AI Act Art. 10(2)(f), HLEG Req. 5): language, author-country and discipline coverage.
A key that cannot be measured is left out (the matrix then says "unknown"), never guessed.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import REPO_ROOT
from ..data import input_shards, iter_batches
from ..runtime import Context, log, stage
from ..stats import rate_row

PROJECT_ROOT = Path(REPO_ROOT)
KU_TEXT = ("executive_summary", "methodological_details", "procedures_architectures", "key_results",
           "ethical_considerations", "contradictions_limitations")

DIRECT_ID = re.compile(
    r"\b(?:MRN|medical record (?:no\.?|number)|hospital (?:no\.?|number)|patient (?:id|no\.?|number))\s*[:#]?\s*\d{4,}"
    r"|\b(?:born|date of birth|DOB|admitted|admission)\s*(?:on|:)?\s*\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}", re.I)
TITLED_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]{2,}\b")    # case-sensitive: "MR" / "MS" are methods
QUASI_ID = re.compile(r"\b\d{1,3}[- ]year[- ]old\s+(?:man|woman|male|female|boy|girl|patient|infant)\b", re.I)
SIGNED = re.compile(r"(?m)^Approved-by:[ \t]*\S")
SIGNED_DOCS = {"dpia", "privacy_notice", "copyright_policy", "training_data_summary"}   # need DPO / legal sign-off
CASE_REPORT = re.compile(r"\bcase (?:report|study|presentation)\b|\bwe (?:report|present|describe) (?:a|the) case\b", re.I)


def _j(p: Path) -> dict:
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _countries(authorships: Any) -> set[str]:
    if not authorships:
        return set()
    try:
        a = json.loads(authorships) if isinstance(authorships, str) else authorships
    except json.JSONDecodeError:
        return set()
    return {c for au in a or [] for c in (au.get("countries") or []) if c}


def health_flags(rec: dict) -> dict:
    """Direct identifiers / quasi-identifiers / case report, from the KU text fields of one record."""
    text = " ".join(str(rec.get(f) or "") for f in KU_TEXT)
    return {"direct_id": bool(DIRECT_ID.search(text) or TITLED_NAME.search(text)), "quasi_id": bool(QUASI_ID.search(text)),
            "case_report": bool(CASE_REPORT.search(text))}


def scan_records(ctx: Context, max_records: int, eu: set[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cols = list(KU_TEXT) + ["oa_language", "oa_authorships", "summary_model", "text_raw"]
    n, lang, country, groups = 0, Counter(), Counter(), Counter()
    health, prov, eu_n, model_n, text_n = [], [], 0, 0, 0
    for sh in input_shards(ctx.cfg):
        for tbl in iter_batches(ctx.cfg, sh, cols, batch_size=512):
            for rec in tbl.to_pylist():
                n += 1
                groups[rec["_group"]] += 1
                lang[rec.get("oa_language") or "(unknown)"] += 1
                cs = _countries(rec.get("oa_authorships"))
                country.update(cs or {"(unknown)"})
                eu_n += bool(cs & eu)
                model_n += bool(rec.get("summary_model"))
                has_text = bool((rec.get("text_raw") or "").strip())               # full text shipped with the record
                text_n += has_text
                prov.append({"_file": rec["_file"], "_row": rec["_row"], "has_text": has_text})
                if rec["_group"].startswith("pubmed"):
                    health.append({"_file": rec["_file"], "_row": rec["_row"], **health_flags(rec)})
                if n >= max_records:
                    break
            if n >= max_records:
                break
        if n >= max_records:
            break
    hf = pd.DataFrame(health, columns=["_file", "_row", "direct_id", "quasi_id", "case_report"])
    known_lang = {k: v for k, v in lang.items() if k != "(unknown)"}
    known_c = {k: v for k, v in country.items() if k != "(unknown)"}
    with_country = n - country.get("(unknown)", 0)
    stats = {
        "records": n, "groups": dict(groups), "languages": dict(lang.most_common(15)),
        "countries": dict(country.most_common(20)),
        "non_english_share": (1 - known_lang.get("en", 0) / sum(known_lang.values())) if known_lang else None,
        "language_known_share": sum(known_lang.values()) / n if n else None,
        "top_country": max(known_c, key=known_c.get) if known_c else None,
        "top_country_share": max(known_c.values()) / with_country if known_c and with_country else None,
        "country_known_share": with_country / n if n else None,
        "eu_subject": rate_row(eu_n, n), "model_field_share": model_n / n if n else None,
        "with_fulltext": text_n,
        "health": {"records": len(hf), "direct_id": rate_row(int(hf["direct_id"].sum()), len(hf)),
                   "quasi_id": rate_row(int(hf["quasi_id"].sum()), len(hf)),
                   "case_report": rate_row(int(hf["case_report"].sum()), len(hf)),
                   "case_report_quasi_id": rate_row(int((hf["case_report"] & hf["quasi_id"]).sum()), len(hf))},
    }
    return stats, hf, pd.DataFrame(prov)


def collect(ctx: Context, scan: dict, prov: pd.DataFrame) -> dict[str, Any]:
    """The flat evidence map. Keys follow `<area>.<measure>` as used by configs/c1.yaml."""
    sd = ctx.cfg.stage_dir
    ev: dict[str, Any] = {}
    pii = _j(sd("s1", "pii") / "pii.json")
    if pii:
        ev["pii.screened"] = True
        ev["pii.summary_rate"] = pii["summary"]["rate"]
        ev["pii.fulltext_unexpected_rate"] = pii.get("fulltext_unexpected", {}).get("rate")
    if (sd("s1", "pii") / "pii_flags.parquet").exists():
        f = pd.read_parquet(sd("s1", "pii") / "pii_flags.parquet")
        med = f[(f["source"] == "summary") & (~f["expected"]) & f["_group"].str.startswith("pubmed")]
        ev["pii.health_summary_flags"] = int(med["paper_key"].nunique())
    h = scan["health"]
    if h["records"]:
        ev["pii.special_category_rate"] = h["direct_id"]["rate"]
        ev["pii.health_quasi_id_rate"] = h["case_report_quasi_id"]["rate"]
    ev["pii.eu_subject_share"] = scan["eu_subject"]["rate"]

    ev["optout.mechanism"] = bool(ctx.cfg.site.optout_file) or _optout_in_code()
    ev["optout.list_configured"] = bool(ctx.cfg.site.optout_file)
    docs = ctx.section("c1").get("docs", {})
    for key, rel in docs.items():
        ev[f"docs.{key}"] = signed(PROJECT_ROOT / rel) if key in SIGNED_DOCS else (PROJECT_ROOT / rel).exists()
    ds = sd("ws1", "datasheet")
    ev["docs.datasheet"] = (ds / "datasheet.md").exists()
    ev["docs.croissant_rai"] = (ds / "croissant_rai.json").exists()
    rai = _j(ds / "croissant_rai.json")
    base = _j(Path(rai["base"])) if rai.get("base") else {}
    ev["synthetic.croissant_flag"] = bool(base.get("rai:hasSyntheticData"))
    ev["synthetic.model_field"] = (scan["model_field_share"] or 0) >= 0.999
    ev["synthetic.model_field_share"] = scan["model_field_share"]

    lic = sd("ws1", "licenses") / "licenses.parquet"
    if lic.exists():
        lt = pd.read_parquet(lic)
        ev["licenses.found_share"] = float((lt["license_class"] != "none").mean())
        ev["licenses.cc_by_conflict_share"] = float(lt["cc_by_conflict"].mean())
        m = lt.merge(prov, on=["_file", "_row"], how="inner")
        ft = m[m["has_text"]]
        ev["licenses.fulltext_records"] = len(ft)
        if len(ft):
            ev["licenses.fulltext_nonpermissive_share"] = float((~ft["fulltext_redistributable"].astype(bool)).mean())
        ev["licenses.classes"] = lt["license_class"].value_counts().to_dict()
    tdm = _j(sd("c1", "tdm") / "tdm.json")
    ev["tdm.reservations_checked"] = bool(tdm)
    if tdm:
        ev["tdm.checked_share"] = tdm.get("checked_share")
        ev["tdm.reserved_share"] = tdm.get("reserved_share")

    ev["repr.examined"] = scan["records"] > 0
    ev["repr.non_english_share"] = scan["non_english_share"]
    ev["repr.top_country_share"] = scan["top_country_share"]
    ev["repr.top_country"] = scan["top_country"]
    desc = _j(sd("ws1", "describe") / "describe.json")
    if desc.get("domains"):
        tot = sum(desc["domains"].values())
        ev["repr.top_domain_share"] = max(desc["domains"].values()) / tot if tot else None

    q = _j(sd("ws1", "quality") / "dashboard.json")
    for panel in q.get("panels", []):
        for it in panel.get("items", []) if panel.get("type") == "kpis" else []:
            if it.get("label") == "with a critical flag":
                ev["quality.flag_rate"] = it["value"]
    ev["quality.measured"] = "quality.flag_rate" in ev

    g = _j(sd("s1", "guard") / "guard.json")
    if g:
        ev["guard.measured"] = True
        ev["guard.unsafe_rate"] = g["unsafe"]["rate"]
    du = _j(sd("s1", "dual-use") / "dual_use.json")
    if du:
        ev["dualuse.actionable"] = int(du.get("actionable", 0))
    inj = _j(sd("s1", "injection") / "injection.json")
    if inj:
        ev["injection.summary_rate"] = inj["summary"]["rate"]
    parts = sorted((sd("s1", "injection-live") / "parts").glob("*.parquet"))
    if parts:
        lv = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        lv = lv[lv["condition"] == "injected"]
        if len(lv):
            ev["injection.live_follow_rate"] = float(lv["followed"].mean())
    up = sd("s1", "results") / "wmdp_uplift.csv"
    if up.exists():
        u = pd.read_csv(up)
        if len(u):
            ev["uplift.wmdp_delta"] = float(u["uplift"].max())          # the worst (largest) uplift over models x benches
    pv = _j(sd("ws1", "provenance") / "provenance.json")
    if pv:
        ev["provenance.prompt_published"] = bool(pv.get("prompt_published"))
        ev["provenance.cost_usd_estimate"] = pv.get("cost_usd_estimate")
    ev["compute.reported"] = bool(pv.get("cost_usd_estimate")) or (sd("x6", "results") / "dashboard.json").exists()
    ev["repro.manifests"] = sum(1 for _ in ctx.cfg.run_root().glob("*/*/manifest*.json"))
    return ev


def signed(p: Path) -> bool:
    """A legal document counts only once approved: it exists and has a filled `Approved-by:` line."""
    try:
        return p.exists() and bool(SIGNED.search(p.read_text(encoding="utf-8")))
    except OSError:
        return False


def _optout_in_code() -> bool:
    from .. import data
    return hasattr(data, "optout_mask") and hasattr(data, "optout_ids")


@stage("c1", "evidence", config_keys=("c1.max_records", "c1.docs", "c1.eu_countries"),
       help="collect the compliance evidence (S1, WS1, X6 outputs) and measure health identifiers, EU scope, coverage")
def run(ctx: Context) -> dict:
    c = ctx.section("c1")
    scan, hf, prov = scan_records(ctx, int(c.get("max_records", 200000)), set(c.get("eu_countries", [])))
    hf.to_parquet(ctx.out / "health_flags.parquet", index=False)        # ids + booleans only, no text
    ev = collect(ctx, scan, prov)
    ctx.write_json("evidence.json", {"evidence": ev, "scan": scan})
    log(f"evidence: {len(ev)} keys from {scan['records']} records")
    h = scan["health"]
    lang = [{"label": k, "n": v, "share": v / scan["records"]} for k, v in scan["languages"].items()]
    ctry = [{"label": k, "n": v, "share": v / scan["records"]} for k, v in scan["countries"].items()]
    pct = lambda r: f"{r['rate']:.1%} [{r['lo']:.1%}, {r['hi']:.1%}]"  # noqa: E731
    ctx.publish("C1 compliance evidence", [
        {"type": "kpis", "items": [
            {"label": "records scanned", "value": scan["records"], "fmt": "int"},
            {"label": "authors in EU / EEA (GDPR scope)", "value": scan["eu_subject"]["rate"], "fmt": "pct1",
             "hint": "papers with >= 1 EU/EEA-affiliated author: their names are personal data under the GDPR"},
            {"label": "health KUs with direct identifiers", "value": h["direct_id"]["rate"], "fmt": "pct1",
             "status": "good" if h["direct_id"]["k"] == 0 else "bad", "hint": f"{h['direct_id']['k']} of {h['records']} PubMed KUs"},
            {"label": "case reports with exact age + sex", "value": h["case_report_quasi_id"]["rate"], "fmt": "pct1",
             "status": "warn" if h["case_report_quasi_id"]["k"] else "good", "hint": "quasi-identifiers (GDPR Art. 9 context)"},
            {"label": "non-English (known language)", "value": scan["non_english_share"], "fmt": "pct1"},
            {"label": f"top author country ({scan['top_country']})", "value": scan["top_country_share"], "fmt": "pct1"}]},
        {"type": "bar", "title": "Languages (OpenAlex)", "labels": [r["label"] for r in lang],
         "series": [{"name": "share of records", "data": [r["share"] for r in lang]}], "fmt": "pct", "horizontal": True},
        {"type": "bar", "title": "Author countries (share of records with >= 1 author there)", "labels": [r["label"] for r in ctry],
         "series": [{"name": "share of records", "data": [r["share"] for r in ctry]}], "fmt": "pct", "horizontal": True},
        {"type": "table", "title": "Health-data screen (PubMed KUs, GDPR Art. 9)",
         "hint": "Regex screen; counts only. Direct identifiers = record numbers, titled names, exact dates of birth / admission.",
         "columns": [{"key": "measure", "label": "measure"}, {"key": "value", "label": "rate [95% CI]"}],
         "rows": [{"measure": "direct identifiers", "value": pct(h["direct_id"])},
                  {"measure": "case report", "value": pct(h["case_report"])},
                  {"measure": "exact age + sex (any)", "value": pct(h["quasi_id"])},
                  {"measure": "case report with exact age + sex", "value": pct(h["case_report_quasi_id"])}]},
        {"type": "table", "title": "All evidence keys", "columns": [{"key": "key", "label": "key"}, {"key": "value", "label": "value"}],
         "rows": [{"key": k, "value": json.dumps(v) if isinstance(v, dict) else str(v)} for k, v in sorted(ev.items())],
         "pageSize": 15}],
        summary=f"{len(ev)} evidence keys; {scan['records']} records scanned")
    return {"keys": len(ev), "records": scan["records"]}
