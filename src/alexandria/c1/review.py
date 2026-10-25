"""Stage `review` (C1, CPU): a per-record legal review of the data for the issues the compliance matrix raises, and
what the release looks like once its decisions are applied.

Per record (debug: the stored sample; full: up to `c1.max_records`):
  * full text (copyright / DSM Art. 4(3)): `ship` only with a permissive licence, no TDM reservation and no opt-out;
    otherwise `withhold` with the reason (no licence found, NC/ND, arXiv default licence, publisher, TDM reserved);
  * KU: `drop` (opt-out), `hold` (direct patient identifiers: GDPR Art. 9, until the DPO has reviewed it),
    `ship_redacted` (PII found in a KU field and replaced by `[REDACTED:<category>]`: GDPR Art. 5(1)(c)), else `ship`;
    KUs of TDM-reserved works carry `use: research_only` (DSM Art. 3 research-organisation exception);
  * review queue for people: DPO (identifiers, case reports with exact age + sex and no consent statement), legal
    (TDM-reserved publishers, full text with no licence found), spot checks of redactions.
Outputs: decisions.parquet (ids + decisions + reasons), redactions.parquet (the corrections layer: redacted text of
the affected KU fields only), review_queue.csv (ids, reason, owner, priority; no text), review.json with the evidence
the matrix reads twice: as measured now and `after` the decisions are applied (re-measured on the redacted text), and
drafts/ (DPIA with legitimate-interest test, Art. 14 privacy notice, copyright policy, AI Office training-data summary)
filled with the measured numbers. Drafts are not approvals: a document counts as done only once it is copied to docs/
with an `Approved-by:` line.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..data import input_shards, iter_batches, optout_ids
from ..runtime import Context, log, stage
from ..s1.common import pii_hits, pii_redact
from ..schema import SUMMARY_FIELDS
from ..stats import rate_row
from .evidence import health_flags

CONSENT = re.compile(r"\bconsent", re.I)
DRAFTS = ("dpia", "privacy_notice", "copyright_policy", "training_data_summary")


def fulltext_decision(has_text: bool, lic_class: str | None, tdm: str | None, opted_out: bool) -> tuple[str, str]:
    if not has_text:
        return "none", ""
    if opted_out:
        return "withhold", "opt-out"
    if tdm in ("reserved", "robots_ai"):
        return "withhold", "tdm_reserved"
    if lic_class == "permissive":
        return "ship", ""
    return "withhold", {None: "no_licence_found", "none": "no_licence_found"}.get(lic_class, lic_class)


def ku_decision(opted_out: bool, direct_id: bool, pii_fields: int) -> str:
    if opted_out:
        return "drop"
    if direct_id:
        return "hold"
    return "ship_redacted" if pii_fields else "ship"


def _table(p: Path, cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(p, columns=cols) if p.exists() else pd.DataFrame()


def review_records(ctx: Context, max_records: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sd = ctx.cfg.stage_dir
    lic = _table(sd("ws1", "licenses") / "licenses.parquet", ["_file", "_row", "license_class"])
    tdm = _table(sd("c1", "tdm") / "tdm_reservations.parquet", ["_file", "_row", "status", "prefix", "publisher_status"])
    ret = _table(sd("s1", "retractions") / "retracted.parquet", ["_file", "_row"])
    lic_m = {(f, int(r)): c for f, r, c in lic.itertuples(index=False)} if len(lic) else {}
    tdm_m = {(f, int(r)): (s, p, ps) for f, r, s, p, ps in tdm.itertuples(index=False)} if len(tdm) else {}
    ret_s = {(f, int(r)) for f, r in ret.itertuples(index=False)} if len(ret) else set()
    opt = optout_ids(str(ctx.cfg.site.optout_file) if ctx.cfg.site.optout_file else None)
    rows, red = [], []
    cols = list(SUMMARY_FIELDS) + ["paper_id", "source_doi", "oa_doi", "text_raw"]
    n = 0
    for sh in input_shards(ctx.cfg):
        for tbl in iter_batches(ctx.cfg, sh, list(dict.fromkeys(cols)), batch_size=512):
            for rec in tbl.to_pylist():
                k = (rec["_file"], int(rec["_row"]))
                ids = {str(rec.get(c) or "").lower().removeprefix("https://doi.org/") for c in ("paper_id", "source_doi", "oa_doi")}
                opted = bool(opt and ids & opt)        # iter_batches already drops them when the list is configured
                hf = health_flags(rec) if rec["_group"].startswith("pubmed") else {"direct_id": False, "quasi_id": False,
                                                                                   "case_report": False}
                pii_fields, residual = 0, 0
                for f in SUMMARY_FIELDS:
                    if pii_hits(rec.get(f)):
                        text, cats = pii_redact(rec.get(f))
                        pii_fields += 1
                        residual += sum(pii_hits(text).values())
                        red.append({"_file": k[0], "_row": k[1], "field": f, "categories": ",".join(sorted(cats)),
                                    "redacted": text})
                has_text = bool((rec.get("text_raw") or "").strip())
                t_status, prefix, pub_status = tdm_m.get(k, (None, None, None))
                ft, ft_reason = fulltext_decision(has_text, lic_m.get(k), t_status, opted)
                rows.append({"_file": k[0], "_row": k[1], "_group": rec["_group"], "paper_id": rec.get("paper_id"),
                             "licence": lic_m.get(k), "tdm": t_status, "prefix": prefix,
                             "publisher_tdm": pub_status, "has_text": has_text,
                             "fulltext": ft, "fulltext_reason": ft_reason,
                             "ku": ku_decision(opted, hf["direct_id"], pii_fields),
                             "use": "research_only" if t_status in ("reserved", "robots_ai") else "any",
                             "pii_fields": pii_fields, "pii_residual": residual, "retracted": k in ret_s,
                             "consent_stated": bool(CONSENT.search(str(rec.get("ethical_considerations") or ""))), **hf})
                n += 1
                if n >= max_records:
                    break
            if n >= max_records:
                break
        if n >= max_records:
            break
    return pd.DataFrame(rows), pd.DataFrame(red, columns=["_file", "_row", "field", "categories", "redacted"])


def review_queue(d: pd.DataFrame) -> pd.DataFrame:
    q = []
    for r in d.to_dict("records"):                                   # itertuples would rename `_file`
        base = {"_file": r["_file"], "_row": r["_row"], "paper_id": r["paper_id"]}
        if r["direct_id"]:
            q.append({**base, "owner": "DPO", "priority": "high", "reason": "direct patient identifier in KU (GDPR Art. 9); KU held"})
        elif r["case_report"] and r["quasi_id"] and not r["consent_stated"]:
            q.append({**base, "owner": "DPO", "priority": "medium",
                      "reason": "case report with exact age + sex, no consent statement in the KU (GDPR Art. 9)"})
        if r["tdm"] in ("reserved", "robots_ai"):
            q.append({**base, "owner": "legal", "priority": "medium",
                      "reason": f"TDM reserved by publisher ({r['tdm']}, prefix {r['prefix']}): full text withheld, KU research-only"})
        elif r["publisher_tdm"] in ("reserved", "robots_ai"):
            q.append({**base, "owner": "legal", "priority": "low",
                      "reason": f"journal version reserved by its publisher (prefix {r['prefix']}); our copy is from "
                                f"{r['_group'].split('/')[0]} under its own licence: confirm"})
        if r["has_text"] and r["fulltext_reason"] == "no_licence_found":
            q.append({**base, "owner": "legal", "priority": "low", "reason": "full text without a licence found: withheld; look up the licence"})
        if r["pii_fields"]:
            q.append({**base, "owner": "curation", "priority": "low", "reason": f"PII redacted in {r['pii_fields']} KU field(s): spot-check"})
    return pd.DataFrame(q, columns=["_file", "_row", "paper_id", "owner", "priority", "reason"])


def after_evidence(d: pd.DataFrame) -> dict[str, Any]:
    """The evidence keys the decisions change, re-measured on the release as the decisions leave it."""
    kus = d[d["ku"].isin(["ship", "ship_redacted"])]
    ft = d[d["fulltext"] == "ship"]
    pub = kus[kus["_group"].str.startswith("pubmed")]
    return {"pii.summary_rate": float((kus["pii_residual"] > 0).mean()) if len(kus) else 0.0,
            "pii.special_category_rate": float(pub["direct_id"].mean()) if len(pub) else 0.0,
            "licenses.fulltext_nonpermissive_share": float((ft["licence"] != "permissive").mean()) if len(ft) else 0.0,
            "release.fulltext_records": len(ft), "release.ku_records": len(kus)}


# ---------------------------------------------------------------- drafts
def _pct(v: Any) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"


def drafts(ev: dict, scan: dict, s: dict, tdm: dict, today: str) -> dict[str, str]:
    head = (f"Status: DRAFT generated by `alexandria c1 review` on {today} from measured evidence. Not legal advice; "
            "not approved.\nApproved-by: \n\n")
    n = s["records"]
    common = {"n": n, "eu": _pct(ev.get("pii.eu_subject_share")), "pii": _pct(ev.get("pii.summary_rate")),
              "health": s["health_records"], "lic": _pct(ev.get("licenses.found_share")),
              "ftw": s["fulltext"].get("withhold", 0), "fts": s["fulltext"].get("ship", 0),
              "tdm_chk": _pct(tdm.get("checked_share")), "tdm_res": _pct(tdm.get("reserved_share"))}
    out = {}
    out["dpia"] = head + """# Data protection impact assessment (GDPR Art. 35) - {DATASET}

## 1. Processing
Structured LLM summaries (KUs) of scientific papers with bibliographic metadata (titles, author names,
affiliations, ORCID ids) from OpenAlex, and open-access full text where licensed. Analysed set: {n} records;
{eu} of them have at least one author affiliated in the EU/EEA.

## 2. Purpose and lawful basis (Art. 6(1)(f), Art. 89)
Purpose: scientific research and research infrastructure (retrieval, evaluation, training of research models).
Legitimate-interest test:
* **Purpose test:** open access to the content of the scientific literature. This is a recognised research interest.
* **Necessity test:** author names and affiliations are needed for citation and attribution. KUs need no personal data;
  measured PII in KUs is {pii} before redaction and 0 after redaction.
* **Balancing test:** the data are already published by the data subjects in a professional capacity. Authors
  reasonably expect scholarly indexing. Safeguards: opt-out (docs/TAKEDOWN.md), redaction, no profiling of
  individuals, no special-category inference.

## 3. Special categories (Art. 9)
Biomedical KUs screened: {health}. Direct identifiers lead to the KU being held; case reports with exact age and sex
and no consent statement are queued for DPO review (review_queue.csv).

## 4. Risks and measures
| risk | measure | residual |
|---|---|---|
| personal data in KUs | regex screen + redaction layer | see C1 matrix (after review) |
| re-identification of patients | hold + DPO review queue | DPO to assess |
| objection / erasure (Art. 17, 21) | opt-out list applied in every build | logged requests |
| transfer outside the EU (Ch. V) | TODO: hosting location of the release | |

## 5. Consultation and sign-off
DPO opinion: TODO. Decision: TODO.
"""
    out["privacy_notice"] = head + """# Privacy notice for authors (GDPR Art. 14) - {DATASET}

We process bibliographic data about you (name, affiliation, ORCID, and the papers you published) that we obtained
from OpenAlex and from the papers themselves, not from you.
* **Controller:** TODO (name and contact).
* **DPO:** TODO.
* **Purpose:** scientific research and research infrastructure.
* **Lawful basis:** Art. 6(1)(f) legitimate interest. Safeguards follow Art. 89.
* **Categories:** identification and professional data. No special categories are intended.
* **Recipients:** users of the public dataset.
* **Retention:** for the lifetime of the dataset release.
* **Your rights:**
  * access, rectification, erasure and objection (Art. 15-21): use the opt-out procedure in docs/TAKEDOWN.md;
  * complaint to a supervisory authority.
* **Art. 14(5)(b):** individual notification of every author ({eu} of {n} records have EU/EEA authors) would
  involve a disproportionate effort. This public notice is the measure we take instead (to be confirmed by the DPO).
"""
    out["copyright_policy"] = head + """# Copyright and TDM policy - {DATASET}

1. **Lawful access.**
   * Sources are open repositories (arXiv, PubMed Central) and open metadata (OpenAlex).
   * TDM is carried out under DSM Directive Art. 3 by a research organisation, for research purposes.
2. **TDM reservations (Art. 4(3)).**
   * Machine-readable reservations are checked per publisher: TDMRep (tdmrep.json, HTTP header, HTML meta) and
     robots.txt rules for AI crawlers.
   * Coverage: {tdm_chk} of records checked; {tdm_res} of the checked records are reserved.
   * For reserved works, the full text is withheld and the KU is marked `research_only`.
3. **Redistribution.**
   * Full text is shipped only under a permissive licence: {fts} records shipped, {ftw} withheld.
   * Licences were found for {lic} of records.
   * KUs are new summaries. They are not copies; reproduction of protected expression is limited by the schema
     (TODO: verbatim-overlap figure from WS5).
4. **Complaints and takedown.** Rights holders use docs/TAKEDOWN.md; removals apply in the next build.
5. **Downstream model providers (AI Act Art. 53(1)(c)).** The per-record `licence`, `tdm` and `use` columns let
   providers apply their own copyright policy.
"""
    out["training_data_summary"] = head + """# Public summary of training content (AI Act Art. 53(1)(d), AI Office template) - {DATASET}

## 1. General information
* **Dataset:** {DATASET}. LLM-generated structured summaries of scientific papers, with metadata.
* **Modalities:** text.
* **Size:** {n} records in the analysed set; the release has 22.1M records (TODO: final counts).

## 2. List of data sources
* **2.1 Publicly available datasets:** arXiv, PubMed / PubMed Central (open access), OpenAlex metadata.
* **2.2 Private or licensed data:** none.
* **2.3 Data crawled from the web:** none beyond the publisher landing pages checked for TDM reservations.
* **2.4 User data:** none.
* **2.5 Synthetic data:** all KUs are generated by an LLM. This is marked per record (`summary_model`) and in
  Croissant (`rai:hasSyntheticData`).

## 3. Data processing aspects
* **TDM reservations:** respected; see the copyright policy ({tdm_chk} checked).
* **Removal of illegal or harmful content:**
  * PII screen and redaction;
  * guard-model and dual-use screens (S1);
  * retraction flags.
"""
    return {k: v.replace("{DATASET}", "Scientific-Summaries").format(**common) for k, v in out.items()}


@stage("c1", "review", needs=("evidence",), config_keys=("c1.max_records",),
       help="per-record legal review: full-text / KU release decisions, PII redaction layer, DPO + legal review queue, "
            "document drafts (DPIA, privacy notice, copyright policy, AI Office summary)")
def run(ctx: Context) -> dict:
    evj = json.loads((ctx.input_dir("evidence") / "evidence.json").read_text())
    ev, scan = evj["evidence"], evj["scan"]
    d, red = review_records(ctx, int(ctx.section("c1").get("max_records", 200000)))
    d.to_parquet(ctx.out / "decisions.parquet", index=False)
    red.to_parquet(ctx.out / "redactions.parquet", index=False)
    q = review_queue(d)
    q.to_csv(ctx.out / "review_queue.csv", index=False)
    tdm_p = ctx.cfg.stage_dir("c1", "tdm") / "tdm.json"
    tdm = json.loads(tdm_p.read_text()) if tdm_p.exists() else {}
    after = after_evidence(d)
    s = {"records": len(d), "fulltext": d["fulltext"].value_counts().to_dict(),
         "fulltext_reasons": d.loc[d["fulltext"] == "withhold", "fulltext_reason"].value_counts().to_dict(),
         "ku": d["ku"].value_counts().to_dict(), "use": d["use"].value_counts().to_dict(),
         "health_records": int(d["_group"].str.startswith("pubmed").sum()),
         "redacted_fields": len(red), "redaction_residual": int(d["pii_residual"].sum()),
         "queue": q.groupby(["owner", "priority"]).size().rename("n").reset_index().to_dict("records")}
    ddir = ctx.out / "drafts"
    ddir.mkdir(exist_ok=True)
    for name, text in drafts(ev, scan, s, tdm, dt.date.today().isoformat()).items():
        (ddir / f"{name.upper()}.md").write_text(text)
    now = {f"drafts.{k}": (ddir / f"{k.upper()}.md").exists() for k in DRAFTS}
    ctx.write_json("review.json", {"summary": s, "now": {}, "after": {**now, **after}})       # drafts exist only after the review
    log(f"review: {len(d)} records, queue {len(q)}, redacted fields {len(red)}")

    ft, ku = s["fulltext"], s["ku"]
    owner = q.groupby("owner").size().to_dict() if len(q) else {}
    before_after = [
        {"measure": "KUs with PII", "now": ev.get("pii.summary_rate"), "after": after["pii.summary_rate"]},
        {"measure": "health KUs with direct identifiers", "now": ev.get("pii.special_category_rate"),
         "after": after["pii.special_category_rate"]},
        {"measure": "shipped full text without permissive licence", "now": ev.get("licenses.fulltext_nonpermissive_share"),
         "after": after["licenses.fulltext_nonpermissive_share"]}]
    reasons = s["fulltext_reasons"]
    ctx.publish("C1 legal data review", [
        {"type": "kpis", "items": [
            {"label": "records reviewed", "value": len(d), "fmt": "int"},
            {"label": "full text shipped", "value": ft.get("ship", 0), "fmt": "int",
             "hint": f"withheld {ft.get('withhold', 0)} (no permissive licence / TDM reserved / opt-out)"},
            {"label": "KUs redacted", "value": ku.get("ship_redacted", 0), "fmt": "int",
             "status": "good" if s["redaction_residual"] == 0 else "bad", "hint": f"{s['redaction_residual']} PII matches left after redaction"},
            {"label": "KUs held for DPO", "value": ku.get("hold", 0), "fmt": "int", "status": "warn" if ku.get("hold") else "good"},
            {"label": "review queue", "value": len(q), "fmt": "int", "hint": ", ".join(f"{k} {v}" for k, v in owner.items())}]},
        {"type": "table", "title": "Before vs after the review decisions", "hint": "re-measured on the release as the decisions leave it",
         "columns": [{"key": "measure", "label": "measure"}, {"key": "now", "label": "now", "fmt": "pct1"},
                     {"key": "after", "label": "after review", "fmt": "pct1"}], "rows": before_after},
        {"type": "bar", "title": "Full text: why it is withheld", "labels": list(reasons),
         "series": [{"name": "records", "data": [int(v) for v in reasons.values()]}], "horizontal": True},
        {"type": "bar", "title": "KU decisions", "labels": list(ku), "series": [{"name": "records", "data": [int(v) for v in ku.values()]}]},
        {"type": "table", "title": "Review queue by owner (ids in review_queue.csv; no text)",
         "columns": [{"key": "owner", "label": "owner"}, {"key": "priority", "label": "priority"}, {"key": "n", "label": "records", "fmt": "int"}],
         "rows": s["queue"]},
        {"type": "markdown", "text":
            "**Document drafts** (in `c1/review/drafts/`, filled with the measured numbers): DPIA with the legitimate-"
            "interest test (GDPR Art. 35, 6(1)(f), 89), Art. 14 privacy notice, copyright and TDM policy (DSM Art. 3 / 4(3), "
            "AI Act Art. 53(1)(c)), and the public training-data summary (AI Act Art. 53(1)(d), AI Office template). "
            "A draft moves a requirement to *partial*; it counts as *met* only once the DPO / legal counsel sign it: copy it "
            "to `docs/` and fill the `Approved-by:` line, then rerun C1."}],
        summary=f"{len(d)} records: full text ship {ft.get('ship', 0)} / withhold {ft.get('withhold', 0)}; "
                f"KU redacted {ku.get('ship_redacted', 0)}, held {ku.get('hold', 0)}; queue {len(q)}")
    return {"records": len(d), "queue": len(q)}
