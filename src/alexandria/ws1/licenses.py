"""Stage `licenses` (WS1, LOGIN node: needs internet): a per-record licence field.

A blanket CC-BY-4.0 for the release conflicts with the original open-access licences (arXiv's default
non-exclusive licence, NC / ND licences on bioRxiv / PMC, bronze / green OA without a licence). This stage looks up
each record's licence and classifies what it allows:
  1. OpenAlex (api.openalex.org, batched by OpenAlex id): `best_oa_location.license`, else the first location with a
     licence; also the OA status (gold / green / hybrid / bronze / closed);
  2. arXiv OAI-PMH (export.arxiv.org/oai2, metadataPrefix=arXiv, rate-limited) for arXiv ids without an OpenAlex
     licence: the licence URL the authors chose at submission.
Classes: permissive (CC-BY, CC-BY-SA, CC0 / public domain), nc_or_nd (any NC or ND), arxiv_default (arXiv
non-exclusive distribution licence: no redistribution right for third parties), publisher (publisher-specific OA),
none (no licence found; e.g. bronze OA). `fulltext_redistributable` is true only for permissive licences;
`cc_by_conflict` flags records a blanket CC-BY release would mislabel. Debug: the WS1 sample; full: every record
(use the OpenAlex snapshot rather than the API for 22M records; see README).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from ..data import input_shards, iter_batches
from ..runtime import Context, log, stage
from .audit import arxiv_id

PERMISSIVE = re.compile(r"^(cc-by|cc-by-sa|cc0|public-domain|pd)$")
_ARXIV_LIC = {"nonexclusive-distrib": "arxiv_default", "by/4.0": "cc-by", "by-sa/4.0": "cc-by-sa", "by-nc-sa/4.0": "cc-by-nc-sa",
              "by-nc-nd/4.0": "cc-by-nc-nd", "publicdomain/zero": "cc0", "by/3.0": "cc-by", "by-nc-sa/3.0": "cc-by-nc-sa"}


def classify(lic: str | None) -> str:
    if not lic:
        return "none"
    lic = lic.lower()
    if lic == "arxiv_default":
        return "arxiv_default"
    if "nc" in lic.split("-") or "nd" in lic.split("-"):
        return "nc_or_nd"
    if PERMISSIVE.match(lic):
        return "permissive"
    if "publisher" in lic or lic.startswith("other"):
        return "publisher"
    return "other"


def arxiv_license(raw: str | None) -> str | None:
    """Licence label from an OAI-PMH answer: None = not fetched; "none_given" = arXiv's default licence."""
    if raw is None:
        return None
    if raw == "none_given":
        return "arxiv_default"
    for k, v in _ARXIV_LIC.items():
        if k in raw:
            return v
    return "other"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "alexandria-licence-audit/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def openalex_licences(ids: list[str], pause: float = 0.2) -> dict[str, dict]:
    out = {}
    short = [i.rsplit("/", 1)[-1] for i in ids]
    for k in range(0, len(short), 50):
        chunk = short[k:k + 50]
        url = ("https://api.openalex.org/works?per-page=50&select=id,open_access,best_oa_location,locations&filter=openalex:"
               + "|".join(chunk))
        try:
            res = json.loads(_get(url))["results"]
        except Exception as e:                                        # noqa: BLE001  (network: recorded, not fatal)
            log(f"licenses: OpenAlex batch failed: {e}")
            continue
        for w in res:
            best = w.get("best_oa_location") or {}
            lic = best.get("license") or next((loc.get("license") for loc in w.get("locations") or [] if loc.get("license")), None)
            arx = next((loc.get("landing_page_url") for loc in w.get("locations") or []
                        if "arxiv.org/abs/" in str(loc.get("landing_page_url") or "")), None)
            out["https://openalex.org/" + w["id"].rsplit("/", 1)[-1]] = {
                "openalex_license": lic, "oa_status": (w.get("open_access") or {}).get("oa_status"),
                "arxiv_from_openalex": re.sub(r"v\d+$", "", arx.rsplit("/abs/", 1)[-1]) if arx else None}
        time.sleep(pause)
    return out


def oai_licence(aid: str) -> str | None:
    url = "https://export.arxiv.org/oai2?verb=GetRecord&metadataPrefix=arXiv&identifier=" + urllib.parse.quote(f"oai:arXiv.org:{aid}")
    try:
        root = ElementTree.fromstring(_get(url))
    except Exception:                                                  # noqa: BLE001
        return None
    el = root.find(".//{http://arxiv.org/OAI/arXiv/}license")
    return el.text.strip() if el is not None and el.text else "none_given"


@stage("ws1", "licenses", config_keys=("ws1.licenses",),
       help="per-record licence (OpenAlex + arXiv OAI-PMH; LOGIN node) and conflicts with a blanket CC-BY")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws1.licenses")
    rows = []
    for sh in input_shards(ctx.cfg):
        for tbl in iter_batches(ctx.cfg, sh, ["paper_id", "openalex_id", "source_doi", "oa_doi", "oa_is_open_access"], 4096):
            rows += tbl.to_pylist()
    df = pd.DataFrame(rows)
    df["arxiv_id"] = [arxiv_id(p, a, b)[0] for p, a, b in zip(df["paper_id"], df["source_doi"], df["oa_doi"], strict=True)]
    oa = openalex_licences(sorted(df["openalex_id"].dropna().unique()), float(cfg.get("openalex_pause", 0.2)))
    df["openalex_license"] = df["openalex_id"].map(lambda i: (oa.get(i) or {}).get("openalex_license"))
    df["oa_status"] = df["openalex_id"].map(lambda i: (oa.get(i) or {}).get("oa_status"))
    df["arxiv_id"] = df["arxiv_id"].fillna(df["openalex_id"].map(lambda i: (oa.get(i) or {}).get("arxiv_from_openalex")))
    need = df[df["arxiv_id"].notna() & (df["openalex_license"].isna() | df["_group"].str.startswith("arxiv"))]["arxiv_id"].unique()
    need = list(need)[:int(cfg.get("max_oai_requests", 200))]
    log(f"licenses: {len(oa)} OpenAlex works; arXiv OAI-PMH for {len(need)} ids (3 s apart)")
    oai = {}
    for a in need:
        oai[a] = oai_licence(a)
        time.sleep(float(cfg.get("oai_pause", 3.0)))
    df["arxiv_license"] = df["arxiv_id"].map(lambda a: arxiv_license(oai.get(a)))
    df["license"] = df["arxiv_license"].fillna(df["openalex_license"])        # the author's arXiv choice wins
    df["license_source"] = np.where(df["arxiv_license"].notna(), "arxiv_oai",
                                    np.where(df["openalex_license"].notna(), "openalex", "none"))
    df["license_class"] = df["license"].map(classify)
    df["fulltext_redistributable"] = df["license_class"] == "permissive"
    df["cc_by_conflict"] = df["license_class"] != "permissive"
    keep = ["_file", "_row", "_group", "paper_id", "arxiv_id", "openalex_id", "oa_status", "license", "license_source",
            "license_class", "fulltext_redistributable", "cc_by_conflict"]
    df[keep].to_parquet(ctx.out / "licenses.parquet", index=False)
    dist = df.groupby(["_group", "license_class"]).size().unstack(fill_value=0)
    lic = df["license"].fillna("none").value_counts().head(15)
    ctx.publish("WS1 per-record licences", [
        {"type": "kpis", "items": [
            {"label": "records checked", "value": len(df), "fmt": "int"},
            {"label": "permissive (CC-BY / CC0)", "value": float((df["license_class"] == "permissive").mean()), "fmt": "pct1"},
            {"label": "conflict with blanket CC-BY", "value": float(df["cc_by_conflict"].mean()), "fmt": "pct1", "status": "warn",
             "hint": "NC / ND, arXiv default licence, publisher-specific or no licence"},
            {"label": "licence found", "value": float(df["license"].notna().mean()), "fmt": "pct1"}]},
        {"type": "bar", "title": "Licence class per subset", "horizontal": True, "fmt": "int", "stacked": True,
         "labels": [str(x) for x in dist.index], "series": [{"name": c, "data": dist[c].tolist()} for c in dist.columns]},
        {"type": "bar", "title": "Most frequent licences", "horizontal": True, "fmt": "int",
         "labels": [str(x) for x in lic.index], "series": [{"name": "records", "data": lic.tolist()}]},
        {"type": "markdown", "text": "Per-record licence column for the release: `licenses.parquet` (licence, source, class, "
                                     "`fulltext_redistributable`, `cc_by_conflict`). Opt-out / takedown: docs/TAKEDOWN.md "
                                     "and `site.optout_file` (records listed there are dropped by every stage)."},
    ], summary=f"{df['cc_by_conflict'].mean():.0%} of records conflict with a blanket CC-BY")
    return {"records": len(df), "conflict_share": float(df["cc_by_conflict"].mean())}
