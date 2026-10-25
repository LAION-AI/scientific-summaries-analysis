"""Stage `tdm` (C1, LOGIN node: needs the internet): machine-readable text-and-data-mining reservations per record.

DSM Directive Art. 4(3) lets rights holders reserve TDM "in an appropriate manner, such as machine-readable means";
AI Act Art. 53(1)(c) and the GPAI Code of Practice (copyright chapter) ask model providers to identify and respect
them, including robots.txt. Per record we check, for the publisher's landing page of its DOI (arXiv records: arxiv.org):
  * TDMRep (W3C Community Group): `/.well-known/tdmrep.json` rules matching the landing path, the `tdm-reservation`
    HTTP header and the `<meta name="tdm-reservation">` tag (1 = reserved);
  * robots.txt: whether AI / TDM crawlers (GPTBot, CCBot, Google-Extended, ClaudeBot, ...) may fetch the landing page.
DOIs are resolved per DOI prefix (registrant): up to `dois_per_prefix` per prefix, so the full release (22M records,
a few thousand prefixes) needs a bounded number of requests; other DOIs of the prefix inherit the prefix's result.
Status per record: reserved (TDMRep / header / meta) > robots_ai (AI crawlers blocked) > none > unknown (not resolved,
publisher unreachable or blocking automated access). Only ids and statuses are stored.
"""
from __future__ import annotations

import fnmatch
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter

import pandas as pd

from ..data import input_shards, iter_batches
from ..runtime import Context, log, stage

UA = "alexandria-tdm-audit/0.1 (dataset compliance check)"
PMC = re.compile(r"PMC\d{4,}")
BOT_WALLS = ("perfdrive.com", "captcha", "challenges.cloudflare.com", "/cdn-cgi/challenge", "distil", "incapsula")
META = re.compile(r"<meta[^>]+name=[\"']tdm-reservation[\"'][^>]*content=[\"']\s*([01])", re.I)
META_REV = re.compile(r"<meta[^>]+content=[\"']\s*([01])\s*[\"'][^>]*name=[\"']tdm-reservation[\"']", re.I)


def norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = str(d).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p):]
    return d if d.startswith("10.") and "/" in d else None


def doi_prefix(doi: str) -> str:
    return doi.split("/", 1)[0]


def _open(url: str, timeout: float, method: str = "GET", limit: int = 65536) -> tuple[int, str, dict, str]:
    """-> (status, final url, headers, body prefix). Status 0 = network error."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, "Accept": "text/html,application/json,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(limit).decode("utf-8", "replace") if method == "GET" else ""
            return r.status, r.geturl(), {k.lower(): v for k, v in r.headers.items()}, body
    except urllib.error.HTTPError as e:
        return e.code, getattr(e, "url", url) or url, {k.lower(): v for k, v in (e.headers or {}).items()}, ""
    except Exception:  # noqa: BLE001  (DNS, TLS, timeouts, resets: the publisher is unreachable)
        return 0, url, {}, ""


def tdmrep_rules(text: str) -> list[dict]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    return [r for r in data if isinstance(r, dict) and "location" in r] if isinstance(data, list) else []


def tdmrep_match(rules: list[dict], path: str) -> int | None:
    """The tdm-reservation of the most specific rule whose location pattern matches the path (None: no rule)."""
    hits = [r for r in rules if fnmatch.fnmatch(path, str(r["location"])) or
            fnmatch.fnmatch(path, str(r["location"]).rstrip("*") + "*")]
    if not hits:
        return None
    best = max(hits, key=lambda r: len(str(r["location"]).rstrip("*")))
    try:
        return int(best.get("tdm-reservation", 0))
    except (TypeError, ValueError):
        return None


def meta_reservation(html: str) -> int | None:
    m = META.search(html or "") or META_REV.search(html or "")
    return int(m.group(1)) if m else None


def robots_blocked(robots_txt: str, url: str, agents: list[str]) -> list[str]:
    """AI / TDM agents that robots.txt does not allow to fetch the url."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse((robots_txt or "").splitlines())
    return [a for a in agents if not rp.can_fetch(a, url)]


def record_status(r: dict) -> str:
    if r.get("tdmrep") == 1 or r.get("header") == 1 or r.get("meta") == 1:
        return "reserved"
    if r.get("robots_ai"):
        return "robots_ai"
    if r.get("resolved"):
        return "none"
    return "unknown"


def combine(res: list[dict]) -> dict:
    """One result per DOI prefix from its sampled landing pages: reserved if any page is (conservative)."""
    ok = [r for r in res if r["resolved"]] or res

    def worst(k):
        vals = [r[k] for r in ok if r[k] is not None]
        return max(vals) if vals else None
    return {**ok[0], "tdmrep": worst("tdmrep"), "header": worst("header"), "meta": worst("meta"),
            "robots_ai": ",".join(sorted({a for r in ok for a in r["robots_ai"].split(",") if a})),
            "resolved": any(r["resolved"] for r in ok), "sampled": len(res)}


class Checker:
    """Cached per-domain (tdmrep.json, robots.txt) and per-landing-page checks, politely paced."""

    def __init__(self, agents: list[str], timeout: float, pause: float):
        self.agents, self.timeout, self.pause = agents, timeout, pause
        self.domains: dict[str, dict] = {}
        self.requests = 0

    def _get(self, url: str, method: str = "GET"):
        self.requests += 1
        time.sleep(self.pause)
        return _open(url, self.timeout, method)

    def domain(self, scheme_host: str) -> dict:
        if scheme_host not in self.domains:
            st, _, _, body = self._get(scheme_host + "/.well-known/tdmrep.json")
            rules = tdmrep_rules(body) if st == 200 else []
            rst, _, _, robots = self._get(scheme_host + "/robots.txt")
            self.domains[scheme_host] = {"tdmrep_status": st, "rules": rules, "robots_status": rst,
                                         "robots": robots if rst == 200 else ""}
        return self.domains[scheme_host]

    def landing(self, url: str) -> dict:
        st, final, headers, body = self._get(url)
        u = urllib.parse.urlsplit(final)
        host = f"{u.scheme}://{u.netloc}"
        d = self.domain(host) if u.netloc else {"rules": [], "robots": "", "tdmrep_status": 0, "robots_status": 0}
        hdr = headers.get("tdm-reservation")
        walled = any(w in final for w in BOT_WALLS) or not 200 <= st < 400      # anti-bot page / error: not checked
        return {"resolved": bool(u.netloc) and "doi.org" not in u.netloc and not walled, "http_status": st, "domain": u.netloc,
                "tdmrep": tdmrep_match(d["rules"], u.path or "/"), "tdmrep_file": d["tdmrep_status"] == 200,
                "header": int(hdr.strip()[0]) if hdr and hdr.strip()[:1] in "01" else None,
                "meta": meta_reservation(body), "tdm_policy": headers.get("tdm-policy"),
                "robots_ai": ",".join(robots_blocked(d["robots"], final, self.agents)) if d["robots"] else ""}


@stage("c1", "tdm", config_keys=("c1.tdm",),
       help="TDM reservations per record: TDMRep (tdmrep.json, header, meta) + robots.txt AI-crawler rules; LOGIN node")
def run(ctx: Context) -> dict:
    cfg = ctx.section("c1.tdm")
    agents = list(cfg.get("ai_agents", ["GPTBot", "CCBot", "Google-Extended", "ClaudeBot"]))
    chk = Checker(agents, float(cfg.get("timeout", 15)), float(cfg.get("pause", 0.3)))
    per_prefix, max_prefixes = int(cfg.get("dois_per_prefix", 3)), int(cfg.get("max_prefixes", 400))
    rows = []
    for sh in input_shards(ctx.cfg):
        for tbl in iter_batches(ctx.cfg, sh, ["paper_id", "oa_doi", "source_doi"], 4096):
            rows += tbl.to_pylist()
    df = pd.DataFrame(rows)
    df["doi"] = [norm_doi(a) or norm_doi(b) for a, b in zip(df["oa_doi"], df["source_doi"], strict=True)]
    df["pmc"] = df["paper_id"].map(lambda p: (PMC.search(str(p)) or [None])[0])
    # the channel the record's text came from: arXiv / PMC copies are under their own (author / OA) licence; the DOI
    # of a journal version points to the publisher, whose reservation is kept separately (`publisher_status`)
    df["source"] = ["arxiv" if g.startswith("arxiv") else "pmc" if m else "doi" if d else "none"
                    for g, d, m in zip(df["_group"], df["doi"], df["pmc"], strict=True)]
    df["prefix"] = df["doi"].map(lambda d: doi_prefix(d) if d else None)

    checks: dict[str, dict] = {}                                # prefix (or "arxiv") -> combined result
    arx = df[df["source"] == "arxiv"]
    if len(arx):
        pid = str(arx["paper_id"].iloc[0])
        checks["arxiv"] = chk.landing(f"https://arxiv.org/abs/{pid}" if re.match(r"^\d{4}\.\d{4,5}", pid)
                                      else "https://arxiv.org/abs/2401.00001")
    pmc = df[df["source"] == "pmc"]
    if len(pmc):            # PubMed Central, the channel the PubMed full texts came from (licences: ws1 licenses)
        checks["pmc"] = combine([chk.landing(f"https://pmc.ncbi.nlm.nih.gov/articles/{i}/")
                                 for i in pmc["pmc"].drop_duplicates().head(per_prefix)])
    prefixes = df["prefix"].dropna().value_counts().index[:max_prefixes]
    for i, pfx in enumerate(prefixes):
        res = [chk.landing("https://doi.org/" + urllib.parse.quote(d, safe="/"))
               for d in df.loc[df["prefix"] == pfx, "doi"].drop_duplicates().head(per_prefix)]
        checks[pfx] = combine(res)
        if (i + 1) % 25 == 0:
            log(f"tdm: {i + 1}/{len(prefixes)} prefixes, {chk.requests} requests")
    key = df["prefix"].where(df["source"] == "doi", df["source"].where(df["source"].isin(["arxiv", "pmc"])))
    out = []
    for r, k in zip(df.to_dict("records"), key, strict=True):          # itertuples would rename the `_` columns
        c = (checks.get(k) if isinstance(k, str) else None) or {}
        pub = checks.get(r["prefix"]) if r["prefix"] and r["prefix"] != "10.48550" else None   # 10.48550 = arXiv DOIs
        out.append({**{f: r[f] for f in ("_file", "_row", "_group", "paper_id", "doi", "source", "prefix")}, "domain": c.get("domain"), "tdmrep": c.get("tdmrep"),
                    "header": c.get("header"), "meta": c.get("meta"), "robots_ai": c.get("robots_ai", ""),
                    "resolved": bool(c.get("resolved")), "status": record_status(c) if c else "unknown",
                    "publisher_status": record_status(pub) if pub else None})
    res = pd.DataFrame(out)
    res.to_parquet(ctx.out / "tdm_reservations.parquet", index=False)
    dom = pd.DataFrame([{"key": k, "domain": v.get("domain"), "resolved": v.get("resolved"), "tdmrep": v.get("tdmrep"),
                         "header": v.get("header"), "meta": v.get("meta"), "robots_ai": v.get("robots_ai"),
                         "status": record_status(v), "records": int((key == k).sum())} for k, v in checks.items()])
    dom.to_csv(ctx.out / "tdm_by_publisher.csv", index=False)
    counts = Counter(res["status"])
    n, known = len(res), len(res) - counts.get("unknown", 0)
    pub_res = res["publisher_status"].isin(["reserved", "robots_ai"])
    summary = {"records": n, "status": dict(counts), "publisher_reserved": int(pub_res.sum()),
               "publisher_reserved_other_channel": int((pub_res & res["source"].isin(["arxiv", "pmc"])).sum()), "checked_share": known / n if n else None,
               "reserved_share": (counts.get("reserved", 0) + counts.get("robots_ai", 0)) / known if known else None,
               "requests": chk.requests, "prefixes": len(prefixes), "ai_agents": agents,
               "tdmrep_files": int(sum(1 for d in chk.domains.values() if d["tdmrep_status"] == 200))}
    ctx.write_json("tdm.json", summary)
    log(f"tdm: {dict(counts)} from {chk.requests} requests")
    pub = dom.sort_values("records", ascending=False).head(40)
    ctx.publish("C1 TDM reservations (DSM Art. 4(3))", [
        {"type": "kpis", "items": [
            {"label": "records checked", "value": summary["checked_share"], "fmt": "pct1",
             "hint": f"{known} of {n}; the rest: no DOI / unresolved / publisher blocks automated access"},
            {"label": "TDM reserved (TDMRep / header / meta)", "value": counts.get("reserved", 0), "fmt": "int",
             "status": "warn" if counts.get("reserved") else "good"},
            {"label": "AI crawlers blocked in robots.txt", "value": counts.get("robots_ai", 0), "fmt": "int",
             "status": "warn" if counts.get("robots_ai") else "good"},
            {"label": "reserved share (of checked)", "value": summary["reserved_share"], "fmt": "pct1"},
            {"label": "journal version reserved, our copy from arXiv / PMC", "value": summary["publisher_reserved_other_channel"],
             "fmt": "int", "hint": "the copy we use is under its own arXiv / PMC licence; queued for legal review"},
            {"label": "publishers with tdmrep.json", "value": summary["tdmrep_files"], "fmt": "int"}]},
        {"type": "bar", "title": "Records by TDM status", "labels": list(counts),
         "series": [{"name": "records", "data": [int(v) for v in counts.values()]}]},
        {"type": "table", "title": "Per publisher (DOI prefix; arXiv)", "hint": "robots_ai = AI / TDM crawlers not allowed on the landing page",
         "columns": [{"key": "key", "label": "prefix"}, {"key": "domain", "label": "domain"}, {"key": "records", "label": "records", "fmt": "int"},
                     {"key": "status", "label": "status"}, {"key": "tdmrep", "label": "TDMRep"}, {"key": "robots_ai", "label": "blocked agents"}],
         "rows": pub.astype(object).where(pub.notna(), None).to_dict("records")}],
        summary=f"{known}/{n} records checked; reserved {counts.get('reserved', 0)}, robots_ai {counts.get('robots_ai', 0)}")
    return summary
