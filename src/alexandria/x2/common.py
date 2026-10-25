"""X2 shared pieces: arXiv listing and LaTeX-HTML text extraction (login node), gpt-oss output parsing, helpers."""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any

from ..runtime import Context

ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
UA = "Alexandria-ICML-2027 research crawler (TIB; contact: gollam.rabby@tib.eu)"


def setting(ctx: Context, key: str, default: Any) -> Any:
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"x2_{key}", default)
    return default


def stage_table(ctx: Context, stage: str, name: str, ws: str = "x2"):
    import pandas as pd
    p = ctx.cfg.stage_dir(ws, stage) / name
    if not p.exists():
        raise FileNotFoundError(f"{ws}/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def arxiv_listing(category: str, after: str, until: str, n: int, delay: float) -> list[dict]:
    """Newest-first arXiv API listing of one category's papers submitted in [after, until] (YYYY-MM-DD)."""
    a, u = after.replace("-", "") + "0000", until.replace("-", "") + "2359"
    q = f"cat:{category}* AND submittedDate:[{a} TO {u}]"
    out, start = [], 0
    while len(out) < n:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": start, "max_results": min(100, n - len(out)), "sortBy": "submittedDate",
             "sortOrder": "descending"})
        root = ET.fromstring(_get(url))
        entries = root.findall("a:entry", ATOM)
        time.sleep(delay)                                         # arXiv API terms: one request every 3 s
        if not entries:
            break
        for e in entries:
            aid = e.find("a:id", ATOM).text.rsplit("/abs/", 1)[-1]
            prim = e.find("arxiv:primary_category", ATOM)
            out.append({"arxiv_id": aid, "title": " ".join((e.find("a:title", ATOM).text or "").split()),
                        "abstract": " ".join((e.find("a:summary", ATOM).text or "").split()),
                        "published": (e.find("a:published", ATOM).text or "")[:10],
                        "primary_category": prim.get("term") if prim is not None else category,
                        "authors": "; ".join(x.text for x in e.findall("a:author/a:name", ATOM) if x.text),
                        "license": (e.find("arxiv:license", ATOM).text if e.find("arxiv:license", ATOM) is not None else None)})
        start += len(entries)
    return out[:n]


class _Text(HTMLParser):
    """Visible article text of an arxiv.org/html page: paragraphs, headings, list items, captions; math as its
    LaTeX alttext; scripts, navigation, headers / footers dropped (the bibliography is cut before parsing)."""
    SKIP = {"script", "style", "nav", "header", "footer", "annotation", "annotation-xml", "button", "svg"}
    BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "figcaption", "caption", "div", "section", "tr", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":                                       # keep the alttext (LaTeX) of inline math
            alt = dict(attrs).get("alttext")
            if alt and not self.skip:
                self.parts.append(f" {alt} ")
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP or tag == "math":
            self.skip = max(self.skip - 1, 0)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(html: str) -> str:
    """Clean text of an arXiv LaTeX-rendered HTML page: the <article> only, bibliography dropped."""
    m = re.search(r"<article\b.*?</article>", html, re.S)
    cut = re.split(r"""<section[^>]*class=["'][^"']*ltx_bibliography""", m.group(0) if m else html, maxsplit=1)[0]
    p = _Text()
    p.feed(cut)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


_LICENSE = re.compile(r"License:\s*(?:<[^>]+>\s*)*([^<\n]{2,80})")


def fetch_html_text(arxiv_id: str) -> tuple[str | None, str | None]:
    """(clean text, licence shown on the page) of arxiv.org/html/<id>; (None, None) without an HTML rendering."""
    try:
        html = _get(f"https://arxiv.org/html/{arxiv_id}").decode("utf-8", "replace")
    except Exception:                                              # noqa: BLE001  (no HTML rendering: skipped)
        return None, None
    lic = _LICENSE.search(html)
    return html_text(html), (" ".join(lic.group(1).split()) if lic else None)


def final_text(text: str | None) -> str:
    """The final-channel text of a gpt-oss (harmony) completion: everything after the last 'final' channel marker."""
    s = text or ""
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in s:
            s = s.rsplit(marker, 1)[1]
    return s.replace("<|return|>", "").replace("<|end|>", "").strip()


def parse_final_json(text: str | None) -> dict | None:
    s = final_text(text)
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
