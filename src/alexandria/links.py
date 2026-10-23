"""Links from a record to the paper's public pages, built from the dataset's link rules (configs/datasets/*.yaml).

    rules = compile_rules(cfg.dataset.links)
    paper_links(record, rules)  -> [("arXiv", "https://arxiv.org/abs/2407.18923"), ("DOI", "https://doi.org/...")]
    markdown_links(json_text)   -> "[arXiv](https://...) · [DOI](https://...)"   (dashboard tables, PDF)

`features` stores the links of every paper as a JSON list in its `links` column, so every later stage can show them
without reading the release again.
"""
from __future__ import annotations

import json
import re
from typing import Any


def compile_rules(rules) -> list[tuple[str, str, re.Pattern, str]]:
    out = []
    for r in rules or []:
        d = r if isinstance(r, dict) else r.model_dump()
        out.append((d["name"], d["column"], re.compile(d.get("pattern") or ".+"), d["url"]))
    return out


def paper_links(rec: dict[str, Any], rules) -> list[tuple[str, str]]:
    """Every rule whose column value matches; one link per (name, url), in rule order."""
    seen, out = set(), []
    for name, col, pat, url in rules:
        v = rec.get(col)
        if v is None or str(v).strip() == "":
            continue
        m = pat.search(str(v).strip())
        if not m:
            continue
        link = url.format(m.group(0), *m.groups())
        if (name, link) not in seen and not any(n == name for n, _ in out):   # first match per link name
            seen.add((name, link))
            out.append((name, link))
    return out


def to_json(links: list[tuple[str, str]]) -> str | None:
    return json.dumps(links) if links else None


def from_json(text: str | None) -> list[tuple[str, str]]:
    if not text:
        return []
    try:
        return [tuple(x) for x in json.loads(text)]
    except (json.JSONDecodeError, TypeError):
        return []


def markdown_links(text: str | None) -> str:
    return " · ".join(f"[{n}]({u})" for n, u in from_json(text))


def first_url(text: str | None) -> str | None:
    links = from_json(text)
    return links[0][1] if links else None
