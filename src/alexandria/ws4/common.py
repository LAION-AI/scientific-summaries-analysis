"""WS4 shared pieces: renderings of a paper's KU, the training-run grid, resources and small helpers.

Representations of one paper (all from the same row; the KU is the 18-field summary with KU-Fixes applied):
  prose       the cleaned KU as a plain-prose summary (markdown sections)       -> condition `summary`
  json        the cleaned KU as a structured record (claims as a list)          -> condition `ku`
  yaml / html / qa_bullets   other surface forms of the same KU              -> condition `diverse` (rotated)
  full        the paper's full text                                             -> condition `full`
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..runtime import Context
from ..schema import PROSE_FIELDS, load_summary

LABELS = {
    "executive_summary": "Summary", "research_context": "Research context",
    "research_question_hypothesis": "Research question and hypotheses", "methodological_details": "Methods",
    "procedures_architectures": "Procedures and architectures", "key_results": "Key results",
    "interpretation_implications": "Interpretation and implications",
    "contradictions_limitations": "Limitations and contradictions", "data_code_availability": "Data and code",
    "robustness_ablation_notes": "Robustness and ablations", "ethical_considerations": "Ethical considerations",
    "key_figures_tables": "Key figures and tables", "three_takeaways": "Three takeaways",
}
QUESTIONS = {f: f"What does the paper report about {lab.lower()}?" for f, lab in LABELS.items()}
FORMATS = ("prose", "json", "yaml", "html", "qa_bullets")
DOC_CONDITIONS = ("summary", "ku", "full", "diverse", "generic", "none")
PAPER_CONDITIONS = ("summary", "ku", "full", "diverse")          # built from the trained papers themselves
_MARKUP = re.compile(r"(\*\*|__|^#+\s*|```(?:json)?)", re.M)


def clean_ku(ku: dict) -> dict:
    """The cleaned KU (KU-Fixes already applied by the caller): WS1 normalizer view, markup stripped."""
    s = load_summary(ku)["summary"]
    out: dict[str, Any] = {}
    for k in ("summary_title", "summary_authors", "field_subfield", "type_of_paper"):
        if s.get(k):
            out[k] = str(s[k]).strip()
    for f in PROSE_FIELDS:
        if s.get(f):
            out[f] = _MARKUP.sub("", str(s[f])).strip()
    claims = [{k: str(v).strip() for k, v in c.items() if v} for c in (s.get("claims") or []) if c.get("description")]
    if claims:
        out["claims"] = claims
    return out


def render(ku: dict, fmt: str) -> str:
    """One surface form of a cleaned KU."""
    title = ku.get("summary_title") or "Untitled study"
    fields = [(f, ku[f]) for f in PROSE_FIELDS if ku.get(f)]
    claims = ku.get("claims") or []
    if fmt == "prose":
        parts = [f"# {title}"] + [f"## {LABELS.get(f, f)}\n{v}" for f, v in fields]
        if claims:
            parts.append("## Claims\n" + "\n".join(
                f"- {c['description']}" + (f" Evidence: {c['supporting_evidence']}" if c.get("supporting_evidence") else "")
                for c in claims))
        return "\n\n".join(parts)
    if fmt == "json":
        return json.dumps(ku, ensure_ascii=False, indent=1)
    if fmt == "yaml":
        import yaml
        return yaml.safe_dump(ku, allow_unicode=True, sort_keys=False, width=120)
    if fmt == "html":
        body = "".join(f"<section><h2>{html.escape(LABELS.get(f, f))}</h2><p>{html.escape(v)}</p></section>"
                       for f, v in fields)
        if claims:
            body += "<section><h2>Claims</h2><ul>" + "".join(f"<li>{html.escape(c['description'])}</li>"
                                                             for c in claims) + "</ul></section>"
        return f"<article><h1>{html.escape(title)}</h1>{body}</article>"
    if fmt == "qa_bullets":
        lines = [f"Study: {title}"] + [f"Q: {QUESTIONS.get(f, f)}\nA: {v}" for f, v in fields]
        lines += [f"Q: What is claimed?\nA: {c['description']}" for c in claims]
        return "\n\n".join(lines)
    raise ValueError(f"unknown format {fmt!r}")


def condition_text(cond: str, paper: dict, epoch: int, formats: list[str]) -> str:
    """The training text of one paper for a paper-based condition (`diverse` rotates formats by epoch and paper)."""
    if cond == "summary":
        return render(paper["ku"], "prose")
    if cond == "ku":
        return render(paper["ku"], "json")
    if cond == "full":
        return paper["full"]
    if cond == "diverse":
        return render(paper["ku"], formats[(epoch + paper["i"]) % len(formats)])
    raise ValueError(cond)


def setting(ctx: Context, key: str, default: Any) -> Any:
    """A debug-profile setting (profiles.debug.ws4_<key>) wins over the default."""
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"ws4_{key}", default)
    return default


def run_grid(ctx: Context) -> list[dict]:
    """Every training run: model x format x condition x seed (SFT), + GRPO and tool-RL runs."""
    t, g, tl = ctx.section("ws4.train"), ctx.section("ws4.grpo"), ctx.section("ws4.tools")
    models = setting(ctx, "models", t.get("models", ["qwen3-4b"]))
    seeds = setting(ctx, "seeds", t.get("seeds", [0]))
    conds = list(ctx.section("ws4").get("conditions", DOC_CONDITIONS))
    runs = []
    for m in models:
        for s in seeds:
            for fmt in t.get("formats", ["doc", "qa"]):
                for c in conds:
                    runs.append({"run": f"{m}__sft-{fmt}__{c}__s{s}", "model": m, "method": f"sft-{fmt}",
                                 "condition": c, "seed": s})
            for c in g.get("conditions", []):
                runs.append({"run": f"{m}__grpo__{c}__s{s}", "model": m, "method": "grpo", "condition": c, "seed": s})
            if tl.get("enabled"):
                runs.append({"run": f"{m}__tools__ku__s{s}", "model": m, "method": "tools", "condition": "ku",
                             "seed": s})
    return runs


def resource_path(ctx: Context, name: str) -> list[Path]:
    """Local files of a `ws4_resources` entry (downloaded by `alexandria fetch-ws4` into site.hf_home)."""
    from huggingface_hub import hf_hub_download
    r = ctx.cfg.raw["ws4_resources"][name]
    return [Path(hf_hub_download(r["repo"], f, repo_type="dataset", cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                                 local_files_only=True)) for f in r["files"]]


def load_resource(ctx: Context, name: str, max_rows: int | None = None) -> pd.DataFrame:
    import pyarrow.parquet as pq
    frames, n = [], 0
    for p in resource_path(ctx, name):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=4096):
            df = b.to_pandas()
            frames.append(df)
            n += len(df)
            if max_rows and n >= max_rows:
                return pd.concat(frames, ignore_index=True).head(max_rows)
    return pd.concat(frames, ignore_index=True)


def stage_table(ctx: Context, stage: str, name: str) -> pd.DataFrame:
    p = ctx.cfg.stage_dir("ws4", stage) / name
    if not p.exists():
        raise FileNotFoundError(f"ws4/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


_W = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _W.findall((text or "").lower())


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
