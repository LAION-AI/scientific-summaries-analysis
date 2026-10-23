"""Stage `abstract-support`: how much of the long methods / results fields can the summarizer's input support?

~76% of the corpus was summarized from metadata + abstract (NeurIPS text), yet the prompt asks for 450-550-word
methods and results fields. Per input type (metadata_only / abstract_only / short_text / full_text, from `features`)
and per field in ws1b.abstract_support.fields this stage compares:
  * words written (features words__<field>) and the share of summary content words found in the input
    (features content_support, paper-level)
  * number precision and entity precision (from `lexical` parts, if present)
  * MiniCheck supported-sentence share (from `grounding` parts, if present)
  * LLM-judge unsupported + contradicted share (from `judge` parts, if present)
"Supportable words" = words x support share, where support share is the MiniCheck supported-sentence share when
available, else number precision, else content_support (the column `support_source` says which was used).

Recommendation (a heuristic, not a config knob, so it reads the same in the paper):
  support >= 0.80 keep · 0.60-0.80 flag (warn users) · 0.40-0.60 shorten to the supportable words · < 0.40 drop
  the field for that input type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...runtime import Context, stage
from .common import features_table, paper_key

INPUT_ORDER = ["metadata_only", "abstract_only", "short_text", "full_text", "unknown"]


def recommend(support: float | None) -> str:
    if support is None or (isinstance(support, float) and np.isnan(support)):
        return "n/a"
    if support >= 0.8:
        return "keep"
    if support >= 0.6:
        return "flag"
    if support >= 0.4:
        return "shorten"
    return "drop"


def _parts_df(ctx: Context, stage_name: str) -> pd.DataFrame:
    import duckdb
    d = ctx.cfg.stage_dir("ws1", stage_name) / "parts"
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    return duckdb.sql(f"select * from read_parquet({[str(p) for p in files]!r}, union_by_name=true)").df()


@stage("ws1", "abstract-support", needs=("features",), config_keys=("ws1b.abstract_support",),
       help="by input type: what share of long methods/results fields the input can support; keep/flag/shorten/drop")
def run(ctx: Context) -> dict:
    fields = list(ctx.section("ws1b.abstract_support").get("fields",
                  ["methodological_details", "procedures_architectures", "key_results"]))
    feats = features_table(ctx, ["file", "row", "input_type", "content_support", "has_text"]
                           + [f"words__{f}" for f in fields])
    feats["input_type"] = feats["input_type"].fillna("unknown")
    feats["paper_key"] = [paper_key(f, r) for f, r in zip(feats["file"], feats["row"], strict=True)]
    lex = _parts_df(ctx, "lexical")
    if not lex.empty:
        feats = feats.merge(lex, on=["file", "row"], how="left", suffixes=("", "_lex"))
    grounding = _parts_df(ctx, "grounding")
    judge = _parts_df(ctx, "judge")

    rows = []
    for it in [i for i in INPUT_ORDER if i in set(feats["input_type"])]:
        g = feats[feats["input_type"] == it]
        for f in fields:
            r: dict = {"input_type": it, "field": f, "papers": len(g),
                       "words_mean": float(g[f"words__{f}"].mean()) if f"words__{f}" in g else None,
                       "content_support": float(g["content_support"].mean()) if g["content_support"].notna().any() else None}
            for kind in ("num", "ent"):
                n_col, ok_col = f"{kind}_n__{f}", f"{kind}_ok__{f}"
                if n_col in g and g[n_col].sum() > 0:
                    r[f"{kind}_precision"] = float(g[ok_col].sum() / g[n_col].sum())
                else:
                    r[f"{kind}_precision"] = None
            if not grounding.empty and "input_type" in grounding:
                gg = grounding[(grounding["input_type"] == it) & (grounding["field"] == f)]
                r["minicheck_supported"] = float(gg["supported"].mean()) if len(gg) else None
                r["minicheck_sentences"] = int(len(gg))
            if not judge.empty and "label" in judge:
                jj = judge[(judge["input_type"] == it) & (judge["field"] == f)]
                r["judge_unsupported"] = float(jj["label"].isin(["unsupported", "contradicted"]).mean()) if len(jj) else None
            for key, src in (("minicheck_supported", "minicheck"), ("num_precision", "numbers"),
                             ("content_support", "content words")):
                if r.get(key) is not None:
                    r["support"], r["support_source"] = r[key], src
                    break
            else:
                r["support"], r["support_source"] = None, "none"
            r["supportable_words"] = (r["words_mean"] * r["support"]) if r["support"] is not None and r["words_mean"] else None
            r["recommendation"] = recommend(r["support"])
            rows.append(r)
    df = pd.DataFrame(rows)
    ctx.write_json("abstract_support.json", rows)
    mix = feats["input_type"].value_counts(normalize=True).reindex(INPUT_ORDER).dropna()

    cols = [{"key": "input_type", "label": "input"}, {"key": "field", "label": "field"},
            {"key": "papers", "label": "papers", "fmt": "int"}, {"key": "words_mean", "label": "words", "fmt": "int"},
            {"key": "content_support", "label": "content words in input", "fmt": "pct1"},
            {"key": "num_precision", "label": "number prec.", "fmt": "pct1"},
            {"key": "ent_precision", "label": "entity prec.", "fmt": "pct1"},
            {"key": "minicheck_supported", "label": "MiniCheck supported", "fmt": "pct1"},
            {"key": "judge_unsupported", "label": "judge unsupported", "fmt": "pct1"},
            {"key": "supportable_words", "label": "supportable words", "fmt": "int"},
            {"key": "support_source", "label": "support from"}, {"key": "recommendation", "label": "action"}]
    panels = [
        {"type": "kpis", "items": [{"label": f"{k} papers", "value": float(v), "fmt": "pct1"} for k, v in mix.items()]},
        {"type": "bar", "title": "Words written vs words the input can support", "fmt": "int",
         "hint": "mean per paper; supportable = words x support share",
         "labels": [f"{r.input_type} · {r.field}" for r in df.itertuples()],
         "series": [{"name": "words", "data": df["words_mean"].fillna(0).tolist()},
                    {"name": "supportable", "data": df["supportable_words"].fillna(0).tolist()}],
         "horizontal": True},
        {"type": "table", "title": "Support by input type and field", "columns": [c for c in cols if c["key"] in df],
         "rows": df.replace({np.nan: None}).to_dict("records")},
        {"type": "markdown", "text": "**Action rule**: support ≥ 0.80 keep · 0.60–0.80 flag · 0.40–0.60 shorten to the "
                                     "supportable words · < 0.40 drop the field for that input type. Support = MiniCheck "
                                     "share when available, else number precision, else content-word share."},
    ]
    ctx.publish("Abstract-only inputs vs long fields", panels,
                summary=f"{len(df)} (input type, field) cells; sources: lexical={not lex.empty}, "
                        f"grounding={not grounding.empty}, judge={not judge.empty}")
    return {"cells": len(df), "lexical": not lex.empty, "grounding": not grounding.empty, "judge": not judge.empty}
