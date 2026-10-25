"""Stage `contexts` (WS2): the context each arm gives the answerer, per paper, with token counts.

Arms (common.ARMS): closed_book (none), full_text (first `full_text_chars` characters of the source),
full_text_matched (the source cut to the same token count as the paper's clean KU: compression control),
ku_raw (the paper's one KU, i.e. its 18 summary fields, as released), ku_clean (the KU with KU-Fixes applied, then the
WS1 normalizer view). Both KU arms come from the `ku` stage (one KU per paper).
Every context is capped at `max_context_tokens` (recorded). Tokens are counted with one reference tokenizer
(ws2.contexts.tokenizer) so arms are comparable; the answer stage re-truncates per model if needed.
Per question it also stores the lexical overlap between question and each context (share of question 5-grams
found in the context): a covariate for the "questions favour the arm they were written from" bias.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from ..data import fetch_records
from ..qa import ngram_overlap
from ..runtime import Context, stage
from ..ws1.faith.common import source_text
from .common import ARMS, BASE_ARMS, ablation_texts, load_stage_table, render_ku_clean, render_ku_raw


def reference_tokenizer(cfg, name: str):
    from transformers import AutoTokenizer

    from ..llm import model_spec
    os.environ.setdefault("HF_HOME", str(cfg.site.hf_home))
    spec = model_spec(cfg, name)
    return AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision"))


def cut_tokens(tok, text: str, n: int) -> tuple[str, int]:
    ids = tok(text or "", add_special_tokens=False)["input_ids"]
    return (text, len(ids)) if len(ids) <= n else (tok.decode(ids[:n]), n)


@stage("ws2", "contexts", needs=("papers", "ku", "questions"), config_keys=("ws2.contexts",),
       help="per paper and arm: the answerer's context (full text, matched full text, raw / clean KU)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.contexts")
    papers = load_stage_table(ctx, "papers", "papers.parquet")
    kus = load_stage_table(ctx, "ku", "kus.parquet").set_index("paper_key")
    tok = reference_tokenizer(ctx.cfg, cfg.get("tokenizer", "qwen2.5-7b"))
    cap = int(cfg.get("max_context_tokens", 16000))
    full_chars = int(cfg.get("full_text_chars", 30000))
    recs = fetch_records(ctx.cfg, list(zip(papers["file"], papers["row"], strict=True)),
                         ["paper_id", "text_raw", "text_sanitized"])
    acfg = ctx.section("ws2.ablation")
    ablate = bool(acfg.get("enabled", True))
    gen_p = ctx.cfg.stage_dir("ws2", "ablation-gen") / "generated.parquet"
    gen = pd.read_parquet(gen_p).set_index("paper_key") if ablate and gen_p.exists() else None
    rows = []
    for p in papers.itertuples():
        rec = recs.get((p.file, int(p.row))) or {}
        src = source_text(rec)
        ku = kus.loc[p.paper_key]
        clean, issues = render_ku_clean(json.loads(ku["ku_fixed"]))
        texts = {"closed_book": "", "full_text": src[:full_chars], "ku_raw": render_ku_raw(json.loads(ku["ku_raw"])),
                 "ku_clean": clean}
        _, n_clean = cut_tokens(tok, clean, cap)
        texts["full_text_matched"], _ = cut_tokens(tok, src, max(n_clean, 1))
        if ablate:                                     # structure ablation (common.ABLATION_GROUPS, ablation-gen)
            texts.update(ablation_texts(json.loads(ku["ku_fixed"]), seed=int(acfg.get("seed", 0)) + p.Index))
            if gen is not None and p.paper_key in gen.index and gen.at[p.paper_key, "ku_tokens"] > 0:
                texts["ku_local"] = gen.at[p.paper_key, "ku_text"]
                texts["prose_local"] = gen.at[p.paper_key, "prose_text"]
        for arm in [a for a in ARMS if a in texts]:
            t, n = cut_tokens(tok, texts[arm], cap)
            rows.append({"paper_key": p.paper_key, "arm": arm, "context": t, "tokens": n, "chars": len(t),
                         "capped": n >= cap and arm != "closed_book",
                         "normalizer_issues": len(issues) if arm == "ku_clean" else None})
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.out / "contexts.parquet", index=False)

    try:
        qs = load_stage_table(ctx, "questions", "questions.parquet")
        by = {(r.paper_key, r.arm): r.context for r in df.itertuples()}
        ov = [{"qid": q.qid, "arm": arm, "overlap5": ngram_overlap(q.question, by.get((q.paper_key, arm), ""))}
              for q in qs.itertuples() for arm in BASE_ARMS if arm != "closed_book"]
        pd.DataFrame(ov).to_parquet(ctx.out / "question_overlap.parquet", index=False)
        ov_tbl = pd.DataFrame(ov).groupby("arm")["overlap5"].mean().reindex([a for a in ARMS if a != "closed_book"])
    except FileNotFoundError:
        ov_tbl = pd.Series(dtype=float)
    present = [a for a in ARMS if a in set(df["arm"])]
    tok_tbl = df.groupby("arm")["tokens"].describe(percentiles=[.5]).reindex(present)
    src_tok = df[df["arm"] == "full_text"].set_index("paper_key")["tokens"]
    rows_tbl = []
    for arm in present:
        t = df[df["arm"] == arm].set_index("paper_key")["tokens"]
        rows_tbl.append({"arm": arm, "what": ARMS[arm], "median_tokens": float(tok_tbl.loc[arm, "50%"]),
                         "mean_tokens": float(tok_tbl.loc[arm, "mean"]),
                         "compression": float((t / src_tok.reindex(t.index)).median()) if arm != "closed_book" else 0.0,
                         "capped": int(df[(df["arm"] == arm)]["capped"].sum()),
                         "question_overlap": float(ov_tbl.get(arm, float("nan"))) if arm != "closed_book" else None})
    ctx.write_json("context_stats.json", rows_tbl)
    ctx.publish("WS2 contexts per arm", [
        {"type": "table", "title": "Context size per arm",
         "hint": f"tokens with the {cfg.get('tokenizer', 'qwen2.5-7b')} tokenizer; compression = arm tokens / "
                 f"full-text tokens (median over papers); question overlap = share of question 5-grams in the context",
         "columns": [{"key": "arm", "label": "arm"}, {"key": "what", "label": "context"},
                     {"key": "median_tokens", "label": "median tokens", "fmt": "int"},
                     {"key": "compression", "label": "vs full text", "fmt": "float"},
                     {"key": "capped", "label": f"capped at {cap}", "fmt": "int"},
                     {"key": "question_overlap", "label": "question overlap", "fmt": "pct1"}], "rows": rows_tbl},
        {"type": "bar", "title": "Median context tokens per arm", "fmt": "int", "horizontal": True,
         "labels": [r["arm"] for r in rows_tbl], "series": [{"name": "tokens", "data": [r["median_tokens"] for r in rows_tbl]}]},
    ], summary=f"{len(papers)} papers x {len(present)} arms")
    return {"papers": len(papers), "arms": len(present)}
