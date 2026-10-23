"""Stages `answer` and `grade` (WS2, GPU).

answer  Every answerer model (ws2.answer.models; several families) answers every question under every arm, with
        the same system prompt (qa.ANSWER_SYSTEM) and only the context block differing. Context first, question last,
        so all questions of one paper and arm share a cached prefix. temperature 0 (deterministic) unless seeds and a
        temperature are set. One process per model; with a SLURM array (`--array N`) task i answers models[i::N].
        Output: parts/<model>.parquet (resumable: a finished model is skipped unless --force).
grade   One fixed judge (ws2.grade.judge) grades every answer against the gold answer with the SimpleQA protocol
        (CORRECT / INCORRECT / NOT_ATTEMPTED; qa.GRADER_SYSTEM), plus model-free normalised exact match and token F1.
"""
from __future__ import annotations

import time

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..qa import ANSWER_SYSTEM, answer_user, exact_match, grade_batch, is_abstention, token_f1
from ..runtime import Context, log, stage
from .common import ARMS, load_parts, load_stage_table


def _answer(llm, tok, spec, items: list[dict], cfg: dict) -> list[dict]:
    budget = int(spec.get("max_model_len", 32768)) - int(cfg.get("max_tokens", 64)) - 512
    prompts, meta = [], []
    for it in items:
        ctx_text, n, cut = truncate_tokens(tok, it["context"], budget) if it["context"] else ("", 0, False)
        prompts.append(chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(it["question"], ctx_text)))
        meta.append({"context_tokens_model": n, "context_cut_for_model": cut})
    rows = []
    for seed in cfg.get("seeds", [0]):
        t0 = time.time()
        outs = llm.generate(prompts, json_params(None, int(cfg.get("max_tokens", 64)),
                                                 float(cfg.get("temperature", 0.0)), seed=int(seed)), use_tqdm=False)
        dt = time.time() - t0
        for it, m, o in zip(items, meta, outs, strict=True):
            text = o.outputs[0].text.strip() if o.outputs else ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            rows.append({"qid": it["qid"], "arm": it["arm"], "model": spec["name"], "seed": int(seed),
                         "prediction": text.split("\n")[0][:300], **m,
                         "prompt_tokens": len(o.prompt_token_ids or []),
                         "output_tokens": len(o.outputs[0].token_ids) if o.outputs else 0,
                         "batch_seconds": dt})
    return rows


@stage("ws2", "answer", kind="map", needs=("questions", "contexts"), resources="gpu", env="llm",
       config_keys=("ws2.answer",),
       help="every answerer model answers every question under every arm (same prompt; one process per model)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws2.answer")
    qs = load_stage_table(ctx, "questions", "questions.parquet")
    cx = load_stage_table(ctx, "contexts", "contexts.parquet")
    arms = [a for a in cfg.get("arms", list(ARMS)) if a in ARMS]
    by = {(r.paper_key, r.arm): r.context for r in cx.itertuples()}
    items = [{"qid": q.qid, "arm": a, "question": q.question, "context": by.get((q.paper_key, a), "")}
             for q in qs.itertuples() for a in arms]
    models = list(cfg.get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = []
    for name in models:
        dst = parts / f"{name}.parquet"
        if dst.exists() and not ctx.force:
            log(f"answer: {name} already done")
            continue
        spec = model_spec(ctx.cfg, name)
        log(f"answer: {name}: {len(items)} prompts ({len(qs)} questions x {len(arms)} arms)")
        rows = with_model(spec, _answer, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
        pd.DataFrame(rows).to_parquet(dst, index=False)
        done.append(name)
    ctx.publish("WS2 answers", [{"type": "markdown", "text":
                f"Answered by: {', '.join(sorted(p.stem for p in parts.glob('*.parquet')))}. "
                f"{len(qs)} questions x {len(arms)} arms per model. Accuracy is in `results`."}],
                summary=f"{len(list(parts.glob('*.parquet')))} models answered")
    return {"models_done": done, "prompts_per_model": len(items)}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("ws2", "grade", needs=("answer",), resources="gpu", env="llm", config_keys=("ws2.grade",),
       help="SimpleQA-style LLM judge (CORRECT / INCORRECT / NOT_ATTEMPTED) + exact match + token F1")
def run_grade(ctx: Context) -> dict:
    cfg = ctx.section("ws2.grade")
    ans = load_parts(ctx, "answer")
    if ans.empty:
        raise FileNotFoundError("ws2/answer has no parts; run `answer` first")
    qs = load_stage_table(ctx, "questions", "questions.parquet")[["qid", "question", "answer"]]
    df = ans.merge(qs, on="qid", how="left")
    judge = model_spec(ctx.cfg, cfg.get("judge", "qwen3-8b"))
    items = [{"question": r.question, "gold": r.answer, "predicted": r.prediction} for r in df.itertuples()]
    log(f"grade: {len(items)} answers with {judge['name']}")
    df["grade"] = with_model(judge, _grade, items, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["attempted"] = (df["grade"] != "NOT_ATTEMPTED").astype(int)
    df["abstained"] = [int(is_abstention(p)) for p in df["prediction"]]
    df["em"] = [int(exact_match(p, g)) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df["f1"] = [token_f1(p, g) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df["judge"] = judge["name"]
    df.drop(columns=["question", "answer"]).to_parquet(ctx.out / "graded.parquet", index=False)
    agree = float(((df["em"] == 1) == (df["correct"] == 1)).mean())
    ctx.publish("WS2 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"},
        {"label": "judge", "value": judge["name"]},
        {"label": "judge CORRECT", "value": float(df["correct"].mean()), "fmt": "pct1"},
        {"label": "exact match", "value": float(df["em"].mean()), "fmt": "pct1"},
        {"label": "judge vs exact-match agreement", "value": agree, "fmt": "pct1",
         "hint": "model-free check of the judge"}]}], summary=f"{len(df)} answers graded")
    return {"graded": len(df), "judge_correct": float(df["correct"].mean()), "em": float(df["em"].mean())}
