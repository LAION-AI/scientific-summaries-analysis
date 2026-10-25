"""Stages `eval`, `grade` and `results` of X2.

eval     The WS3 harness (alexandria.ws3.agent._run_model: same prompts, constrained JSON actions, limits, cache) on
         the post-cutoff questions, with the backend pointed at the incrementally updated index: A0 (model alone:
         it cannot know these papers), R (plain RAG), F (agent over full text), S (agent over KUs). One array task
         per answer model.
grade    SimpleQA judge (as WS2 / WS3) + exact match / token F1.
results  Accuracy per arm x model with 95% bootstrap CIs resampling papers, paired Δ vs A0 (Holm), gold-paper
         retrieved rate; headline: the skill's gain when the model cannot know the answer (WS3-H3).
"""
from __future__ import annotations

import pandas as pd

from ..llm import embedder_spec, model_spec, with_model
from ..qa import exact_match, grade_batch, is_abstention, token_f1
from ..runtime import Context, log, stage
from ..stats import cluster_bootstrap, holm, paired_difference
from ..ws3.agent import _run_model
from .common import stage_table


def items(ctx: Context) -> list[dict]:
    q = stage_table(ctx, "questions", "questions.parquet")
    return [{"bench": "x2-postcutoff", "item_id": r.qid, "question": r.question, "answer": r.answer,
             "answer_type": "short", "options": [], "cluster": r.paper_key, "gold_paper_key": r.paper_key,
             "category": r.qtype, "judge": "simpleqa", "abstain_option": None} for r in q.itertuples()]


@stage("x2", "eval", kind="map", needs=("index", "questions"), resources="gpu", env="llm", config_keys=("x2.eval",),
       help="WS3 harness arms (A0, R, F, S) on the post-cutoff questions over the updated index; one task per model")
def run_eval(ctx: Context) -> dict:
    cfg = ctx.section("x2.eval")
    its = items(ctx)
    arms = {a: cfg["arm_defs"][a] for a in cfg.get("arms", ["A0", "R", "F", "S"])}
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = []
    for name in list(cfg.get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]:
        dst = parts / f"{name}.parquet"
        if dst.exists() and not ctx.force:
            continue
        payload = {"cfg": cfg, "items": its, "arms": arms, "oracle": {},
                   "index_dir": str(ctx.cfg.stage_dir("x2", "index")),
                   "embedder": embedder_spec(ctx.cfg, ctx.section("x2.index").get("embedder", "bge-m3")),
                   "cache": str(ctx.out / "cache" / f"{name}.sqlite"), "dense": bool(cfg.get("dense", True))}
        log(f"eval: {name}: {len(its)} questions x {len(arms)} arms")
        rows = with_model(model_spec(ctx.cfg, name), _run_model, payload,
                          gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.88)))
        pd.DataFrame(rows).to_parquet(dst, index=False)
        done.append(name)
    ctx.publish("X2 evaluation runs", [{"type": "kpis", "items": [
        {"label": "models run", "value": len(list(parts.glob("*.parquet"))), "fmt": "int"},
        {"label": "questions", "value": len(its), "fmt": "int"}, {"label": "arms", "value": ", ".join(arms)}]}],
        summary=f"{len(list(parts.glob('*.parquet')))} models")
    return {"models_done": done}


def _grade(llm, tok, spec, its: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, its)


@stage("x2", "grade", needs=("eval",), resources="gpu", env="llm", config_keys=("x2.grade",),
       help="SimpleQA judge over every post-cutoff answer (+ exact match, token F1)")
def run_grade(ctx: Context) -> dict:
    parts = sorted((ctx.cfg.stage_dir("x2", "eval") / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("x2/eval has no parts")
    a = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    q = stage_table(ctx, "questions", "questions.parquet")[["qid", "paper_key", "question", "answer"]]
    df = a.merge(q.rename(columns={"qid": "item_id", "answer": "gold"}), on="item_id", how="left")
    judge = model_spec(ctx.cfg, ctx.section("x2.grade").get("judge", "qwen3-8b"))
    df["grade"] = with_model(judge, _grade, [{"question": r.question, "gold": r.gold, "predicted": r.answer or ""}
                                             for r in df.itertuples()], gpu_memory_utilization=0.85)
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["abstained"] = [int(is_abstention(x)) for x in df["answer"]]
    df["em"] = [int(exact_match(p, g)) for p, g in zip(df["answer"], df["gold"], strict=True)]
    df["f1"] = [token_f1(p, g) for p, g in zip(df["answer"], df["gold"], strict=True)]
    df.drop(columns=["question", "trace"], errors="ignore").to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("X2 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"}, {"label": "judge", "value": judge["name"]},
        {"label": "CORRECT overall", "value": float(df["correct"].mean()), "fmt": "pct1"}]}], summary=f"{len(df)} answers")
    return {"graded": len(df)}


@stage("x2", "results", needs=("grade",), config_keys=("x2.results",),
       help="accuracy per arm x model (paper-clustered CIs), Δ vs A0 (Holm), gold-paper retrieved; WS3-H3 headline")
def run_results(ctx: Context) -> dict:
    g = pd.read_parquet(ctx.cfg.stage_dir("x2", "grade") / "graded.parquet")
    g["cluster"] = g["paper_key"]
    g["gold_retrieved"] = pd.to_numeric(g["gold_retrieved"], errors="coerce")
    per = g.groupby(["model", "arm", "item_id", "cluster"], as_index=False).agg(
        correct=("correct", "mean"), abstained=("abstained", "mean"), gold=("gold_retrieved", "mean"),
        calls=("tool_calls", "mean"))
    tbl, boot = cluster_bootstrap(per, "correct", "cluster", ["model", "arm"],
                                  n_boot=int(ctx.section("x2.results").get("n_boot", 2000)))
    extra = per.groupby(["model", "arm"]).agg(gold=("gold", "mean"), abstained=("abstained", "mean"),
                                              calls=("calls", "mean")).reset_index()
    tbl = tbl.merge(extra, on=["model", "arm"])
    pt = {(r.model, r.arm): r.mean for r in tbl.itertuples()}
    cmp = [{"model": m, "arm": a, **paired_difference(boot, (m, a), (m, "A0"), pt[(m, a)], pt[(m, "A0")])}
           for (m, a) in pt if a != "A0" and (m, "A0") in pt]
    cmp = pd.DataFrame(cmp)
    if len(cmp):
        cmp["p_holm"] = holm(cmp["p"].tolist())
    tbl.to_csv(ctx.out / "accuracy.csv", index=False)
    cmp.to_csv(ctx.out / "comparisons.csv", index=False)
    best = cmp.groupby("arm")["diff"].mean().sort_values(ascending=False) if len(cmp) else pd.Series(dtype=float)
    a0 = tbl[tbl["arm"] == "A0"]["mean"].mean()
    kp = [{"label": "model alone (A0), mean over models", "value": float(a0), "fmt": "pct1",
           "hint": "post-cutoff: the models cannot know these papers"}]
    for arm, d in best.head(2).items():
        kp.append({"label": f"Δ {arm} vs A0 (mean over models)", "value": float(d), "fmt": "pct1"})
    kp.append({"label": "questions", "value": int(g["item_id"].nunique()), "fmt": "int"})
    ctx.publish("X2 results: does the skill help when the model cannot know?", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "table", "title": "Accuracy per model and arm (95% CI resampling papers)", "rows": tbl.to_dict("records"),
         "columns": [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "mean", "label": "accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "gold", "label": "gold paper retrieved", "fmt": "pct1"},
                     {"key": "abstained", "label": "abstained", "fmt": "pct1"}, {"key": "calls", "label": "tool calls", "fmt": "float"}]},
        {"type": "table", "title": "Paired differences vs the model alone (A0)", "rows": cmp.to_dict("records"),
         "columns": [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "p_holm", "label": "p (Holm)", "fmt": "float3"}]},
    ], summary=f"A0 {a0:.1%}; best Δ " + (f"{best.index[0]} {best.iloc[0]:+.1%}" if len(best) else "n/a"))
    return {"rows": len(tbl)}

