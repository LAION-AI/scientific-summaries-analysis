"""Stages `mcq`, `judge`, `answer` and `results` (WS5): how good and how cheap is each system's KU?

mcq      Coverage questions (WS2 protocol, multiple choice): the generator reads the test paper's FULL TEXT and
         writes 4-option questions with an exact evidence quote; kept if the quote is in the paper and the answer
         is one option. Independent of every system's KU.
judge    Faithfulness: the KU's claims and key-result sentences are checked against the paper by a judge from a
         third family (not the teacher's, not the students'): SUPPORTED / NOT_SUPPORTED / CONTRADICTED each.
answer   Coverage: an answerer from yet another family answers each MCQ from the system's KU only (no paper);
         `closed_book` (no context) is the floor. Coverage = accuracy.
results  Per system: schema validity (WS1), faithfulness, coverage (95% CI resampling papers), verbatim overlap
         (must not rise with distillation), tokens per paper, papers per GPU-hour; held-out disciplines separately;
         learning curves (examples -> coverage / faithfulness) per student and method.
"""
from __future__ import annotations

import re

import pandas as pd

from ..figures import plt, save, style
from ..llm import chat_prompt, json_params, model_spec, parse_json, truncate_tokens, with_model
from ..runtime import Context, log, stage
from ..stats import cluster_bootstrap
from ..ws4.common import clean_ku, render
from .common import overlap, parse_ku, schema_check, stage_table
from .students import runs

MCQ_SYSTEM = """You write multiple-choice questions that test whether a reader knows the specific content of one scientific paper.
Each question: about a specific result, method, setting, comparison or limitation of THIS paper; self-contained (name the system or method, never "this paper"); four options, exactly one correct, distractors plausible; "evidence" is an exact quote (<= 40 words) from the paper that shows the answer. Answer with JSON only."""
JUDGE_SYSTEM = """You check statements from a summary against the scientific paper they summarise.
For each numbered statement answer SUPPORTED (the paper states or directly implies it), NOT_SUPPORTED (the paper does not say this) or CONTRADICTED (the paper says otherwise). Judge only against the paper text given. Answer with JSON only."""
ANSWER_SYSTEM = "Answer the multiple-choice question using only the summary given. Reply with the letter of the correct option."


def _mcq_schema(k: int) -> dict:
    return {"type": "object", "properties": {"questions": {"type": "array", "minItems": 1, "maxItems": k, "items": {
        "type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "minItems": 4,
                                                                                    "maxItems": 4, "items": {"type": "string"}},
                                         "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                                         "evidence": {"type": "string"}},
        "required": ["question", "options", "answer", "evidence"], "additionalProperties": False}}},
        "required": ["questions"], "additionalProperties": False}


def _mcq_gen(llm, tok, spec, items: list[dict], cfg: dict) -> list[list[dict]]:
    k = int(cfg.get("generate_per_paper", 10))
    prompts = [chat_prompt(tok, spec, MCQ_SYSTEM, "PAPER:\n<<<\n" + truncate_tokens(tok, it["source"], int(cfg.get("max_source_tokens", 9000)))[0]
                           + f"\n>>>\n\nWrite {k} questions.") for it in items]
    outs = llm.generate(prompts, json_params(_mcq_schema(k), 3000), use_tqdm=False)
    return [((parse_json(o.outputs[0].text if o.outputs else "") or {}).get("questions") or []) for o in outs]


@stage("ws5", "mcq", needs=("papers",), resources="gpu", env="llm", config_keys=("ws5.mcq",),
       help="coverage MCQs from each test paper's full text (WS2 protocol, evidence-checked)")
def run_mcq(ctx: Context) -> dict:
    from rapidfuzz import fuzz
    cfg = ctx.section("ws5.mcq")
    test = stage_table(ctx, "papers", "papers.parquet")
    test = test[test["split"] == "test"]
    items = [{"paper_key": p.paper_key, "source": p.source} for p in test.itertuples()]
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    outs = with_model(gen, _mcq_gen, items, cfg, gpu_memory_utilization=0.85)
    rows = []
    for it, qs in zip(items, outs, strict=True):
        src = " ".join(it["source"].lower().split())
        kept = 0
        for j, q in enumerate(qs):
            ok = (isinstance(q, dict) and len(q.get("options") or []) == 4 and len(set(q["options"])) == 4
                  and len(q.get("evidence") or "") >= 10 and not re.search(r"\bthis (paper|study)\b", q.get("question", ""), re.I)
                  and fuzz.partial_ratio(" ".join(q["evidence"].lower().split()), src) >= 85)
            if ok and kept < int(cfg.get("per_paper", 6)):
                rows.append({"qid": f"{it['paper_key']}:m{j}", "paper_key": it["paper_key"], "question": q["question"],
                             "options": q["options"], "answer": q["answer"]})
                kept += 1
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.out / "mcq.parquet", index=False)
    ctx.publish("WS5 coverage questions", [{"type": "kpis", "items": [
        {"label": "MCQs", "value": len(df), "fmt": "int"}, {"label": "test papers with MCQs", "value": int(df["paper_key"].nunique()), "fmt": "int"},
        {"label": "generator", "value": gen["name"]}]}], summary=f"{len(df)} MCQs")
    return {"mcq": len(df)}


def all_kus(ctx: Context) -> pd.DataFrame:
    parts = sorted((ctx.cfg.stage_dir("ws5", "generate") / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("ws5/generate has no parts; run generate first")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def statements(ku: dict | None, n: int) -> list[str]:
    if not ku:
        return []
    out = [str(c.get("description", "")).strip() for c in (ku.get("claims") or []) if isinstance(c, dict)]
    out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(ku.get("key_results") or "")) if len(s.split()) >= 5]
    return [s for s in out if s][:n]


def _judge(llm, tok, spec, items: list[dict], cfg: dict) -> list[list[str]]:
    schema = lambda n: {"type": "object", "properties": {"verdicts": {"type": "array", "minItems": n, "maxItems": n, "items": {
        "type": "string", "enum": ["SUPPORTED", "NOT_SUPPORTED", "CONTRADICTED"]}}}, "required": ["verdicts"],
        "additionalProperties": False}
    todo = [it for it in items if it["statements"]]                    # one request per (system, paper)
    prompts, params = [], []
    for it in todo:
        src = truncate_tokens(tok, it["source"], int(cfg.get("max_source_tokens", 9000)))[0]
        st = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(it["statements"]))
        prompts.append(chat_prompt(tok, spec, JUDGE_SYSTEM, f"PAPER:\n<<<\n{src}\n>>>\n\nSTATEMENTS:\n{st}"))
        params.append(json_params(schema(len(it["statements"])), 40 * len(it["statements"]) + 50))
    outs = iter(llm.generate(prompts, params, use_tqdm=False))
    verdicts = []
    for it in items:
        if not it["statements"]:
            verdicts.append([])
            continue
        o = next(outs)
        verdicts.append((parse_json(o.outputs[0].text if o.outputs else "") or {}).get("verdicts") or [])
    return verdicts


@stage("ws5", "judge", needs=("generate",), resources="gpu", env="llm", config_keys=("ws5.judge",),
       help="faithfulness: claim entailment of every system's KU against the paper, by a third-family judge")
def run_judge(ctx: Context) -> dict:
    cfg = ctx.section("ws5.judge")
    papers = stage_table(ctx, "papers", "papers.parquet").set_index("paper_key")
    kus = all_kus(ctx)
    items = [{"system": r.system, "paper_key": r.paper_key, "source": papers.at[r.paper_key, "source"],
              "statements": statements(parse_ku(r.text), int(cfg.get("max_claims", 12)))} for r in kus.itertuples()]
    judge = model_spec(ctx.cfg, cfg.get("model", "granite3.3-8b"))
    log(f"judge: {len(items)} KUs with {judge['name']}")
    verdicts = with_model(judge, _judge, items, cfg, gpu_memory_utilization=0.85)
    rows = [{"system": it["system"], "paper_key": it["paper_key"], "statement": s, "verdict": v}
            for it, vs in zip(items, verdicts, strict=True) for s, v in zip(it["statements"], vs, strict=False)]
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.out / "verdicts.parquet", index=False)
    f = df.assign(sup=df["verdict"] == "SUPPORTED").groupby("system")["sup"].mean()
    ctx.publish("WS5 faithfulness judge", [{"type": "kpis", "items": [
        {"label": "statements judged", "value": len(df), "fmt": "int"}, {"label": "judge", "value": judge["name"]},
        {"label": "supported (all systems)", "value": float(f.mean()) if len(f) else None, "fmt": "pct1"}]}],
        summary=f"{len(df)} statements")
    return {"statements": len(df)}


def _answer(llm, tok, spec, items: list[dict]) -> list[str]:
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
              "required": ["answer"], "additionalProperties": False}
    prompts = []
    for it in items:
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(it["options"]))
        ctx_txt = truncate_tokens(tok, it["context"], 6000)[0] if it["context"] else "(no summary)"
        prompts.append(chat_prompt(tok, spec, ANSWER_SYSTEM, f"SUMMARY:\n{ctx_txt}\n\nQUESTION: {it['question']}\n{opts}"))
    outs = llm.generate(prompts, json_params(schema, 20), use_tqdm=False)
    return [((parse_json(o.outputs[0].text if o.outputs else "") or {}).get("answer") or "") for o in outs]


@stage("ws5", "answer", needs=("generate", "mcq"), resources="gpu", env="llm", config_keys=("ws5.answer",),
       help="coverage: an answerer from another family answers the MCQs from each system's KU only")
def run_answer(ctx: Context) -> dict:
    cfg = ctx.section("ws5.answer")
    mcq = stage_table(ctx, "mcq", "mcq.parquet")
    kus = all_kus(ctx)
    ctxs = {(r.system, r.paper_key): render(clean_ku(parse_ku(r.text) or {}), "prose") for r in kus.itertuples()}
    systems = sorted(kus["system"].unique()) + ["closed_book"]
    items = [{"system": s, "qid": q.qid, "paper_key": q.paper_key, "question": q.question, "options": list(q.options),
              "gold": q.answer, "context": "" if s == "closed_book" else ctxs.get((s, q.paper_key), "")}
             for s in systems for q in mcq.itertuples()]
    ans = model_spec(ctx.cfg, cfg.get("model", "phi4-mini"))
    preds = with_model(ans, _answer, items, gpu_memory_utilization=0.85)
    df = pd.DataFrame([{k: it[k] for k in ("system", "qid", "paper_key", "gold")} | {"pred": p, "correct": int(p == it["gold"])}
                       for it, p in zip(items, preds, strict=True)])
    df.to_parquet(ctx.out / "answers.parquet", index=False)
    ctx.publish("WS5 coverage answers", [{"type": "kpis", "items": [
        {"label": "answers", "value": len(df), "fmt": "int"}, {"label": "answerer", "value": ans["name"]},
        {"label": "closed-book floor", "value": float(df.loc[df["system"] == "closed_book", "correct"].mean()), "fmt": "pct1"}]}],
        summary=f"{len(df)} answers")
    return {"answers": len(df)}


@stage("ws5", "results", needs=("judge", "answer"), config_keys=("ws5.results",),
       help="per system: validity, faithfulness, coverage (CI), verbatim overlap, tokens / paper, papers / GPU-hour; learning curves")
def run_results(ctx: Context) -> dict:
    papers = stage_table(ctx, "papers", "papers.parquet").set_index("paper_key")
    req = list(ctx.section("ws5.validate").get("required_fields", []))
    kus = all_kus(ctx)
    rows = []
    for r in kus.itertuples():
        ku = parse_ku(r.text)
        sc, ov = schema_check(ku, req), overlap(ku, papers.at[r.paper_key, "source"])
        rows.append({"system": r.system, "paper_key": r.paper_key, "valid": sc["valid"], "filled": sc["filled"], **ov,
                     "tokens_out": r.tokens_out, "gpu_s": r.gpu_seconds_per_paper,
                     "heldout_domain": bool(papers.at[r.paper_key, "heldout_domain"])})
    per = pd.DataFrame(rows)
    v = pd.read_parquet(ctx.cfg.stage_dir("ws5", "judge") / "verdicts.parquet")
    faith = v.assign(sup=(v["verdict"] == "SUPPORTED").astype(float)).groupby(["system", "paper_key"])["sup"].mean().rename("faithful")
    per = per.merge(faith.reset_index(), on=["system", "paper_key"], how="left")
    a = pd.read_parquet(ctx.cfg.stage_dir("ws5", "answer") / "answers.parquet")
    a["cluster"] = a["paper_key"]
    cov, _ = cluster_bootstrap(a, "correct", "cluster", ["system"], n_boot=int(ctx.section("ws5.results").get("n_boot", 2000)))
    cov = cov.rename(columns={"mean": "coverage", "lo": "cov_lo", "hi": "cov_hi"})[["system", "coverage", "cov_lo", "cov_hi"]]
    # context-dependent coverage: only MCQs the answerer gets wrong closed book (4-option MCQs are partly guessable)
    hard = set(a.loc[(a["system"] == "closed_book") & (a["correct"] == 0), "qid"])
    cov_dep = a[a["qid"].isin(hard)].groupby("system")["correct"].mean().rename("coverage_ctx_dependent")
    cov_held = a.merge(per[["paper_key", "heldout_domain"]].drop_duplicates(), on="paper_key").query("heldout_domain") \
        .groupby("system")["correct"].mean().rename("coverage_heldout_domain")
    summ = per.groupby("system").agg(papers=("paper_key", "size"), valid=("valid", "mean"), filled=("filled", "mean"),
                                     faithful=("faithful", "mean"), max_span=("max_span", "median"),
                                     copied_kgram_share=("copied_kgram_share", "mean"), tokens_out=("tokens_out", "median"),
                                     gpu_s=("gpu_s", "mean")).reset_index()
    summ["papers_per_gpu_hour"] = 3600 / summ["gpu_s"]
    summ = summ.merge(cov, on="system", how="outer").merge(cov_held.reset_index(), on="system", how="left") \
        .merge(cov_dep.reset_index(), on="system", how="left")
    meta = {r["run"]: r for r in runs(ctx)}
    summ["student"] = summ["system"].map(lambda s: meta.get(s, {}).get("student"))
    summ["method"] = summ["system"].map(lambda s: meta.get(s, {}).get("method"))
    summ["size"] = summ["system"].map(lambda s: meta.get(s, {}).get("size"))
    summ.to_csv(ctx.out / "systems.csv", index=False)
    per.to_csv(ctx.out / "per_paper.csv", index=False)

    style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    lc = summ.dropna(subset=["size"]).sort_values("size")
    for (st, m), g in lc.groupby(["student", "method"]):
        axes[0].plot(g["size"], g["coverage"], marker="o", label=f"{st} {m}")
        axes[1].plot(g["size"], g["faithful"], marker="o", label=f"{st} {m}")
    for ax, col, lab in ((axes[0], "coverage", "coverage (MCQ accuracy)"), (axes[1], "faithful", "faithfulness (supported share)")):
        for ref, ls in (("teacher", "--"), ("released", ":")):
            val = summ.loc[summ["system"] == ref, col]
            if len(val) and pd.notna(val.iloc[0]):
                ax.axhline(float(val.iloc[0]), ls=ls, color="#555", lw=1, label=ref)
        ax.set_xscale("log")
        ax.set_xlabel("distillation examples")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6)
    figs = save(fig, ctx.out / "figures", "ws5_learning_curves")
    best = summ.dropna(subset=["size"]).sort_values("coverage", ascending=False).head(1)
    t = summ.loc[summ["system"] == "teacher"]
    kp = [{"label": "systems", "value": len(summ), "fmt": "int"}]
    if len(best):
        kp.append({"label": "best student coverage", "value": float(best["coverage"].iloc[0]), "fmt": "pct1", "hint": best["system"].iloc[0]})
    if len(t):
        kp.append({"label": "teacher coverage (ceiling)", "value": float(t["coverage"].iloc[0]), "fmt": "pct1"})
    if len(best) and pd.notna(best["papers_per_gpu_hour"].iloc[0]):
        kp.append({"label": "best student papers / GPU-hour", "value": float(best["papers_per_gpu_hour"].iloc[0]), "fmt": "float"})
    cols = [{"key": "system", "label": "system"}, {"key": "valid", "label": "schema valid", "fmt": "pct1"},
            {"key": "faithful", "label": "faithful", "fmt": "pct1"}, {"key": "coverage", "label": "coverage", "fmt": "pct1"},
            {"key": "cov_lo", "label": "lo", "fmt": "pct1"}, {"key": "cov_hi", "label": "hi", "fmt": "pct1"},
            {"key": "coverage_ctx_dependent", "label": f"coverage, context-dependent MCQs (n={len(hard)})", "fmt": "pct1"},
            {"key": "coverage_heldout_domain", "label": "coverage (held-out discipline)", "fmt": "pct1"},
            {"key": "max_span", "label": "longest copied span (median words)", "fmt": "int"},
            {"key": "copied_kgram_share", "label": "copied 8-grams", "fmt": "pct1"},
            {"key": "tokens_out", "label": "tokens / paper", "fmt": "int"},
            {"key": "papers_per_gpu_hour", "label": "papers / GPU-hour", "fmt": "float"}]
    ctx.publish("WS5 results: does distillation reach the teacher?", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "table", "title": "Systems on the test papers", "columns": cols,
         "rows": summ.sort_values("coverage", ascending=False).to_dict("records")},
        {"type": "image", "title": "Learning curves (teacher dashed, released v2.0 dotted)", "src": figs["png"],
         "alt": "coverage and faithfulness vs distillation examples"},
    ], summary=f"{len(summ)} systems; best student coverage "
               + (f"{best['coverage'].iloc[0]:.1%}" if len(best) else "n/a"))
    return {"systems": len(summ)}
