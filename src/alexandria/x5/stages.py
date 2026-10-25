"""X5 stages: items -> counterfactual -> answer -> grade -> results (+ report).

items           QA items whose gold paper is in a retrieval index: WS2 QA-bench questions (pre-cutoff papers, WS3
                index) and X2 post-cutoff questions (X2 index). Per item: the gold paper's cleaned KU and the top-k KU
                units the hybrid backend retrieves for the question (R arm, summary view).
counterfactual  A counterfactual answer (numbers: the first number scaled; otherwise proposed by the generator) and an
                edited KU in which only the sentences stating the answer are rewritten (generator, JSON-constrained),
                plus literal replacement of left-over mentions. Kept only if the edited KU states the new answer, no
                longer states the original one, and nothing else changed.
answer          Each answer model (one SLURM array task per model), same prompt as WS2: closed book; original KU;
                counterfactual KU; R arm with the original retrieved units; R arm with the gold paper's units rebuilt
                from the edited KU (a copy of the retrieved results; the index is never modified).
grade           SimpleQA judge twice per answer: against the original and against the counterfactual answer.
results         Original-answer / counterfactual-answer rates per model x condition with bootstrap CIs; split by prior
                (model answers the original closed book or not) and by pre- / post-cutoff papers.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..llm import chat_prompt, embedder_spec, json_params, model_spec, parse_json, truncate_tokens, with_model
from ..qa import ANSWER_SYSTEM, answer_user, grade_batch
from ..runtime import Context, log, stage
from ..ws4.common import clean_ku, render
from .common import (apply_edits, check, ku_text, label, mentions, perturb_number, rank_candidates, replace_everywhere,
                     sentences, summary_units, swap_units)


def _table(ctx: Context, stage_: str, name: str) -> pd.DataFrame:
    p = ctx.cfg.stage_dir("x5", stage_) / name
    if not p.exists():
        raise FileNotFoundError(f"x5/{stage_} has no {name}; run that stage first")
    return pd.read_parquet(p)


def _n(ctx: Context, key: str, default):
    return ctx.cfg.profile_settings.get(f"x5_{key}", default) if ctx.cfg.profile == "debug" else default


@stage("x5", "items", config_keys=("x5.items",),
       help="QA items with the gold paper's KU (WS2 pre-cutoff + X2 post-cutoff) and the R arm's retrieved KU units")
def run_items(ctx: Context) -> dict:
    from ..skill.backend import Backend
    cfg = ctx.section("x5.items")
    seed, per = int(cfg.get("seed", 0)), int(cfg.get("max_per_paper", 2))
    rows = []
    q = pd.read_parquet(ctx.cfg.stage_dir("ws2", "questions") / "questions.parquet")
    kus = pd.read_parquet(ctx.cfg.stage_dir("ws2", "ku") / "kus.parquet").drop_duplicates("paper_key").set_index("paper_key")
    q = q[(q["status"] == "ok") & q["paper_key"].isin(kus.index)].sample(frac=1.0, random_state=seed)
    q = q.groupby("paper_key").head(per).head(int(_n(ctx, "pre_items", cfg.get("pre_items", 600))))
    for r in q.itertuples():
        rows.append({"qid": r.qid, "period": "pre-cutoff", "index": "ws3", "paper_key": r.paper_key, "question": r.question,
                     "answer": r.answer, "qtype": r.qtype, "ku": json.dumps(clean_ku(json.loads(kus.at[r.paper_key, "ku_fixed"])))})
    x2q = ctx.cfg.stage_dir("x2", "questions") / "questions.parquet"
    x2k = ctx.cfg.stage_dir("x2", "summarize") / "kus.parquet"
    if x2q.exists() and x2k.exists():
        q2 = pd.read_parquet(x2q)
        k2 = pd.read_parquet(x2k).drop_duplicates("paper_key").set_index("paper_key")
        q2 = q2[q2["paper_key"].isin(k2.index)]
        n2 = _n(ctx, "post_items", cfg.get("post_items"))
        q2 = q2.head(int(n2)) if n2 else q2
        for r in q2.itertuples():
            rows.append({"qid": r.qid, "period": "post-cutoff", "index": "x2", "paper_key": r.paper_key,
                         "question": r.question, "answer": r.answer, "qtype": r.qtype,
                         "ku": json.dumps(clean_ku(json.loads(k2.at[r.paper_key, "ku"])))})
    items = pd.DataFrame(rows)
    k = int(cfg.get("retrieve_k", 10))
    retrieved = {}
    for idx_name, g in items.groupby("index"):
        be = Backend(ctx.cfg.stage_dir(idx_name, "index"), embedder_spec(ctx.cfg, cfg.get("embedder", "bge-m3")),
                     ctx.out / f"cache_{idx_name}.sqlite", device="cpu")
        u = be.units
        for r in g.itertuples():
            ids = be.retrieve(r.question, ["summary"], k)
            retrieved[r.qid] = json.dumps([{"paper_key": u.iloc[i]["paper_key"], "field": u.iloc[i]["field"],
                                            "text": str(u.iloc[i]["text"])} for i in ids])
    items["retrieved"] = items["qid"].map(retrieved)
    items["gold_in_topk"] = [any(x["paper_key"] == pk for x in json.loads(rt)) for pk, rt in zip(items["paper_key"], items["retrieved"], strict=True)]
    items.to_parquet(ctx.out / "items.parquet", index=False)
    by = items.groupby("period").agg(items=("qid", "size"), papers=("paper_key", "nunique"), gold_in_topk=("gold_in_topk", "mean"))
    ctx.publish("X5 items", [
        {"type": "kpis", "items": [{"label": "items", "value": len(items), "fmt": "int"},
                                   {"label": "gold paper in the R arm's top-k", "value": float(items["gold_in_topk"].mean()), "fmt": "pct1"}]},
        {"type": "table", "title": "Items per period", "rows": by.reset_index().to_dict("records"), "columns": [
            {"key": "period", "label": "period"}, {"key": "items", "label": "items", "fmt": "int"},
            {"key": "papers", "label": "papers", "fmt": "int"}, {"key": "gold_in_topk", "label": "gold in top-k", "fmt": "pct1"}]},
    ], summary=f"{len(items)} items ({(items['period'] == 'post-cutoff').sum()} post-cutoff)")
    return {"items": len(items)}


EDIT_SYSTEM = """You create a counterfactual version of a scientific summary for a controlled experiment.
You get a question, its correct answer according to the summary, and numbered sentences from the summary.
Rewrite ONLY the sentences that state the answer so that they state the NEW answer instead, changing as little else as possible (same style, same length). Do not change sentences that do not state the answer. Answer with JSON only."""


def _edit_schema(n_max: int, propose: bool) -> dict:
    props = {"edits": {"type": "array", "minItems": 1, "maxItems": n_max, "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "new": {"type": "string"}},
        "required": ["id", "new"], "additionalProperties": False}}}
    req = ["edits"]
    if propose:
        props = {"new_answer": {"type": "string"}, **props}
        req = ["new_answer", "edits"]
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def _edit(llm, tok, spec, jobs: list[dict], cfg: dict) -> list[dict]:
    prompts, params = [], []
    for j in jobs:
        sent = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(j["cands"]))
        ask = (f"Choose a plausible but DIFFERENT answer of the same kind (an entity, method, value or outcome that "
               f"could appear in such a paper) and put it in new_answer." if j["cf"] is None
               else f"NEW answer: {j['cf']}")
        user = f"Question: {j['question']}\nCorrect answer: {j['answer']}\n{ask}\n\nSentences:\n{sent}"
        prompts.append(chat_prompt(tok, spec, EDIT_SYSTEM, user))
        params.append(json_params(_edit_schema(int(cfg.get("max_edits", 4)), j["cf"] is None), int(cfg.get("max_tokens", 1500))))
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [parse_json(o.outputs[0].text if o.outputs else "") or {} for o in outs]


@stage("x5", "counterfactual", needs=("items",), resources="gpu", env="llm", config_keys=("x5.counterfactual",),
       help="counterfactual answer + minimally edited KU per item (checked: new answer stated, original gone)")
def run_counterfactual(ctx: Context) -> dict:
    cfg = ctx.section("x5.counterfactual")
    items = _table(ctx, "items", "items.parquet")
    jobs = []
    for r in items.itertuples():
        ku = json.loads(r.ku)
        cf = perturb_number(r.answer, r.qid, list(cfg.get("numeric_factors", [0.5, 2.0, 3.0])))
        jobs.append({"qid": r.qid, "question": r.question, "answer": r.answer, "cf": cf, "ku": ku,
                     "cands": rank_candidates(sentences(ku), r.question, r.answer, int(cfg.get("candidates", 12)))})
    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    log(f"counterfactual: {len(jobs)} items with {gen['name']}")
    outs = with_model(gen, _edit, jobs, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    rows = []
    for j, o in zip(jobs, outs, strict=True):
        cf = j["cf"] or str(o.get("new_answer") or "").strip()
        edits = [(j["cands"][e["id"]], str(e["new"]).strip()) for e in (o.get("edits") or [])
                 if isinstance(e, dict) and isinstance(e.get("id"), int) and 0 <= e["id"] < len(j["cands"]) and str(e.get("new") or "").strip()]
        cf_ku = apply_edits(j["ku"], edits) if edits else j["ku"]
        cf_ku = replace_everywhere(cf_ku, j["answer"], cf) if cf else cf_ku
        status = check(j["ku"], cf_ku, j["answer"], cf) if edits else "no_edits"
        rows.append({"qid": j["qid"], "cf_answer": cf, "cf_kind": "numeric" if j["cf"] else "generated",
                     "n_edits": len(edits), "cf_ku": json.dumps(cf_ku), "status": status,
                     "orig_states_answer": mentions(ku_text(j["ku"]), j["answer"])})
    cfd = pd.DataFrame(rows)
    cfd.to_parquet(ctx.out / "counterfactuals.parquet", index=False)
    st = cfd["status"].value_counts()
    ctx.publish("X5 counterfactuals", [
        {"type": "kpis", "items": [{"label": "counterfactuals kept", "value": int((cfd["status"] == "ok").sum()), "fmt": "int"},
                                   {"label": "kept share", "value": float((cfd["status"] == "ok").mean()), "fmt": "pct1"},
                                   {"label": "numeric (rule-based)", "value": float((cfd["cf_kind"] == "numeric").mean()), "fmt": "pct1"},
                                   {"label": "editor", "value": gen["name"]}]},
        {"type": "bar", "title": "Check outcome", "horizontal": True, "fmt": "int", "labels": list(st.index),
         "series": [{"name": "items", "data": st.tolist()}]},
    ], summary=f"{int((cfd['status'] == 'ok').sum())} of {len(cfd)} counterfactuals kept")
    return {"kept": int((cfd["status"] == "ok").sum())}


def _answer(llm, tok, spec, jobs: list[dict], cfg: dict) -> list[str]:
    prompts = []
    for j in jobs:
        ctx_txt = truncate_tokens(tok, j["context"], int(cfg.get("max_context_tokens", 8000)))[0] if j["context"] else None
        prompts.append(chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(j["question"], ctx_txt)))
    outs = llm.generate(prompts, json_params(None, int(cfg.get("max_tokens", 64))), use_tqdm=False)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]


def contexts(item: dict, cf: dict) -> dict[str, str | None]:
    ku, cku = json.loads(item["ku"]), json.loads(cf["cf_ku"])
    retrieved = json.loads(item["retrieved"])
    ou, cu = summary_units(ku), summary_units(cku)
    return {"closed_book": None, "ku_original": render(ku, "prose"), "ku_counterfactual": render(cku, "prose"),
            "rag_original": "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(swap_units(retrieved, item["paper_key"], ou, cu, False))),
            "rag_counterfactual": "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(swap_units(retrieved, item["paper_key"], ou, cu, True)))}


@stage("x5", "answer", kind="map", needs=("counterfactual",), resources="gpu", env="llm", config_keys=("x5.answer",),
       help="answers under closed book / original KU / counterfactual KU / R arm original / R arm counterfactual")
def run_answer(ctx: Context) -> dict:
    cfg = ctx.section("x5.answer")
    items = _table(ctx, "items", "items.parquet").set_index("qid")
    cfd = _table(ctx, "counterfactual", "counterfactuals.parquet")
    cfd = cfd[cfd["status"] == "ok"]
    conds = list(cfg.get("conditions", []))
    jobs = []
    for c in cfd.to_dict("records"):
        it = {**items.loc[c["qid"]].to_dict(), "qid": c["qid"]}
        cx = contexts(it, c)
        jobs += [{"qid": c["qid"], "condition": k, "question": it["question"], "context": cx[k]} for k in conds]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = []
    for name in list(cfg.get("models", []))[ctx.shard_index::ctx.shard_count]:
        dst = parts / f"{name}.parquet"
        if dst.exists() and not ctx.force:
            continue
        spec = model_spec(ctx.cfg, name)
        log(f"answer: {name}: {len(jobs)} prompts")
        preds = with_model(spec, _answer, jobs, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
        pd.DataFrame([{"qid": j["qid"], "condition": j["condition"], "model": name, "prediction": p}
                      for j, p in zip(jobs, preds, strict=True)]).to_parquet(dst, index=False)
        done.append(name)
    ctx.publish("X5 answers", [{"type": "kpis", "items": [
        {"label": "models answered", "value": len(list(parts.glob('*.parquet'))), "fmt": "int"},
        {"label": "prompts per model", "value": len(jobs), "fmt": "int"},
        {"label": "conditions", "value": ", ".join(conds)}]}], summary=f"{len(jobs)} prompts per model")
    return {"done": done}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("x5", "grade", needs=("answer",), resources="gpu", env="llm", config_keys=("x5.grade",),
       help="SimpleQA judge against the original and against the counterfactual answer")
def run_grade(ctx: Context) -> dict:
    import glob
    items = _table(ctx, "items", "items.parquet").set_index("qid")
    cfd = _table(ctx, "counterfactual", "counterfactuals.parquet").set_index("qid")
    a = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ctx.cfg.stage_dir("x5", "answer") / "parts" / "*.parquet")))],
                  ignore_index=True)
    jobs = []
    for r in a.itertuples():
        q = items.at[r.qid, "question"]
        jobs.append({"question": q, "gold": items.at[r.qid, "answer"], "predicted": r.prediction})
        jobs.append({"question": q, "gold": cfd.at[r.qid, "cf_answer"], "predicted": r.prediction})
    judge = model_spec(ctx.cfg, ctx.section("x5.grade").get("judge", "qwen3-8b"))
    log(f"grade: {len(jobs)} judgements with {judge['name']}")
    g = with_model(judge, _grade, jobs, gpu_memory_utilization=0.85)
    a["vs_original"], a["vs_counterfactual"] = g[0::2], g[1::2]
    a["label"] = [label(o, c) for o, c in zip(a["vs_original"], a["vs_counterfactual"], strict=True)]
    a["period"] = a["qid"].map(items["period"])
    a["paper_key"] = a["qid"].map(items["paper_key"])
    a.to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("X5 grading", [{"type": "kpis", "items": [{"label": "answers graded", "value": len(a), "fmt": "int"},
                                                          {"label": "judge", "value": judge["name"]}]}],
                summary=f"{len(a)} answers")
    return {"graded": len(a)}


def boot_ci(x: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


@stage("x5", "results", needs=("grade",), config_keys=("x5.results",),
       help="original vs counterfactual answer rates per model x condition, by prior strength and cutoff period")
def run_results(ctx: Context) -> dict:
    from ..figures import plt, save, style
    nb = int(ctx.section("x5.results").get("n_boot", 2000))
    g = _table(ctx, "grade", "graded.parquet")
    cb = g[g["condition"] == "closed_book"].set_index(["model", "qid"])["label"]
    g["prior"] = [("knows original" if cb.get((m, q)) in ("original", "both") else "does not know")
                  for m, q in zip(g["model"], g["qid"], strict=True)]
    rows = []
    for keys, d in [((m, c, "all", "all"), x) for (m, c), x in g.groupby(["model", "condition"])] + \
                   [((m, c, "all", p), x) for (m, c, p), x in g.groupby(["model", "condition", "prior"])] + \
                   [((m, c, per, "all"), x) for (m, c, per), x in g.groupby(["model", "condition", "period"])]:
        m, c, per, pr = keys
        r = {"model": m, "condition": c, "period": per, "prior": pr, "n": len(d)}
        for lab in ("original", "counterfactual"):
            mean, lo, hi = boot_ci((d["label"].isin([lab, "both"])).to_numpy(float), nb)
            r.update({f"{lab}": mean, f"{lab}_lo": lo, f"{lab}_hi": hi})
        r["abstain"] = float((d["label"] == "abstain").mean())
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(ctx.out / "rates.csv", index=False)
    main = res[(res["period"] == "all") & (res["prior"] == "all")]
    conds = ["closed_book", "ku_original", "ku_counterfactual", "rag_original", "rag_counterfactual"]
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.38
    for k, m in enumerate(sorted(main["model"].unique())):
        d = main[main["model"] == m].set_index("condition").reindex(conds)
        x = np.arange(len(conds)) + (k - 0.5) * w
        ax.bar(x, d["counterfactual"], w * 0.95, label=f"{m}: counterfactual answer")
        ax.scatter(x, d["original"], marker="_", s=180, color="black")
    ax.set_xticks(np.arange(len(conds)), [c.replace("_", "\n") for c in conds], fontsize=7)
    ax.set_ylabel("share of answers")
    ax.legend(fontsize=6)
    figs = save(fig, ctx.out / "figures", "x5_conflict")
    cf_ku = main[main["condition"] == "ku_counterfactual"]
    cf_rag = main[main["condition"] == "rag_counterfactual"]
    know = res[(res["condition"] == "ku_counterfactual") & (res["prior"] == "knows original") & (res["period"] == "all")]
    kp = [{"label": "follows the edited KU (KU in context)", "value": float(cf_ku["counterfactual"].mean()), "fmt": "pct1",
           "hint": "mean over models"},
          {"label": "follows the edited KU (R arm)", "value": float(cf_rag["counterfactual"].mean()), "fmt": "pct1"},
          {"label": "keeps the original answer (edited KU)", "value": float(cf_ku["original"].mean()), "fmt": "pct1"},
          {"label": "follows edited KU when the model knows the original", "value": float(know["counterfactual"].mean()) if len(know) else None,
           "fmt": "pct1"}]
    cols = [{"key": "model", "label": "model"}, {"key": "condition", "label": "condition"}, {"key": "n", "label": "answers", "fmt": "int"},
            {"key": "original", "label": "original answer", "fmt": "pct1"}, {"key": "original_lo", "label": "lo", "fmt": "pct1"},
            {"key": "original_hi", "label": "hi", "fmt": "pct1"}, {"key": "counterfactual", "label": "counterfactual answer", "fmt": "pct1"},
            {"key": "counterfactual_lo", "label": "lo", "fmt": "pct1"}, {"key": "counterfactual_hi", "label": "hi", "fmt": "pct1"},
            {"key": "abstain", "label": "abstain", "fmt": "pct1"}]
    ctx.publish("X5 results: do models follow the corpus or their prior?", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "image", "title": "Counterfactual-answer rate per condition (bars) and original-answer rate (ticks)",
         "src": figs["png"], "alt": "knowledge conflict rates"},
        {"type": "table", "title": "Rates per model and condition (95% bootstrap CI over items)",
         "rows": main.to_dict("records"), "columns": cols},
        {"type": "table", "title": "By prior: does the model answer the original question closed book?",
         "rows": res[(res["prior"] != "all")].to_dict("records"), "columns": [*cols[:2], {"key": "prior", "label": "prior"}, *cols[2:]]},
        {"type": "table", "title": "Pre- vs post-cutoff papers", "rows": res[(res["period"] != "all")].to_dict("records"),
         "columns": [*cols[:2], {"key": "period", "label": "period"}, *cols[2:]]},
    ], summary=f"edited KU followed {cf_ku['counterfactual'].mean():.0%} (KU) / {cf_rag['counterfactual'].mean():.0%} (R arm)")
    return {"rows": len(res)}
