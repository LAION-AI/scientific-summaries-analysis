"""Stages `answer` and `grade` (X4, GPU).

answer  Every answer model x arm x item, through the WS3 harness (alexandria.ws3.agent: same system prompt, JSON
        actions constrained to the arm's tools, the same retrieved-token budget for every retrieval arm):
          A0          closed book
          R-fulltext  top-k full-text chunks (plain RAG)
          R-KU        the whole KUs of the top-k papers
          R-field     top-k units of the KU field the question asks about (rule-based router; `any` if unclear)
          S           agent with `search` (incl. its `field` argument) + `get_paper` over the KUs
          F           agent with `search_fulltext`
        Logged per episode: answer, tool calls, retrieved tokens, papers seen (gold-paper retrieved), GPU-seconds.
grade   SimpleQA judge (as WS2 / WS3) + exact match / token F1.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from ..llm import embedder_spec, model_spec, truncate_tokens, with_model
from ..qa import exact_match, grade_batch, is_abstention, token_f1
from ..runtime import Context, log, stage
from ..ws3.agent import Episode, _vllm_generate, run_agent, system_prompt
from .common import gold_recall, ku_text, paper_fields, route_field

AGENT_ARMS = {"S": (["search", "get_paper"], "summary"), "F": (["search_fulltext"], "fulltext")}


def passages(arm: str, q: str, backend, pf: dict, cfg: dict, tok) -> tuple[str, list[str], str]:
    """(passage text, paper keys in it, field used) of a RAG arm, cut to the retrieved-token budget."""
    budget, k = int(cfg.get("max_retrieved_tokens", 8000)), int(cfg.get("k_units", 20))
    field = "-"
    if arm == "R-fulltext":
        rows = backend.retrieve(q, ["fulltext"], k)
        text, papers = backend.render(rows), list(dict.fromkeys(backend.units.iloc[rows]["paper_key"])) if rows else []
    elif arm == "R-field":
        field = route_field(q)
        ranked = sorted(backend._rank(q, "summary", None if field == "any" else field, False).items(), key=lambda kv: -kv[1])
        rows = [i for i, _ in ranked[:k]]
        text, papers = backend.render(rows), list(dict.fromkeys(backend.units.iloc[rows]["paper_key"])) if rows else []
    else:                                                                   # R-KU: whole KUs of the top papers
        ranked = sorted(backend._rank(q, "summary", None, False).items(), key=lambda kv: -kv[1])
        papers = list(dict.fromkeys(backend.units.iloc[[i for i, _ in ranked]]["paper_key"]))[: int(cfg.get("k_papers", 5))]
        meta = backend.papers
        text = "\n\n".join(f"[{n}] {meta.at[p, 'title'] or meta.at[p, 'paper_id']} (id: {p}; whole KU)\n{ku_text(pf.get(p, {}))}"
                           for n, p in enumerate(papers, 1))
    text, _, _ = truncate_tokens(tok, text, budget)
    return text, papers, field


def _run(llm, tok, spec, payload: dict) -> list[dict]:
    from ..skill.backend import Backend
    cfg = payload["cfg"]
    backend = Backend(payload["index_dir"], payload["embedder"], payload["cache"], dense=bool(cfg.get("dense", True)),
                      device=cfg.get("encoder_device", "cpu"))
    pf = paper_fields(backend.units)
    gen = _vllm_generate(llm, tok, spec)
    count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])          # noqa: E731
    cut = lambda t, n: truncate_tokens(tok, t, n)[0]                             # noqa: E731
    max_calls = int(cfg.get("max_tool_calls", 6))
    rows = []
    for arm in payload["arms"]:
        t0 = time.time()
        eps, fields_used = [], []
        for it in payload["items"]:
            if arm in AGENT_ARMS:
                tools, view = AGENT_ARMS[arm]
                msgs = [{"role": "system", "content": system_prompt(it, tools, max_calls)},
                        {"role": "user", "content": it["question"]}]
                eps.append(Episode(item=it, arm=arm, seed=0, tools=tools, view=view, only_full_text=False, messages=msgs))
                fields_used.append("-")
            elif arm == "A0":
                msgs = [{"role": "system", "content": system_prompt(it, [], max_calls)}, {"role": "user", "content": it["question"]}]
                eps.append(Episode(item=it, arm=arm, seed=0, tools=[], view=None, only_full_text=False, messages=msgs))
                fields_used.append("-")
            else:
                text, papers, field = passages(arm, it["question"], backend, pf, cfg, tok)
                msgs = [{"role": "system", "content": system_prompt(it, [], max_calls, passages=True)},
                        {"role": "user", "content": f"Passages:\n<<<\n{text}\n>>>\n\nQuestion: {it['question']}"}]
                e = Episode(item=it, arm=arm, seed=0, tools=[], view=None, only_full_text=False, messages=msgs, papers=papers)
                e.retrieved = count(text)
                eps.append(e)
                fields_used.append(field)
        run_agent(eps, gen, backend, count, cut, cfg, float(cfg.get("temperature", 0.0)))
        log(f"answer: {spec['name']} {arm}: {len(eps)} episodes in {time.time() - t0:.0f} s")
        for e, fu in zip(eps, fields_used, strict=True):
            rows.append({"item_id": e.item["item_id"], "type": e.item["type"], "field": e.item["field"], "arm": arm,
                         "model": spec["name"], "answer": e.answer, "error": e.error, "tool_calls": e.calls,
                         "retrieved_tokens": e.retrieved, "steps": e.steps, "tokens_in": e.tokens_in,
                         "tokens_out": e.tokens_out, "gpu_seconds": e.gpu_seconds, "routed_field": fu,
                         "gold_retrieved": gold_recall(e.papers, [e.item["gold_paper_key"]]),
                         "group_recall": gold_recall(e.papers, list(e.item["group"])),
                         "trace": json.dumps(e.trace, ensure_ascii=False)})
    return rows


@stage("x4", "answer", kind="map", needs=("items",), resources="gpu", env="llm", config_keys=("x4.answer",),
       help="every model x arm (A0, R-fulltext, R-KU, R-field, S, F) x item through the WS3 harness")
def run_answer(ctx: Context) -> dict:
    cfg = ctx.section("x4.answer")
    items = pd.read_parquet(ctx.cfg.stage_dir("x4", "items") / "items.parquet")
    recs = items.to_dict("records")
    for r in recs:
        r["options"], r["group"], r["gold_papers"] = [], list(r["group"]), list(r["gold_papers"])
    models = list(cfg.get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = []
    for name in models:
        dst = parts / f"{name}.parquet"
        if dst.exists() and not ctx.force:
            continue
        payload = {"cfg": cfg, "items": recs, "arms": list(cfg.get("arms", [])),
                   "index_dir": str(ctx.cfg.stage_dir("ws3", "index")),
                   "embedder": embedder_spec(ctx.cfg, cfg.get("embedder", "bge-m3")),
                   "cache": str(ctx.out / "cache" / f"{name}.sqlite")}
        rows = with_model(model_spec(ctx.cfg, name), _run, payload,
                          gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.88)))
        pd.DataFrame(rows).to_parquet(dst, index=False)
        done.append(name)
    n = len(list(parts.glob("*.parquet")))
    ctx.publish("X4 answers", [{"type": "kpis", "items": [
        {"label": "models answered", "value": n, "fmt": "int"},
        {"label": "items", "value": len(items), "fmt": "int"},
        {"label": "arms", "value": len(cfg.get("arms", [])), "fmt": "int"}]}], summary=f"{n} models")
    return {"done": done}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("x4", "grade", needs=("answer",), resources="gpu", env="llm", config_keys=("x4.grade",),
       help="SimpleQA judge over every answer + exact match / token F1")
def run_grade(ctx: Context) -> dict:
    cfg = ctx.section("x4.grade")
    parts = sorted((ctx.cfg.stage_dir("x4", "answer") / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("x4/answer has no parts; run answer first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    items = pd.read_parquet(ctx.cfg.stage_dir("x4", "items") / "items.parquet")[["item_id", "question", "answer"]]
    df = df.merge(items.rename(columns={"answer": "gold"}), on="item_id", how="left")
    judge = model_spec(ctx.cfg, cfg.get("judge", "qwen3-8b"))
    log(f"grade: {len(df)} answers with {judge['name']}")
    df["grade"] = with_model(judge, _grade, [{"question": r.question, "gold": r.gold, "predicted": r.answer or ""}
                                             for r in df.itertuples()],
                             gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["abstained"] = [int(is_abstention(a)) for a in df["answer"]]
    df["em"] = [int(exact_match(a, g)) for a, g in zip(df["answer"], df["gold"], strict=True)]
    df["f1"] = [token_f1(a, g) for a, g in zip(df["answer"], df["gold"], strict=True)]
    df.drop(columns=["question", "trace"]).to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("X4 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"}, {"label": "judge", "value": judge["name"]},
        {"label": "CORRECT overall", "value": float(df["correct"].mean()), "fmt": "pct1"}]}], summary=f"{len(df)} answers")
    return {"graded": len(df)}
