"""Stage `rag-budget` (X6, GPU): accuracy against the retrieved-context token budget, KU units vs full-text chunks.

Questions: WS3 qabench-open (over the WS3 index) and the X2 post-cutoff questions (over the X2 index: papers the
models cannot have seen). For each question and unit kind (`summary` = KU fields + claims, `fulltext` = chunks of the
source text) the hybrid retriever (BM25 + bge-m3, RRF; skill.backend) ranks `candidates` units; the context is the
ranked units packed in order up to N tokens of the answer model's tokenizer (the last unit cut to fit), for every N in
`budgets`. Same answer prompt as WS2; SimpleQA judge. Also records whether the gold paper is in the packed context.
One SLURM array task per answer model. Output: parts/<model>.parquet.
"""
from __future__ import annotations

import time

import pandas as pd

from ..llm import chat_prompt, embedder_spec, json_params, model_spec, with_model
from ..qa import ANSWER_SYSTEM, answer_user
from ..runtime import Context, log, stage
from .common import setting, stage_dir
from .inference import grade_frame


def load_items(ctx: Context, n_limit: int | None) -> pd.DataFrame:
    it = pd.read_parquet(stage_dir(ctx, "ws3", "benchmarks") / "items.parquet")
    it = it[it["bench"] == "qabench-open"]
    if n_limit and len(it) > n_limit:
        it = it.sample(n=int(n_limit), random_state=0)
    rows = [{"item_id": r.item_id, "question": r.question, "answer": r.answer, "gold_paper_key": r.gold_paper_key,
             "source": "qabench-open", "index": "ws3"} for r in it.itertuples()]
    x2 = stage_dir(ctx, "x2", "questions") / "questions.parquet"
    if x2.exists():
        q = pd.read_parquet(x2)
        q = q[q["status"] == "ok"] if "status" in q else q
        rows += [{"item_id": r.qid, "question": r.question, "answer": r.answer, "gold_paper_key": r.paper_key,
                  "source": "x2-post-cutoff", "index": "x2"} for r in q.itertuples()]
    return pd.DataFrame(rows)


def candidates(ctx: Context, items: pd.DataFrame, kinds: list[str], k: int) -> dict:
    """{(item_id, kind): [(header, text, paper_key), ...]} in rank order (CPU retrieval, before the GPU model)."""
    from ..skill.backend import Backend
    out = {}
    for index, g in items.groupby("index"):
        d = stage_dir(ctx, index, "index")
        if not (d / "units.parquet").exists():
            log(f"rag-budget: no {index}/index; skipping {len(g)} items")
            continue
        be = Backend(d, embedder_spec(ctx.cfg, "bge-m3"), None, device="cpu", dense=True)
        for r in g.itertuples():
            for kind in kinds:
                rows = be.retrieve(r.question, [kind], k)
                u = be.units.iloc[rows]
                out[(r.item_id, kind)] = [(f"[{n}] {t if isinstance(t, str) and t else pid} "
                                           f"({kd}{'/' + f if isinstance(f, str) and f else ''})", str(x), pk)
                                          for n, (t, pid, kd, f, x, pk) in enumerate(zip(
                                              u["title"], u["paper_id"], u["kind"], u["field"], u["text"],
                                              u["paper_key"], strict=True), 1)]
    return out


def pack(tok, units: list[tuple[str, str, str]], budget: int) -> tuple[str, int, list[str]]:
    """Ranked units in order until `budget` tokens; the last one cut to fit. -> (context, tokens, paper keys)."""
    parts, used, papers = [], 0, []
    for head, text, pk in units:
        block = f"{head}\n{text}"
        ids = tok(block + "\n\n", add_special_tokens=False)["input_ids"]
        if used + len(ids) > budget:
            rest = budget - used
            if rest >= 32:
                parts.append(tok.decode(ids[:rest]))
                papers.append(pk)
                used += rest
            break
        parts.append(block)
        papers.append(pk)
        used += len(ids)
    return "\n\n".join(parts), used, papers


def _answer(llm, tok, spec, items: list[dict], cands: dict, kinds: list[str], budgets: list[int], cfg: dict) -> list[dict]:
    params = json_params(None, int(cfg.get("max_tokens", 64)), 0.0, seed=0)
    rows = []
    for kind in kinds:
        for b in budgets:
            prompts, meta = [], []
            for it in items:
                ctx_text, n, papers = pack(tok, cands.get((it["item_id"], kind), []), int(b))
                prompts.append(chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(it["question"], ctx_text)))
                meta.append({"item_id": it["item_id"], "kind": kind, "budget": int(b), "context_tokens": n,
                             "units": len(papers), "gold_in_context": it["gold_paper_key"] in set(papers)})
            llm.reset_prefix_cache()
            t0 = time.perf_counter()
            outs = llm.generate(prompts, params, use_tqdm=False)
            dt = time.perf_counter() - t0
            for m, o in zip(meta, outs, strict=True):
                text = o.outputs[0].text.strip() if o.outputs else ""
                if "</think>" in text:
                    text = text.split("</think>", 1)[1].strip()
                rows.append({**m, "prediction": text.split("\n")[0][:300],
                             "prompt_tokens": len(o.prompt_token_ids or []),
                             "output_tokens": len(o.outputs[0].token_ids) if o.outputs else 0,
                             "batch_seconds": dt, "batch_n": len(items)})
            log(f"rag-budget: {spec['name']} {kind}@{b}: {dt:.1f}s")
    return rows


@stage("x6", "rag-budget", kind="map", resources="gpu", env="llm", config_keys=("x6.rag", "x6.models", "x6.judge"),
       help="accuracy vs retrieved-token budget (500-8000): KU units vs full-text chunks (WS3 + X2 questions)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x6.rag")
    gpu_mem = float(cfg.get("gpu_memory_utilization", 0.85))
    items = load_items(ctx, setting(ctx, "rag_items", cfg.get("items")))
    kinds, budgets = list(cfg.get("kinds", ["summary", "fulltext"])), [int(b) for b in cfg.get("budgets", [])]
    models = list(ctx.section("x6").get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]
    (ctx.out / "parts").mkdir(parents=True, exist_ok=True)
    todo = [m for m in models if ctx.force or not (ctx.out / "parts" / f"{m}.parquet").exists()]
    cands = candidates(ctx, items, kinds, int(cfg.get("candidates", 60))) if todo else {}
    items = items[[any((i, k) in cands for k in kinds) for i in items["item_id"]]] if todo else items
    for name in todo:
        spec = model_spec(ctx.cfg, name)
        log(f"rag-budget: {name}: {len(items)} items x {len(kinds)} kinds x {len(budgets)} budgets")
        rows = with_model(spec, _answer, items.to_dict("records"), cands, kinds, budgets, cfg,
                          gpu_memory_utilization=gpu_mem)
        df = pd.DataFrame(rows).merge(items[["item_id", "question", "answer", "source"]], on="item_id")
        df = grade_frame(ctx, df, ctx.section("x6").get("judge", "qwen3-8b"), gpu_mem)
        df.drop(columns=["question", "answer"]).assign(model=name).to_parquet(ctx.out / "parts" / f"{name}.parquet",
                                                                             index=False)
    parts = [pd.read_parquet(p) for p in sorted((ctx.out / "parts").glob("*.parquet"))]
    if parts:
        t = pd.concat(parts).groupby(["model", "kind", "budget"]).agg(
            accuracy=("correct", "mean"), gold_in_context=("gold_in_context", "mean"),
            tokens=("context_tokens", "mean"), n=("correct", "size")).reset_index()
        rows_t = t.to_dict("records")
    else:
        rows_t = []
    ctx.publish("X6 RAG token budget", [
        {"type": "table", "title": "Accuracy per retrieved-token budget (KU units vs full-text chunks)",
         "columns": [{"key": "model", "label": "model"}, {"key": "kind", "label": "units"},
                     {"key": "budget", "label": "budget", "fmt": "int"}, {"key": "tokens", "label": "mean tokens", "fmt": "int"},
                     {"key": "accuracy", "label": "accuracy", "fmt": "pct1"},
                     {"key": "gold_in_context", "label": "gold paper in context", "fmt": "pct1"},
                     {"key": "n", "label": "items", "fmt": "int"}], "rows": rows_t, "limit": 60}],
        summary=f"{len(parts)} models, {len(items)} items")
    return {"models_done": todo, "items": len(items)}
