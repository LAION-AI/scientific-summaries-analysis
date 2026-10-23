"""WS3 agent harness and the stages `pilot` and `agent` (GPU, Alex-vllm env).

One harness for every arm: the same system prompt (only the tool descriptions differ), the same answer format, the
same limits (max_tool_calls, max_retrieved_tokens), temperature and seeds. Each turn the model writes one JSON
action {"thought", "action", "arguments"}; generation is constrained to the arm's tool schemas (skill.tools.
action_schema), so malformed tool calls cannot confound the comparison between model families. `pilot` measures
how reliable each model's tool calls are WITHOUT the constraint (plan: >= 95% well-formed before a model is used).

Arms (ws3.agent.arms; definitions in ws3.yaml):
  direct  A0     model alone, one answer step
  sc      A0+    model alone, self-consistency: n samples, majority vote (compute-matched baseline)
  rag     R      top-k hybrid-retrieved passages pasted once (non-agentic), up to the retrieved-token budget
  agent   F/S/S-OA     multi-step tool use over one view of the index
  oracle  Oracle the gold paper's text (QA bench only)
  web     W/W-sch/S+W  need internet; compute nodes have none -> reported as not run
Each episode logs tool calls, retrieved tokens, prompt / output tokens, GPU-seconds and the papers it saw (for the
gold-paper-retrieved rate). Tool responses are cached (skill.backend), so runs replay exactly.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..llm import json_params, messages_prompt, model_spec, parse_json, truncate_tokens, with_model
from ..qa import normalize_answer
from ..runtime import Context, log, stage
from ..skill.tools import TOOLS, action_schema, tool_text

TOOL_ROUTE = {"search": ("search", None), "get_paper": ("get_paper", None),
              "search_fulltext": ("search_fulltext", "fulltext")}

SYSTEM = """You are a research assistant. Answer the user's question.
{tools}
Reply to every message with one JSON object: {{"thought": "<brief reasoning>", "action": "<tool name or answer>", "arguments": {{...}}}}.
- To call a tool, set "action" to its name and "arguments" to its parameters; the result comes back in the next message.
- You can make at most {max_calls} tool calls. When you are confident, or when no calls are left, set "action" to "answer" and "arguments" to {{"answer": "..."}}.
{answer_format}"""

FORMATS = {
    "short": "The answer must be short: a number with its unit, a name, or a phrase of at most 12 words. "
             "If you cannot determine it, answer \"I don't know\".",
    "label": "The answer must be exactly one of: {options}.",
    "mc": "The answer must be the letter of exactly one option.",
}


def answer_schema(item: dict) -> dict:
    opts = list(item.get("options") or [])
    return {"type": "string", "enum": opts} if item["answer_type"] in ("label", "mc") and opts \
        else {"type": "string", "maxLength": 200}


def system_prompt(item: dict, tools: list[str], max_calls: int, passages: bool = False) -> str:
    if tools:
        tb = "Tools (JSON schemas):\n" + tool_text(tools)
    elif passages:
        tb = "You have no tools. Use the passages in the user's message if they help."
    else:
        tb = "You have no tools: answer from your own knowledge."
    fmt = FORMATS[item["answer_type"]].format(options=", ".join(item.get("options") or []))
    return SYSTEM.format(tools=tb, max_calls=max_calls if tools else 0, answer_format=fmt)


@dataclass
class Episode:
    item: dict
    arm: str
    seed: int
    tools: list[str]
    view: str | None
    only_full_text: bool
    messages: list[dict]
    calls: int = 0
    retrieved: int = 0
    steps: int = 0
    done: bool = False
    answer: str | None = None
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    papers: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


Generate = Callable[[list[list[dict]], list[dict | None], float, list[int], int], list[tuple[list[str], int, int]]]


def run_agent(eps: list[Episode], generate: Generate, backend, count_tokens: Callable[[str], int],
              cut_tokens: Callable[[str, int], str], cfg: dict, temperature: float) -> None:
    """Batched multi-step loop over all episodes (each step: one generate call for every unfinished episode)."""
    max_calls, budget = int(cfg.get("max_tool_calls", 6)), int(cfg.get("max_retrieved_tokens", 8000))
    max_tokens = int(cfg.get("max_step_tokens", 400))
    for step in range(max_calls + 1):
        active = [e for e in eps if not e.done]
        if not active:
            break
        schemas = [action_schema(e.tools, answer_schema(e.item),
                                 allow_tools=bool(e.tools) and e.calls < max_calls and e.retrieved < budget
                                 and step < max_calls) for e in active]
        t0 = time.time()
        outs = generate([e.messages for e in active], schemas, temperature, [e.seed for e in active], max_tokens)
        dt = time.time() - t0
        for e, (texts, n_in, n_out) in zip(active, outs, strict=True):
            e.steps += 1
            e.tokens_in += n_in
            e.tokens_out += n_out
            e.gpu_seconds += dt / len(active)
            text = texts[0]
            a = parse_json(text)
            if not isinstance(a, dict) or "action" not in a:
                e.done, e.error, e.answer = True, "unparsable_step", ""
                continue
            act, args = a.get("action"), a.get("arguments") or {}
            if act == "answer":
                e.done, e.answer = True, str(args.get("answer", ""))
                e.trace.append({"action": "answer"})
                continue
            if act not in e.tools:
                e.done, e.error, e.answer = True, f"unknown_tool:{act}", ""
                continue
            tool, view = TOOL_ROUTE[act]
            result, papers = backend.call(tool, args, view=view or e.view, only_full_text=e.only_full_text)
            n = count_tokens(result)
            left = budget - e.retrieved
            if n > left:
                result, n = cut_tokens(result, max(left, 0)) + "\n[retrieved-token budget exhausted]", left
            e.retrieved += n
            e.calls += 1
            e.papers += [p for p in papers if p not in e.papers]
            e.trace.append({"action": act, "arguments": args, "result_tokens": n, "papers": papers[:10]})
            e.messages = e.messages + [
                {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)},
                {"role": "user", "content": f"Tool result ({act}):\n{result}\n\n[{max_calls - e.calls} tool calls "
                                            f"left; {max(budget - e.retrieved, 0)} retrieved tokens left]"}]
    for e in eps:
        if not e.done:
            e.done, e.error, e.answer = True, "step_limit", ""


def majority(answers: list[str]) -> str:
    c = Counter(normalize_answer(a) for a in answers if a is not None)
    if not c:
        return ""
    best = c.most_common(1)[0][0]
    return next(a for a in answers if normalize_answer(a) == best)


# ---------------------------------------------------------------- vLLM binding (runs in the model's process)
def _vllm_generate(llm, tok, spec) -> Generate:
    def gen(msgs, schemas, temperature, seeds, max_tokens, n=1):
        prompts = [messages_prompt(tok, spec, m) for m in msgs]
        params = [json_params(s, max_tokens, temperature, seed=sd, n=n) for s, sd in zip(schemas, seeds, strict=True)]
        outs = llm.generate(prompts, params, use_tqdm=False)
        return [([c.text for c in o.outputs], len(o.prompt_token_ids or []),
                 sum(len(c.token_ids) for c in o.outputs)) for o in outs]
    return gen


def build_episodes(items: list[dict], arm: str, acfg: dict, seed: int, backend, tok, cfg: dict,
                   oracle: dict[str, str]) -> list[Episode]:
    mode, tools = acfg.get("mode", "agent"), list(acfg.get("tools") or [])
    max_calls, budget = int(cfg.get("max_tool_calls", 6)), int(cfg.get("max_retrieved_tokens", 8000))
    eps = []
    for it in items:
        user = it["question"]
        if mode == "rag":
            rows = backend.retrieve(it["question"], list(acfg.get("kinds", ["fulltext", "summary"])),
                                    int(acfg.get("k", 20)), bool(acfg.get("only_full_text", False)))
            text, _, _ = truncate_tokens(tok, backend.render(rows), budget)
            papers = list(dict.fromkeys(backend.units.iloc[rows]["paper_key"])) if rows else []
            user = f"Passages:\n<<<\n{text}\n>>>\n\nQuestion: {it['question']}"
        elif mode == "oracle":
            limit = min(budget * 2, int(cfg.get("_max_model_len", 32768)) - 2048)
            text, _, _ = truncate_tokens(tok, oracle.get(it.get("gold_paper_key") or "", ""), limit)
            papers = [it["gold_paper_key"]]
            user = f"Paper:\n<<<\n{text}\n>>>\n\nQuestion: {it['question']}"
        else:
            papers = []
        sysmsg = system_prompt(it, tools if mode == "agent" else [], max_calls, passages=mode in ("rag", "oracle"))
        eps.append(Episode(item=it, arm=arm, seed=seed, tools=tools if mode == "agent" else [],
                           view=acfg.get("view"), only_full_text=bool(acfg.get("only_full_text", False)),
                           messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
                           papers=papers))
    return eps


def _run_model(llm, tok, spec, payload: dict) -> list[dict]:
    from ..skill.backend import Backend
    cfg, items, arms = {**payload["cfg"], "_max_model_len": spec.get("max_model_len", 32768)}, payload["items"], payload["arms"]
    # query encoder on CPU (short queries, ~tens of ms): the whole GPU slice is left to vLLM, so every model fits
    # the same slice with the same context length
    backend = Backend(payload["index_dir"], payload["embedder"], payload["cache"], dense=payload.get("dense", True),
                      device=payload["cfg"].get("encoder_device", "cpu"))
    gen = _vllm_generate(llm, tok, spec)
    count = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])          # noqa: E731
    cut = lambda t, n: truncate_tokens(tok, t, n)[0]                             # noqa: E731
    temperature = float(cfg.get("temperature", 0.0))
    seeds = list(cfg.get("seeds", [0]))[: (1 if temperature == 0 else None)]
    rows = []
    for arm, acfg in arms.items():
        its = [i for i in items if not acfg.get("benchmarks") or i["bench"] in acfg["benchmarks"]]
        for seed in seeds:
            t0 = time.time()
            eps = build_episodes(its, arm, acfg, int(seed), backend, tok, cfg, payload["oracle"])
            if acfg.get("mode") == "sc":
                n = int(acfg.get("samples", 5))
                outs = gen([e.messages for e in eps], [action_schema([], answer_schema(e.item), False) for e in eps],
                           float(acfg.get("temperature", 0.7)), [e.seed for e in eps],
                           int(cfg.get("max_step_tokens", 400)), n)
                for e, (texts, n_in, n_out) in zip(eps, outs, strict=True):
                    ans = [str(((parse_json(t) or {}).get("arguments") or {}).get("answer", "")) for t in texts]
                    e.answer, e.done, e.steps, e.tokens_in, e.tokens_out = majority(ans), True, 1, n_in, n_out
                    e.gpu_seconds = (time.time() - t0) / max(len(eps), 1)
            else:
                run_agent(eps, gen, backend, count, cut, cfg, temperature)
            log(f"agent: {spec['name']} {arm} seed {seed}: {len(eps)} episodes in {time.time() - t0:.0f} s "
                f"(cache hits {backend.hits}, misses {backend.misses})")
            for e in eps:
                rows.append({"bench": e.item["bench"], "item_id": e.item["item_id"], "arm": arm, "seed": e.seed,
                             "model": spec["name"], "answer": e.answer, "error": e.error, "tool_calls": e.calls,
                             "retrieved_tokens": e.retrieved, "steps": e.steps, "tokens_in": e.tokens_in,
                             "tokens_out": e.tokens_out, "gpu_seconds": e.gpu_seconds,
                             "gold_retrieved": (e.item.get("gold_paper_key") in e.papers)
                             if e.item.get("gold_paper_key") else None,
                             "papers_seen": len(e.papers), "trace": json.dumps(e.trace, ensure_ascii=False)})
    return rows


def _payload(ctx: Context, items: pd.DataFrame, arms: dict, model: str) -> dict:
    from ..llm import embedder_spec
    cfg = ctx.section("ws3.agent")
    oracle = {}
    cp = ctx.cfg.stage_dir("ws2", "contexts") / "contexts.parquet"
    if cp.exists():
        c = pd.read_parquet(cp)
        oracle = dict(zip(c[c["arm"] == "full_text"]["paper_key"], c[c["arm"] == "full_text"]["context"], strict=True))
    recs = items.to_dict("records")
    for r in recs:
        r["options"] = list(r.get("options") if r.get("options") is not None else [])
    return {"cfg": cfg, "items": recs, "arms": arms, "oracle": oracle,
            "index_dir": str(ctx.cfg.stage_dir("ws3", "index")),
            "embedder": embedder_spec(ctx.cfg, ctx.section("ws3.index").get("embedder", "bge-m3")),
            "cache": str(ctx.out / "cache" / f"{model}.sqlite"), "dense": bool(cfg.get("dense", True))}


def planned_arms(ctx: Context) -> tuple[dict, list[dict]]:
    """Arms to run now, and arms that cannot run here (with the reason)."""
    cfg = ctx.section("ws3.agent")
    defs = cfg.get("arm_defs") or {}
    run, skipped = {}, []
    for a in cfg.get("arms", list(defs)):
        d = defs.get(a)
        if d is None:
            skipped.append({"arm": a, "reason": "no definition in ws3.agent.arm_defs"})
        elif d.get("mode") == "web" and not cfg.get("internet", False):
            skipped.append({"arm": a, "reason": "needs internet; Grete compute nodes are offline (ws3.agent.internet)"})
        elif any(t not in TOOLS for t in d.get("tools") or []):
            skipped.append({"arm": a, "reason": "unknown tool"})
        else:
            run[a] = d
    return run, skipped


@stage("ws3", "agent", kind="map", needs=("index", "benchmarks"), resources="gpu", env="llm",
       config_keys=("ws3.agent",),
       help="agent harness: every model x arm x benchmark item x seed (array task i runs models[i::N])")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws3.agent")
    items = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
    arms, skipped = planned_arms(ctx)
    ctx.write_json("arms.json", {"run": arms, "skipped": skipped})
    models = list(cfg.get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = []
    for name in models:
        dst = parts / f"{name}.parquet"
        if dst.exists() and not ctx.force:
            continue
        spec = model_spec(ctx.cfg, name)
        rows = with_model(spec, _run_model, _payload(ctx, items, arms, name),
                          gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.75)))
        pd.DataFrame(rows).to_parquet(dst, index=False)
        done.append(name)
    ctx.publish("WS3 agent runs", [
        {"type": "markdown", "text": f"Models run: {', '.join(sorted(p.stem for p in parts.glob('*.parquet')))}. "
                                     f"Arms: {', '.join(arms)}. Accuracy is in `results`."},
        {"type": "table", "title": "Arms not run on this cluster", "columns": [
            {"key": "arm", "label": "arm"}, {"key": "reason", "label": "reason"}], "rows": skipped}],
        summary=f"{len(list(parts.glob('*.parquet')))} models")
    return {"models_done": done, "arms": list(arms), "skipped_arms": [s["arm"] for s in skipped]}


# ---------------------------------------------------------------- pilot: unconstrained tool-call reliability
def valid_action(a: Any, tools: list[str]) -> tuple[bool, str]:
    if not isinstance(a, dict):
        return False, "not_json"
    act, args = a.get("action"), a.get("arguments")
    if act == "answer":
        return (isinstance(args, dict) and "answer" in args), "answer"
    if act not in tools:
        return False, "unknown_action"
    if not isinstance(args, dict):
        return False, "no_arguments"
    sch = TOOLS[act]["parameters"]
    if any(k not in args for k in sch.get("required", [])):
        return False, "missing_argument"
    for k, v in args.items():
        p = sch["properties"].get(k)
        if p is None:
            return False, "unknown_argument"
        if "enum" in p and v not in p["enum"]:
            return False, "bad_enum"
        if p.get("type") == "integer" and not isinstance(v, int):
            return False, "bad_type"
        if p.get("type") == "string" and not isinstance(v, str):
            return False, "bad_type"
    return True, "tool_call"


def _pilot(llm, tok, spec, payload: dict) -> list[dict]:
    gen = _vllm_generate(llm, tok, spec)
    tools = payload["tools"]
    msgs = [[{"role": "system", "content": system_prompt(it, tools, 6)}, {"role": "user", "content": it["question"]}]
            for it in payload["items"]]
    outs = gen(msgs, [None] * len(msgs), 0.0, [0] * len(msgs), 400)
    rows = []
    for it, (texts, _, _) in zip(payload["items"], outs, strict=True):
        ok, kind = valid_action(parse_json(texts[0]), tools)
        rows.append({"model": spec["name"], "item_id": it["item_id"], "well_formed": ok, "kind": kind,
                     "raw": texts[0][:300]})
    return rows


@stage("ws3", "pilot", kind="map", needs=("benchmarks",), resources="gpu", env="llm", config_keys=("ws3.pilot",),
       help="tool-call reliability per model without constrained decoding (gate: >= 95% well-formed)")
def run_pilot(ctx: Context) -> dict:
    cfg = ctx.section("ws3.pilot")
    items = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
    items = items.groupby("bench", group_keys=False).head(int(cfg.get("per_bench", 20)))
    recs = items.to_dict("records")
    for r in recs:
        r["options"] = list(r.get("options") if r.get("options") is not None else [])
    models = list(cfg.get("models") or ctx.section("ws3.agent").get("models", []))[ctx.shard_index::ctx.shard_count]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for name in models:
        if (parts / f"{name}.parquet").exists() and not ctx.force:
            continue
        rows = with_model(model_spec(ctx.cfg, name), _pilot, {"items": recs, "tools": list(cfg.get("tools", ["search", "get_paper"]))},
                          gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
        pd.DataFrame(rows).to_parquet(parts / f"{name}.parquet", index=False)
    import duckdb
    df = duckdb.sql(f"select * from read_parquet('{parts}/*.parquet')").df()
    gate = float(cfg.get("gate", 0.95))
    summ = df.groupby("model").agg(n=("well_formed", "size"), well_formed=("well_formed", "mean"),
                                   tool_calls=("kind", lambda k: float((k == "tool_call").mean()))).reset_index()
    summ["passes_gate"] = summ["well_formed"] >= gate
    kinds = df.groupby(["model", "kind"]).size().unstack(fill_value=0)
    ctx.write_json("reliability.json", summ.to_dict("records"))
    ctx.publish("WS3 pilot: tool-call reliability", [
        {"type": "table", "title": f"Well-formed first actions without constrained decoding (gate {gate:.0%})",
         "columns": [{"key": "model", "label": "model"}, {"key": "n", "label": "prompts", "fmt": "int"},
                     {"key": "well_formed", "label": "well-formed", "fmt": "pct1"},
                     {"key": "tool_calls", "label": "chose a tool", "fmt": "pct1"},
                     {"key": "passes_gate", "label": "passes gate"}], "rows": summ.to_dict("records")},
        {"type": "bar", "title": "First action by kind", "stacked": True, "horizontal": True, "fmt": "int",
         "labels": list(kinds.index), "series": [{"name": c, "data": kinds[c].tolist()} for c in kinds.columns]},
    ], summary=", ".join(f"{r.model}: {r.well_formed:.0%}" for r in summ.itertuples()))
    return {"models": summ.to_dict("records")}
