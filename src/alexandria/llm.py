"""Shared LLM helpers for the GPU stages (vLLM, Alex-vllm env): model registry, chat prompts, JSON-constrained
sampling, context truncation, and one process per model.

    spec = model_spec(cfg, "qwen2.5-7b")            # configs/models.yaml entry (+ name)
    with_model(spec, fn, *args)                      # run fn(llm, tok, spec, *args) in a fresh process, return result
    prompt = chat_prompt(tok, spec, system, user)    # the model's own chat template
    params = json_params(schema, max_tokens=256)     # structured output: the answer always parses

One process per model: vLLM does not reliably free GPU memory when an engine is deleted, so a stage that loops over
models runs each in a spawned child and only collects its return value.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from typing import Any, Callable

from .runtime import log


def model_spec(cfg, name: str) -> dict[str, Any]:
    """Registry entry + the global overrides in site.llm (quantization, max_model_len cap), which every GPU stage
    applies the same way and every manifest records (e.g. fp8 weights to fit 7-8B models on 20 GB MIG slices)."""
    models = cfg.raw.get("models") or {}
    if name not in models:
        raise KeyError(f"model {name!r} is not in configs/models.yaml (known: {', '.join(models)})")
    spec = {"name": name, **models[name]}
    over = (cfg.raw.get("site") or {}).get("llm") or {}
    if over.get("quantization"):
        spec["quantization"] = over["quantization"]
    if over.get("max_model_len"):
        spec["max_model_len"] = min(int(spec.get("max_model_len", 32768)), int(over["max_model_len"]))
    return spec


def embedder_spec(cfg, name: str) -> dict[str, Any]:
    emb = cfg.raw.get("embedders") or {}
    if name not in emb:
        raise KeyError(f"embedder {name!r} is not in configs/models.yaml (known: {', '.join(emb)})")
    return {"name": name, **emb[name]}


def load_llm(spec: dict, gpu_memory_utilization: float = 0.85, max_model_len: int | None = None):
    from vllm import LLM
    llm = LLM(model=spec["hf"], revision=spec.get("revision"), tokenizer_revision=spec.get("revision"),
              max_model_len=int(max_model_len or spec.get("max_model_len", 32768)),
              gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True, seed=0,
              trust_remote_code=bool(spec.get("trust_remote_code", False)), quantization=spec.get("quantization"))
    return llm, llm.get_tokenizer()


def chat_prompt(tok, spec: dict, system: str | None, user: str, history: list[dict] | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + list(history or []) + \
        [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   **(spec.get("chat_kwargs") or {}))


def messages_prompt(tok, spec: dict, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                   **(spec.get("chat_kwargs") or {}))


def json_params(schema: dict | None, max_tokens: int, temperature: float = 0.0, seed: int | None = None,
                n: int = 1, top_p: float = 1.0):
    """SamplingParams, optionally constrained to a JSON schema (vLLM >= 0.10.2 and older API)."""
    from vllm import SamplingParams
    kw: dict[str, Any] = dict(temperature=temperature, max_tokens=max_tokens, n=n, top_p=top_p, seed=seed)
    if schema is None:
        return SamplingParams(**kw)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**kw, structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**kw, guided_decoding=GuidedDecodingParams(json=schema))


def parse_json(text: str) -> dict | None:
    s = (text or "").strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1]
    try:
        return json.loads(s[s.find("{"): s.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def truncate_tokens(tok, text: str, max_tokens: int) -> tuple[str, int, bool]:
    """(text cut to max_tokens tokens, its token count, was it cut)."""
    ids = tok(text or "", add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text or "", len(ids), False
    return tok.decode(ids[:max_tokens]), max_tokens, True


def _child(q, fn, spec, gpu_mem, max_len, args):
    try:
        llm, tok = load_llm(spec, gpu_mem, max_len)
        q.put(("ok", fn(llm, tok, spec, *args)))
    except BaseException as e:                         # noqa: BLE001  (re-raised in the parent)
        import traceback
        q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def with_model(spec: dict, fn: Callable, *args, gpu_memory_utilization: float = 0.85,
               max_model_len: int | None = None) -> Any:
    """Run fn(llm, tok, spec, *args) in a spawned process with the model loaded; return its (picklable) result.
    fn must be a module-level function. Set ALEXANDRIA_INPROCESS=1 to run in this process (tests, debugging)."""
    if os.environ.get("ALEXANDRIA_INPROCESS") == "1":
        llm, tok = load_llm(spec, gpu_memory_utilization, max_model_len)
        return fn(llm, tok, spec, *args)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, fn, spec, gpu_memory_utilization, max_model_len, args))
    log(f"model {spec['name']} ({spec['hf']}): starting a worker process")
    p.start()
    status, payload = q.get()
    p.join()
    if status != "ok":
        raise RuntimeError(f"model {spec['name']} failed:\n{payload}")
    return payload
