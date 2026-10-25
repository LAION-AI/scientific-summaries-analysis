"""X3 systems: `alexandria` (WS3 arms), `openscholar` (OpenScholar-8B LM), `pqa-corpus` + `paperqa` (PaperQA2).

alexandria   The WS3 harness (alexandria.ws3.agent._run_model) over the WS3 index, arms A0 / R / S, with the shared
             answer model, on every X3 item.
openscholar  OpenScholar/Llama-3.1_OpenScholar-8B with OpenScholar's own answer prompt ("answer based on the provided
             references [n]"). Its 45M-paper datastore (0.74 TB) is not available offline here, so its passages come
             from Alexandria retrieval (hybrid top-k over full-text chunks and KU fields): "OpenScholar-8B LM +
             Alexandria retrieval". Conditions: alexandria (retrieved passages), gold (the benchmark's gold context,
             ScholarQABench only: an upper bound), none (closed book).
pqa-corpus   The PaperQA2 document collection: the full text of every WS3-index paper that has one, one .txt per
             paper (from the index's full-text chunks, overlaps removed).
paperqa      paper-qa (PaperQA2 pipeline: chunk retrieval with a hybrid dense + sparse embedding, LLM re-ranking and
             contextual summaries of the evidence, cited answer), fully open-source and local, no OpenAI API: the LLM
             is the shared open answer model served by vLLM's OpenAI-compatible server inside the job, embeddings are
             all-MiniLM-L6-v2 (sentence-transformers) + sparse, chunking uses tiktoken's open cl100k_base BPE file from a
             local cache; run in the Alex-pqa venv as a subprocess with every other outbound connection blocked.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..llm import chat_prompt, embedder_spec, json_params, model_spec, truncate_tokens, with_model
from ..runtime import Context, log, stage
from ..ws3.agent import _run_model
from .common import parse_answer, stage_table

OS_INSTRUCTION = ("Provide a detailed and informative answer only based on the provided references. Use the references "
                  "to support your answer and cite them with their numbers in brackets.")
FINAL = {"label": "End with a final line 'Answer: <one of: {opts}>'.",
         "mc": "End with a final line 'Answer: <the letter of one option>'.",
         "short": "End with a final line 'Answer: <a short answer: a number with its unit, a name or a phrase>'."}


def _items(ctx: Context) -> list[dict]:
    it = stage_table(ctx, "items", "items.parquet")
    recs = it.to_dict("records")
    for r in recs:
        r["options"] = list(r.get("options") if r.get("options") is not None else [])
    return recs


@stage("x3", "alexandria", needs=("items",), resources="gpu", env="llm", config_keys=("x3.alexandria", "x3.answer_model"),
       help="Alexandria (WS3 arms A0 / R / S over the WS3 index) on every X3 item, shared answer model")
def run_alexandria(ctx: Context) -> dict:
    cfg = ctx.section("x3.alexandria")
    name = ctx.section("x3").get("answer_model", "qwen2.5-7b")
    arms = {a: cfg["arm_defs"][a] for a in cfg.get("arms", ["A0", "R", "S"])}
    payload = {"cfg": cfg, "items": _items(ctx), "arms": arms, "oracle": {},
               "index_dir": str(ctx.cfg.stage_dir("ws3", "index")), "embedder": embedder_spec(ctx.cfg, "bge-m3"),
               "cache": str(ctx.out / "cache.sqlite"), "dense": bool(cfg.get("dense", True))}
    rows = with_model(model_spec(ctx.cfg, name), _run_model, payload,
                      gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.88)))
    df = pd.DataFrame(rows).assign(system=lambda d: "Alexandria " + d["arm"])
    df.to_parquet(ctx.out / "answers.parquet", index=False)
    ctx.publish("X3 Alexandria runs", [{"type": "kpis", "items": [
        {"label": "episodes", "value": len(df), "fmt": "int"}, {"label": "model", "value": name},
        {"label": "arms", "value": ", ".join(arms)}]}], summary=f"{len(df)} episodes ({name})")
    return {"episodes": len(df)}


# ---------------------------------------------------------------- OpenScholar
def _openscholar(llm, tok, spec, items: list[dict], cfg: dict) -> list[dict]:
    """Answers with their cost: retrieved (reference) tokens, prompt / generated tokens, GPU seconds (batch time / n)."""
    prompts, ret = [], []
    for it in items:
        refs = it["refs"]
        body = "\n\n".join(f"[{i}] {r}" for i, r in enumerate(refs)) if refs else ""
        body, n_ret, _ = truncate_tokens(tok, body, int(cfg.get("max_context_tokens", 10000)))
        ret.append(n_ret if refs else 0)
        fin = FINAL[it["answer_type"]].format(opts=", ".join(it["options"]))
        user = (f"{OS_INSTRUCTION}\nReferences:\n{body}\n\nQuestion: {it['question']}\n{fin}" if refs
                else f"Question: {it['question']}\n{fin}")
        prompts.append(chat_prompt(tok, spec, None, user))
    t0 = time.time()
    outs = llm.generate(prompts, json_params(None, int(cfg.get("max_tokens", 512))), use_tqdm=False)
    sec = (time.time() - t0) / max(len(prompts), 1)
    return [{"text": o.outputs[0].text if o.outputs else "", "retrieved_tokens": r, "tokens_in": len(o.prompt_token_ids or []),
             "tokens_out": len(o.outputs[0].token_ids) if o.outputs else 0, "gpu_seconds": sec}
            for o, r in zip(outs, ret, strict=True)]


def final_line(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.lower().lstrip("*# ").startswith("answer"):
            return ln.split(":", 1)[-1].strip(" *")
    return text or ""


@stage("x3", "openscholar", needs=("items",), resources="gpu", env="llm", config_keys=("x3.openscholar",),
       help="OpenScholar-8B LM with Alexandria retrieval / gold context / closed book (its 45M datastore is offline)")
def run_openscholar(ctx: Context) -> dict:
    from ..skill.backend import Backend
    cfg = ctx.section("x3.openscholar")
    items = _items(ctx)
    be = Backend(ctx.cfg.stage_dir("ws3", "index"), embedder_spec(ctx.cfg, "bge-m3"), ctx.out / "cache.sqlite",
                 device="cpu")
    jobs = []
    for it in items:
        rows = be.retrieve(it["question"], ["fulltext", "summary"], int(cfg.get("top_k", 8)))
        units = be.units.iloc[rows]
        retrieved = [f"{u.title}\n{str(u.text)[:1500]}" for u in units.itertuples()]
        gctx = it.get("gold_ctx")
        gold = [f"{c.get('title') or ''}\n{c.get('text') or ''}" for c in (json.loads(gctx) if isinstance(gctx, str) else [])]
        for cond, refs in (("alexandria", retrieved), ("none", []), *((("gold", gold),) if gold else ())):
            jobs.append({**it, "refs": refs, "condition": cond, "papers": list(dict.fromkeys(units["paper_key"]))})
    spec = model_spec(ctx.cfg, cfg.get("model", "openscholar-8b"))
    log(f"openscholar: {len(jobs)} prompts with {spec['hf']}")
    outs = with_model(spec, _openscholar, jobs, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    df = pd.DataFrame([{"bench": j["bench"], "item_id": j["item_id"], "system": f"OpenScholar-8B LM + {j['condition']}",
                        "raw": o["text"], "answer": parse_answer(final_line(o["text"]), j) or parse_answer(o["text"], j),
                        "gold_retrieved": (j["gold_paper_key"] in j["papers"]) if j.get("gold_paper_key") and j["condition"] == "alexandria" else None,
                        **{k: o[k] for k in ("retrieved_tokens", "tokens_in", "tokens_out", "gpu_seconds")}}
                       for j, o in zip(jobs, outs, strict=True)])
    df.to_parquet(ctx.out / "answers.parquet", index=False)
    ctx.publish("X3 OpenScholar runs", [{"type": "kpis", "items": [
        {"label": "answers", "value": len(df), "fmt": "int"},
        {"label": "retrieval", "value": "Alexandria (hybrid, WS3 index)", "hint": "OpenScholar's 0.74 TB datastore is not available offline"},
        {"label": "unparsed answers", "value": float((df["answer"] == "").mean()), "fmt": "pct1"}]}],
        summary=f"{len(df)} answers (OpenScholar-8B LM + Alexandria retrieval / gold / none)")
    return {"answers": len(df)}


# ---------------------------------------------------------------- PaperQA2
@stage("x3", "pqa-corpus", config_keys=("x3.paperqa",),
       help="PaperQA2 documents: the full text of every WS3-index paper (one .txt per paper)")
def run_pqa_corpus(ctx: Context) -> dict:
    cfg = ctx.section("x3.paperqa")
    u = pd.read_parquet(ctx.cfg.stage_dir("ws3", "index") / "units.parquet")
    ft = u[u["kind"] == "fulltext"].sort_values("uid")
    docs = ctx.out / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    meta = []
    overlap = int(ctx.cfg.raw.get("ws3", {}).get("index", {}).get("chunk_overlap", 50))
    for pk, g in ft.groupby("paper_key", sort=False):
        words: list[str] = []
        for i, t in enumerate(g["text"]):
            w = str(t).split()
            words += w if i == 0 else w[overlap:]                 # index chunks overlap by `chunk_overlap` words
        (docs / f"{pk}.txt").write_text(" ".join(words))
        meta.append({"paper_key": pk, "title": str(g["title"].iloc[0]), "words": len(words)})
    m = pd.DataFrame(meta)
    if cfg.get("max_papers"):
        m = m.head(int(cfg["max_papers"]))
    m.to_parquet(ctx.out / "docs.parquet", index=False)
    ctx.publish("X3 PaperQA2 corpus", [{"type": "kpis", "items": [
        {"label": "papers (full text)", "value": len(m), "fmt": "int", "hint": "the WS3 index papers that have full text"},
        {"label": "median words", "value": float(m["words"].median()), "fmt": "int"}]}], summary=f"{len(m)} documents")
    return {"docs": len(m)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@stage("x3", "paperqa", needs=("items", "pqa-corpus"), resources="gpu", env="train", config_keys=("x3.paperqa", "x3.answer_model"),
       help="PaperQA2 (paper-qa) over the WS3 corpus with the shared answer model served by vLLM (OpenAI API)")
def run_paperqa(ctx: Context) -> dict:
    from huggingface_hub import snapshot_download
    cfg = ctx.section("x3.paperqa")
    name = ctx.section("x3").get("answer_model", "qwen2.5-7b")
    spec = model_spec(ctx.cfg, name)
    hub = str(Path(ctx.cfg.site.hf_home) / "hub")
    local = snapshot_download(spec["hf"], revision=spec.get("revision"), cache_dir=hub, local_files_only=True)
    emb = snapshot_download(cfg["embedding"], revision=cfg.get("embedding_revision"), cache_dir=hub, local_files_only=True)
    port = _free_port()
    srv_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", local, "--served-model-name", "local",
               "--port", str(port), "--max-model-len", str(cfg.get("server_max_model_len", 14336)),
               "--gpu-memory-utilization", str(cfg.get("server_gpu_memory_utilization", 0.8))]
    if cfg.get("server_quantization"):
        srv_cmd += ["--quantization", str(cfg["server_quantization"])]
    log(f"paperqa: starting vLLM server for {name} on port {port}")
    srv = subprocess.Popen(srv_cmd, stdout=open(ctx.out / "vllm_server.log", "w"), stderr=subprocess.STDOUT)
    try:
        import urllib.request
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                break
            except Exception:                                      # noqa: BLE001
                if srv.poll() is not None:
                    raise RuntimeError("vLLM server exited; see vllm_server.log")
                time.sleep(5)
        job = {"items": str(ctx.cfg.stage_dir("x3", "items") / "items.parquet"),
               "docs": str(ctx.cfg.stage_dir("x3", "pqa-corpus") / "docs.parquet"),
               "doc_dir": str(ctx.cfg.stage_dir("x3", "pqa-corpus") / "docs"), "out": str(ctx.out / "answers.parquet"),
               "api_base": f"http://127.0.0.1:{port}/v1", "embedding": f"hybrid-st-{emb}", "cfg": cfg, "tokenizer": local,
               "index_dir": str(ctx.out / "pqa_index")}
        (ctx.out / "job.json").write_text(json.dumps(job, default=str))
        py = str(cfg.get("python")).replace("${site.store}", str(ctx.cfg.site.store))
        # everything local and open: no OpenAI API (the key only satisfies the OpenAI-compatible client of the local
        # vLLM server); tiktoken's open cl100k_base BPE file (paper-qa's chunker) from a cache filled on the login node;
        # any other outbound call goes to a dead proxy and fails at once instead of hanging on the offline node
        env = {**os.environ, "OPENAI_API_KEY": "local", "PQA_HOME": str(ctx.out / "pqa_home"),
               "TIKTOKEN_CACHE_DIR": str(Path(ctx.cfg.site.store) / "tiktoken_cache"),
               "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "ANONYMIZED_TELEMETRY": "False",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HTTP_PROXY": "http://127.0.0.1:9",
               "HTTPS_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
               "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "no_proxy": "127.0.0.1,localhost"}
        log(f"paperqa: running the worker in {py}")
        subprocess.run([py, "-m", "alexandria.x3.pqa_worker", str(ctx.out / "job.json")], check=True, env=env)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
    df = pd.read_parquet(ctx.out / "answers.parquet")
    ctx.publish("X3 PaperQA2 runs", [{"type": "kpis", "items": [
        {"label": "answers", "value": len(df), "fmt": "int"}, {"label": "LLM", "value": f"{name} (vLLM, OpenAI API)"},
        {"label": "corpus", "value": "WS3 index papers with full text"},
        {"label": "components", "value": "open-source, local (no OpenAI API)",
         "hint": "LLM: local vLLM; embeddings: all-MiniLM-L6-v2 + sparse; tokenizer: cached cl100k_base BPE"},
        {"label": "failed queries", "value": float(df["error"].notna().mean()), "fmt": "pct1"}]}],
        summary=f"{len(df)} PaperQA2 answers")
    return {"answers": len(df)}
