"""PaperQA2 worker for X3 (run with the Alex-pqa python: `python -m alexandria.x3.pqa_worker job.json`).

paper-qa's own pipeline (Docs.aadd -> chunking + hybrid dense / sparse embedding; Docs.aquery -> evidence
retrieval, LLM contextual summaries / re-ranking, cited answer). Every LLM call goes to the local vLLM
OpenAI-compatible server (litellm router "openai/local"; no OpenAI API); no network: doc-details lookups (Crossref /
Semantic Scholar) and media enrichment are off, and the job blocks every other outbound connection.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import pandas as pd

INSTRUCT = {"mc": "\n\nAnswer with the letter of the correct option. If the papers do not contain enough information, "
                  "choose the option 'Insufficient information to answer the question'. End with 'Answer: <letter>'.",
            "label": "\n\nAnswer with exactly one of: {opts}. End with 'Answer: <{opts}>'.",
            "short": "\n\nGive a short answer (a number with its unit, a name or a phrase). End with 'Answer: <short answer>'."}


def tokens(s) -> dict:
    """Prompt / completion tokens over every LLM call of one query (answer + evidence summaries; paper-qa's
    token_counts: model -> [prompt, completion])."""
    tc = getattr(s, "token_counts", None) or {}
    return {"tokens_in": int(sum(v[0] for v in tc.values())) if tc else None,
            "tokens_out": int(sum(v[1] for v in tc.values())) if tc else None}


def settings(job: dict):
    from paperqa import Settings
    from paperqa.settings import AnswerSettings, ParsingSettings
    cfg = job["cfg"]
    router = {"model_list": [{"model_name": "openai/local",
                              "litellm_params": {"model": "openai/local", "api_base": job["api_base"], "api_key": "local",
                                                 "temperature": 0.0, "timeout": 300}}]}
    parsing = ParsingSettings(use_doc_details=False, reader_config={"chunk_chars": int(cfg.get("chunk_chars", 3000)),
                                                                     "overlap": 250},
                              enrichment_llm="openai/local", enrichment_llm_config=router)
    try:
        from paperqa.settings import MultimodalOptions
        parsing.multimodal = MultimodalOptions.OFF
    except (ImportError, AttributeError):
        pass
    return Settings(llm="openai/local", llm_config=router, summary_llm="openai/local", summary_llm_config=router,
                    embedding=job["embedding"], temperature=0.0, parsing=parsing,
                    answer=AnswerSettings(evidence_k=int(cfg.get("evidence_k", 10)),
                                          answer_max_sources=int(cfg.get("answer_max_sources", 5)),
                                          max_concurrent_requests=int(cfg.get("max_concurrent", 8))))


async def main(job: dict) -> None:
    from paperqa import Docs
    st = settings(job)
    docs_meta = pd.read_parquet(job["docs"])
    docs = Docs()
    t0 = time.time()
    sem = asyncio.Semaphore(8)

    async def add(r):
        async with sem:
            await docs.aadd(f"{job['doc_dir']}/{r.paper_key}.txt", citation=r.title or r.paper_key, docname=r.paper_key,
                            dockey=r.paper_key, title=r.title, settings=st)
    await asyncio.gather(*(add(r) for r in docs_meta.itertuples()))
    print(f"pqa: {len(docs.docs)} docs added in {time.time() - t0:.0f} s", flush=True)
    t_add = time.time() - t0
    tok = None
    if job.get("tokenizer"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(job["tokenizer"])
    items = pd.read_parquet(job["items"]).to_dict("records")
    qsem = asyncio.Semaphore(int(job["cfg"].get("max_concurrent", 8)))

    async def ask(it):
        opts = ", ".join(list(it.get("options") if it.get("options") is not None else []))
        q = it["question"] + INSTRUCT[it["answer_type"]].format(opts=opts)
        async with qsem:
            t = time.time()
            try:
                s = await docs.aquery(q, settings=st)
                keys = sorted({c.text.doc.dockey for c in (s.contexts or []) if getattr(c, "score", 0) > 0})
                chunks = "\n\n".join(c.text.text for c in (s.contexts or []))     # evidence chunks retrieved for the query
                ret = len(tok(chunks, add_special_tokens=False)["input_ids"]) if tok is not None else None
                return {"item_id": it["item_id"], "bench": it["bench"], "raw": s.answer or "", "error": None,
                        "papers": json.dumps(keys), "seconds": time.time() - t, "retrieved_tokens": ret, **tokens(s)}
            except Exception as e:                                  # noqa: BLE001  (recorded, not fatal)
                return {"item_id": it["item_id"], "bench": it["bench"], "raw": "", "error": f"{type(e).__name__}: {e}"[:300],
                        "papers": "[]", "seconds": time.time() - t, "retrieved_tokens": None, "tokens_in": None,
                        "tokens_out": None}
    t1 = time.time()
    rows = await asyncio.gather(*(ask(it) for it in items))
    t_query = time.time() - t1
    df = pd.DataFrame(rows)
    df["gpu_seconds"] = t_query / max(len(df), 1)                   # the served model's GPU, shared by concurrent queries
    df["index_seconds_per_question"] = t_add / max(len(df), 1)      # one-off corpus embedding, amortised
    df.to_parquet(job["out"], index=False)
    print(f"pqa: {len(rows)} questions in {t_query:.0f} s (docs added in {t_add:.0f} s)", flush=True)


if __name__ == "__main__":
    asyncio.run(main(json.loads(open(sys.argv[1]).read())))
