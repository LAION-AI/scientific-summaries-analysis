"""Stages `grounding` and `grounding-report`: sentence-level grounding of summary fields with MiniCheck.

MiniCheck (Tang, Laban & Durrett, 2024, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents");
model bespokelabs/Bespoke-MiniCheck-7B. The prompt, chunking and scoring follow the reference implementation
(github.com/Liyan06/MiniCheck, minicheck/utils.py SYSTEM_PROMPT / USER_PROMPT and minicheck/inference.py):
  * every summary sentence is checked against every source chunk of <= max_model_len - 300 tokens
    (chunks built from newline blocks and sentences, counted with the model tokenizer);
  * one generated token, temperature 0, top-5 logprobs; P(support) = sum of exp(logprob) of the top-5 tokens whose
    text is "yes" (case-insensitive);
  * a sentence's score is the max over chunks; a field is supported when the min over its sentences is > 0.5
    (we also store the mean).
Runs in the vLLM env (Alex-vllm) on one GPU. The paper sample is built here from `features` (has_text and not
source_garbled; stratified by group; debug profile = every eligible paper) and processed in batches; each batch is
one file in parts/, so a re-submitted job resumes at the first missing batch.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow as pa

from ...papers import paper_table, with_paper_info
from ...runtime import Context, atomic_write_parquet, log, stage
from ...stats import rate_row
from .common import features_table, fetch_records, field_items, paper_key, source_text, split_sentences, \
    stratified_pick

SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""
USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""


def yes_prob(top_logprobs) -> float:
    """P(support) from vLLM's top-k logprobs of the first generated token (dict token_id -> Logprob)."""
    p = 0.0
    for lp in (top_logprobs or {}).values():
        tok = getattr(lp, "decoded_token", None) or ""
        if tok.strip().lower() == "yes":
            p += math.exp(lp.logprob)
    return min(p, 1.0)


def aggregate(sentence_chunk_probs: list[list[float]]) -> tuple[list[float], float | None, float | None]:
    """Per sentence: max over chunks. Field: (min, mean) over sentences."""
    sent = [max(ps) if ps else 0.0 for ps in sentence_chunk_probs]
    if not sent:
        return [], None, None
    return sent, float(min(sent)), float(np.mean(sent))


def chunk_document(text: str, tokenizer, chunk_tokens: int) -> list[str]:
    """Greedy chunks of whole sentences (newlines kept) of at most chunk_tokens tokens, as in the reference."""
    units: list[str] = []
    for block in text.split("\n"):
        sents = split_sentences(block) if block.strip() else []
        units.extend(sents)
        units.append("\n")
    if not units:
        return [text] if text else []
    lens = [len(x) for x in tokenizer([u for u in units], add_special_tokens=False)["input_ids"]]
    chunks, cur, cur_len = [], [], 0
    for u, n in zip(units, lens, strict=True):
        if cur and cur_len + n > chunk_tokens:
            chunks.append(" ".join(cur).replace(" \n ", "\n").strip())
            cur, cur_len = [], 0
        if n > chunk_tokens:                                     # one giant "sentence": hard split by tokens
            ids = tokenizer(u, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), chunk_tokens):
                chunks.append(tokenizer.decode(ids[i:i + chunk_tokens]))
            continue
        cur.append(u)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur).replace(" \n ", "\n").strip())
    return [c for c in chunks if c.strip()]


def sample_papers(ctx: Context, n: int, seed: int = 0) -> pd.DataFrame:
    feats = features_table(ctx, ["file", "row", "paper_id", "group", "input_type", "domain", "has_text",
                                 "source_garbled"])
    ok = feats[feats["has_text"].fillna(False) & ~feats["source_garbled"].fillna(False)]
    if ctx.cfg.profile == "debug":                 # small GPU test: profiles.debug.gpu_papers
        n = int(ctx.cfg.profile_settings.get("gpu_papers", n))
    pick = stratified_pick(ok, n, ["group"], seed, min_per=1, debug_all=False)
    pick = pick.sort_values(["file", "row"]).reset_index(drop=True)
    pick["paper_key"] = [paper_key(f, r) for f, r in zip(pick["file"], pick["row"], strict=True)]
    return pick


@stage("ws1", "grounding", needs=("features",), resources="gpu", env="llm", config_keys=("ws1b.grounding",),
       help="MiniCheck-7B sentence-level grounding of summary fields against the source (vLLM)")
def run(ctx: Context) -> dict:
    from vllm import LLM, SamplingParams                          # only in the Alex-vllm env
    cfg = ctx.section("ws1b.grounding")
    fields = list(cfg.get("fields", ["executive_summary", "key_results"]))
    max_sent = int(cfg.get("max_sentences_per_field", 12))
    max_len = int(cfg.get("max_model_len", 32768))
    chunk_tokens = max_len - int(cfg.get("chunk_reserve_tokens", 300))
    batch_papers = int(cfg.get("batch_papers", 32))

    papers = sample_papers(ctx, int(cfg.get("n_papers", 10000)), int(cfg.get("seed", 0)))
    ctx.write_json("sample.json", {"papers": len(papers)})
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    batches = [papers.iloc[i:i + batch_papers] for i in range(0, len(papers), batch_papers)]
    todo = [(k, b) for k, b in enumerate(batches) if ctx.force or not (parts / f"batch_{k:05d}.parquet").exists()]
    log(f"grounding: {len(papers)} papers, {len(batches)} batches, {len(todo)} to do")
    if not todo:
        return {"papers": len(papers), "batches": len(batches), "done": 0}

    llm = LLM(model=cfg.get("model", "bespokelabs/Bespoke-MiniCheck-7B"), max_model_len=max_len,
              enable_prefix_caching=True, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
              trust_remote_code=bool(cfg.get("trust_remote_code", False)), revision=cfg.get("revision"),
              tokenizer_revision=cfg.get("revision"), seed=0)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=5)
    rows_total = 0
    for k, batch in todo:
        recs = fetch_records(ctx.cfg, zip(batch["file"], batch["row"], strict=True),
                             ["paper_id", "text_raw", "text_sanitized", *fields])
        prompts, index, meta = [], [], []                          # index: (sentence slot, chunk)
        for p in batch.itertuples():
            rec = recs.get((p.file, int(p.row)))
            if rec is None:
                continue
            chunks = chunk_document(source_text(rec), tok, chunk_tokens)
            for f in fields:
                sents = []
                for item in field_items(rec, f):
                    sents.extend(split_sentences(item))
                for si, s in enumerate(sents[:max_sent]):
                    slot = len(meta)
                    meta.append({"paper_key": p.paper_key, "paper_id": p.paper_id, "file": p.file, "row": int(p.row),
                                 "group": p.group, "input_type": p.input_type, "domain": p.domain, "field": f,
                                 "sent_index": si, "sentence": s, "n_chunks": len(chunks)})
                    for c in chunks:                               # same chunk first in every prompt -> prefix cache
                        msg = [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": USER_PROMPT.replace("[DOCUMENT]", c).replace("[CLAIM]", s)}]
                        prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False))
                        index.append(slot)
        probs: list[list[float]] = [[] for _ in meta]
        if prompts:
            for out, slot in zip(llm.generate(prompts, params, use_tqdm=False), index, strict=True):
                lp = out.outputs[0].logprobs[0] if out.outputs and out.outputs[0].logprobs else {}
                probs[slot].append(yes_prob(lp))
        sent_scores, _, _ = aggregate(probs)
        rows = [{**m, "prob": s, "supported": s > 0.5} for m, s in zip(meta, sent_scores, strict=True)]
        atomic_write_parquet(parts / f"batch_{k:05d}.parquet", pa.Table.from_pylist(rows) if rows else
                             pa.table({"paper_key": pa.array([], pa.string())}))
        rows_total += len(rows)
        log(f"grounding: batch {k + 1}/{len(batches)} ({len(rows)} sentences, {len(prompts)} prompts)")
    return {"papers": len(papers), "batches": len(batches), "done": len(todo), "sentences": rows_total}


# ---------------------------------------------------------------- report
def load_sentences(ctx: Context, stage_name: str = "grounding") -> pd.DataFrame:
    import duckdb
    parts = ctx.parts(stage_name)
    if not parts:
        return pd.DataFrame()
    df = duckdb.sql(f"select * from read_parquet({[str(p) for p in parts]!r}, union_by_name=true)").df()
    return df[df["sentence"].notna()] if "sentence" in df else pd.DataFrame()


def field_scores(sent: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, field): min and mean sentence score, supported = min > 0.5."""
    g = sent.groupby(["paper_key", "file", "row", "group", "input_type", "domain", "field"], dropna=False)["prob"]
    out = g.agg(score_min="min", score_mean="mean", n_sent="size").reset_index()
    out["supported"] = out["score_min"] > 0.5
    return out


def rate_table(df: pd.DataFrame, by: list[str], flag: str) -> list[dict]:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key, strict=True)), **rate_row(int(g[flag].sum()), len(g))})
    return rows


@stage("ws1", "grounding-report", needs=("grounding", "features"), config_keys=("ws1b.grounding",),
       help="supported rates (sentence / field) by group, input type and field, with Wilson CIs")
def run_report(ctx: Context) -> dict:
    sent = load_sentences(ctx)
    if sent.empty:
        ctx.publish("MiniCheck grounding", [{"type": "markdown", "text": "No grounding results yet."}])
        return {"sentences": 0}
    fields = field_scores(sent)
    by_field = rate_table(sent, ["field"], "supported")
    by_group = rate_table(sent, ["group"], "supported")
    by_input = rate_table(sent, ["input_type"], "supported")
    field_level = rate_table(fields, ["field", "input_type"], "supported")
    hist, edges = np.histogram(sent["prob"].to_numpy(dtype=float), bins=20, range=(0, 1))
    pp = sent.groupby(["paper_key", "file", "row", "input_type"], dropna=False).agg(
        sentences=("prob", "size"), supported=("supported", "mean"), min_p=("prob", "min")).reset_index()
    weak = fields[~fields["supported"]].groupby("paper_key")["field"].agg(lambda x: ", ".join(sorted(x)))
    pp["weak_fields"] = pp["paper_key"].map(weak).fillna("")
    per_paper = with_paper_info(ctx, pp.sort_values(["supported", "min_p"]))
    per_paper.to_csv(ctx.out / "per_paper.csv", index=False)
    ctx.write_json("summary.json", {"by_field": by_field, "by_group": by_group, "by_input_type": by_input,
                                    "field_level": field_level})
    atomic_write_parquet(ctx.out / "field_scores.parquet", pa.Table.from_pandas(fields, preserve_index=False))

    def rate_cols(first: list[tuple[str, str]]) -> list[dict]:
        return [{"key": k, "label": lab} for k, lab in first] + [
            {"key": "n", "label": "n", "fmt": "int"}, {"key": "rate", "label": "supported", "fmt": "pct1"},
            {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"}]

    overall = rate_row(int(sent["supported"].sum()), len(sent))
    field_overall = rate_row(int(fields["supported"].sum()), len(fields))
    ctx.publish("MiniCheck grounding", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": int(sent["paper_key"].nunique()), "fmt": "int"},
            {"label": "sentences checked", "value": len(sent), "fmt": "int"},
            {"label": "sentences supported", "value": overall["rate"], "fmt": "pct1",
             "hint": f"95% CI {overall['lo']:.1%}–{overall['hi']:.1%}"},
            {"label": "fields fully supported", "value": field_overall["rate"], "fmt": "pct1",
             "hint": "every sentence P(support) > 0.5"}]},
        {"type": "bar", "title": "Supported sentences per field", "horizontal": True, "fmt": "pct",
         "labels": [r["field"] for r in by_field],
         "series": [{"name": "supported", "data": [r["rate"] for r in by_field],
                     "lo": [r["lo"] for r in by_field], "hi": [r["hi"] for r in by_field]}]},
        {"type": "bar", "title": "Supported sentences by input type", "fmt": "pct",
         "labels": [r["input_type"] for r in by_input],
         "series": [{"name": "supported", "data": [r["rate"] for r in by_input],
                     "lo": [r["lo"] for r in by_input], "hi": [r["hi"] for r in by_input]}]},
        {"type": "bar", "title": "P(support) distribution", "fmt": "int",
         "labels": [f"{edges[i]:.2f}" for i in range(len(hist))], "series": [{"name": "sentences", "data": hist.tolist()}]},
        {"type": "table", "title": "By group", "columns": rate_cols([("group", "group")]), "rows": by_group},
        {"type": "table", "title": "Field-level support by input type",
         "columns": rate_cols([("field", "field"), ("input_type", "input type")]), "rows": field_level, "limit": 200},
        paper_table("Papers checked by MiniCheck", per_paper,
                    [("input_type", "input", None), ("sentences", "sentences", "int"),
                     ("supported", "supported", "pct1"), ("min_p", "lowest P(support)", "float3"),
                     ("weak_fields", "fields not fully supported", None)],
                    hint="least supported papers first; full list in per_paper.csv"),
    ], summary=f"{overall['rate']:.1%} of {len(sent)} sentences supported")
    return {"sentences": len(sent), "supported_rate": overall["rate"]}
