"""Stages `data` and `decontam` (WS4, CPU).

data      Document-style training corpora, one per condition, all built from the SAME trained papers and matched in
          tokens (reference tokenizer: the first training model). Budget B = `token_budget`, or the `summary`
          condition's total tokens x `budget_multiplier`. Paper conditions fill B round-robin over the papers:
          summary / ku repeat (epochs), `diverse` renders each repeat in another format (prose, JSON, YAML, HTML,
          bullet Q&A) instead of repeating the same text, `full` walks through each paper's text in sequence-length
          chunks (all papers' first chunks first). `generic` = FineWeb-Edu documents (int_score >= 4, disjoint from
          replay). Every condition then gets the same replay tokens (FineWeb-Edu, share `replay_share`); `none` is
          replay only, same total tokens ("no extra data" control).
decontam  Removes training examples that share a 13-gram (normalised words) with any evaluation item (questions,
          options, answers of every WS4 eval set, the knowledge probe and the WS3 items), and QA pairs whose question
          shares an 8-gram with a probe question. Writes the cleaned training files the trainers read and a removal
          table per condition and eval set.
"""
from __future__ import annotations

import json

import pandas as pd

from ..runtime import Context, log, stage
from .common import (DOC_CONDITIONS, PAPER_CONDITIONS, condition_text, load_resource, ngrams, setting,
                     stage_table, words)

EVAL_FILES = {   # eval set -> (HF dataset repo, file, text columns); files fetched by `alexandria fetch-ws4`
    "mmlu_pro_stem": ("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet", ["question", "options", "answer"]),
    "sciq": ("allenai/sciq", "data/test-00000-of-00001.parquet", ["question", "correct_answer", "support"]),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet", ["question", "choices"]),
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet", ["question", "answer"]),
    "ifeval": ("google/IFEval", "ifeval_input_data.jsonl", ["prompt"]),
    "pubmedqa_closed": ("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", ["question", "long_answer"]),
    "gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond.csv", ["Question", "Correct Answer"]),
}


def reference_tokenizer(ctx: Context):
    import os

    from transformers import AutoTokenizer

    from ..llm import model_spec
    os.environ.setdefault("HF_HOME", str(ctx.cfg.site.hf_home))
    models = setting(ctx, "models", ctx.section("ws4.train").get("models", ["qwen3-4b"]))
    spec = model_spec(ctx.cfg, models[0])
    return AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision")), spec["name"]


def _ntok(tok, texts: list[str]) -> list[int]:
    return [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]


def _chunks(tok, text: str, size: int) -> list[str]:
    ids = tok(text or "", add_special_tokens=False)["input_ids"]
    return [tok.decode(ids[i:i + size]) for i in range(0, len(ids), size) if len(ids[i:i + size]) >= 32]


def _fill(stream, budget: int) -> list[dict]:
    out, n = [], 0
    for row in stream:
        if n >= budget:
            break
        out.append(row)
        n += row["tokens"]
    return out


@stage("ws4", "data", needs=("papers",), config_keys=("ws4.data", "ws4.conditions"),
       help="document-style corpora per condition from the same papers, token-matched, with replay")
def run_data(ctx: Context) -> dict:
    cfg = ctx.section("ws4.data")
    conds = [c for c in ctx.section("ws4").get("conditions", DOC_CONDITIONS) if c in DOC_CONDITIONS]
    papers = stage_table(ctx, "papers", "papers.parquet")
    papers = papers[papers["split"] == "train"].reset_index(drop=True)
    plist = [{"i": i, "paper_key": p.paper_key, "ku": json.loads(p.ku), "full": p.full} for i, p in enumerate(papers.itertuples())]
    tok, tok_name = reference_tokenizer(ctx)
    seq, formats = int(cfg.get("doc_max_tokens", 2048)), list(cfg.get("formats", ["prose", "json", "yaml", "html", "qa_bullets"]))
    base = sum(_ntok(tok, [condition_text("summary", p, 0, formats) for p in plist]))
    budget = int(cfg.get("token_budget") or base * float(cfg.get("budget_multiplier", 2)))
    share = float(cfg.get("replay_share", 0.25))
    total = int(budget / (1 - share))
    log(f"data: {len(plist)} papers, summary tokens {base}, budget {budget} + replay -> {total} tokens per condition")

    def paper_stream(cond):
        if cond == "full":
            chunks = [_chunks(tok, p["full"], seq) for p in plist]
            for k in range(max(map(len, chunks), default=0)):
                for p, ch in zip(plist, chunks, strict=True):
                    if k < len(ch):
                        yield {"text": ch[k], "paper_key": p["paper_key"], "fmt": "full", "epoch": 0, "chunk": k}
            return
        epoch = 0
        while True:
            texts = [condition_text(cond, p, epoch, formats) for p in plist]
            for p, t in zip(plist, texts, strict=True):
                for ch in _chunks(tok, t, seq):
                    yield {"text": ch, "paper_key": p["paper_key"], "epoch": epoch,
                           "fmt": {"summary": "prose", "ku": "json"}.get(cond, formats[(epoch + p["i"]) % len(formats)])}
            epoch += 1

    def resource_stream(name, skip=0):
        r = ctx.cfg.raw["ws4_resources"][name]
        df = load_resource(ctx, name, max_rows=200000)
        if r.get("min_int_score"):
            df = df[df["int_score"] >= r["min_int_score"]]
        for t in df[r.get("text", "text")].iloc[skip:]:
            for ch in _chunks(tok, t, seq):
                yield {"text": ch, "paper_key": None, "fmt": name, "epoch": 0}

    def with_tokens(stream):
        for row in stream:
            row["tokens"] = len(tok(row["text"], add_special_tokens=False)["input_ids"])
            yield row

    replay = _fill(with_tokens(resource_stream(cfg.get("replay_docs", "fineweb-edu"))), total)
    stats = []
    for cond in conds:
        if cond in PAPER_CONDITIONS:
            data = _fill(with_tokens(paper_stream(cond)), budget)
        elif cond == "generic":
            data = _fill(with_tokens(resource_stream(cfg.get("generic_docs", "fineweb-edu-generic"))), budget)
        else:                                                         # none: replay only, same total tokens
            data = []
        rep = _fill(iter(replay), total - sum(r["tokens"] for r in data))
        df = pd.DataFrame([{**r, "source": "data"} for r in data] + [{**r, "source": "replay"} for r in rep])
        df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        df.to_parquet(ctx.out / f"docs_{cond}.parquet", index=False)
        d = df[df["source"] == "data"]
        stats.append({"condition": cond, "docs": len(df), "tokens": int(df["tokens"].sum()),
                      "data_tokens": int(d["tokens"].sum()), "replay_tokens": int(df.loc[df["source"] == "replay", "tokens"].sum()),
                      "papers": int(d["paper_key"].nunique()), "epochs": int(d["epoch"].max() + 1) if len(d) else 0,
                      "chunks_per_paper": float(d.groupby("paper_key").size().mean()) if len(d) and d["paper_key"].notna().any() else 0.0,
                      "formats": ", ".join(sorted(d["fmt"].dropna().unique()))})
    st = pd.DataFrame(stats)
    st.to_csv(ctx.out / "data_stats.csv", index=False)
    ctx.write_json("budget.json", {"tokenizer": tok_name, "summary_tokens": base, "budget": budget,
                                   "replay_share": share, "total": total, "seq": seq})
    ctx.publish("WS4 training data (document style)", [
        {"type": "kpis", "items": [
            {"label": "trained papers", "value": len(plist), "fmt": "int"},
            {"label": "data tokens per condition", "value": budget, "fmt": "int", "hint": f"tokenizer: {tok_name}"},
            {"label": "replay share", "value": share, "fmt": "pct1"},
            {"label": "conditions", "value": len(conds), "fmt": "int"}]},
        {"type": "table", "title": "Token-matched corpora per condition", "rows": stats,
         "hint": "same papers, same total tokens; short representations repeat (epochs), `diverse` rotates formats",
         "columns": [{"key": "condition", "label": "condition"}, {"key": "data_tokens", "label": "data tokens", "fmt": "int"},
                     {"key": "replay_tokens", "label": "replay tokens", "fmt": "int"},
                     {"key": "papers", "label": "papers", "fmt": "int"}, {"key": "epochs", "label": "epochs (repeats)", "fmt": "int"},
                     {"key": "chunks_per_paper", "label": "chunks / paper", "fmt": "float"},
                     {"key": "formats", "label": "formats"}]},
    ], summary=f"{len(conds)} conditions x {total} tokens")
    return {"budget": budget, "total": total, "conditions": conds}


# ---------------------------------------------------------------- decontamination
def eval_texts(ctx: Context) -> dict[str, list[str]]:
    """Every evaluation item's text, per eval set (missing / gated sets are skipped and reported)."""
    from huggingface_hub import hf_hub_download
    out: dict[str, list[str]] = {}
    for name, (repo, f, cols) in EVAL_FILES.items():
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                                local_files_only=True)
        except Exception:                                                # noqa: BLE001  (gated / not fetched)
            continue
        df = pd.read_parquet(p) if f.endswith(".parquet") else pd.read_json(p, lines=True) if f.endswith(".jsonl") \
            else pd.read_csv(p)
        out[name] = [" ".join(str(x) for x in r) for r in df[[c for c in cols if c in df.columns]].itertuples(index=False)]
    for name, (st, fn, cols) in {"ws4_probe": ("probe", "probe.parquet", ["question", "answer"])}.items():
        p = ctx.cfg.stage_dir("ws4", st) / fn
        if p.exists():
            out[name] = [" ".join(map(str, r)) for r in pd.read_parquet(p, columns=cols).itertuples(index=False)]
    p = ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet"
    if p.exists():
        out["ws3_items"] = pd.read_parquet(p, columns=["question"])["question"].astype(str).tolist()
    return out


@stage("ws4", "decontam", needs=("data", "qa-data", "probe"), config_keys=("ws4.decontam",),
       help="drop training examples sharing a 13-gram with any eval item (8-gram on probe questions)")
def run_decontam(ctx: Context) -> dict:
    cfg = ctx.section("ws4.decontam")
    n, nq = int(cfg.get("ngram", 13)), int(cfg.get("min_ngram_question", 8))
    ev = eval_texts(ctx)
    index: dict[tuple, str] = {}
    for name, texts in ev.items():
        for t in texts:
            for g in ngrams(t, n):
                index.setdefault(g, name)
    probe_q: set = set()
    pp = ctx.cfg.stage_dir("ws4", "probe") / "probe.parquet"
    if pp.exists():
        for q in pd.read_parquet(pp, columns=["question"])["question"]:
            probe_q |= ngrams(q, nq)
    log(f"decontam: {len(index)} {n}-grams from {len(ev)} eval sets; {len(probe_q)} probe {nq}-grams")

    def hit(text: str, question: str | None = None) -> str | None:
        w = words(text)
        for i in range(len(w) - n + 1):
            name = index.get(tuple(w[i:i + n]))
            if name:
                return name
        if question is not None and ngrams(question, nq) & probe_q:
            return "ws4_probe"
        return None

    removed = []
    for src in sorted(ctx.cfg.stage_dir("ws4", "data").glob("docs_*.parquet")) + \
            sorted(ctx.cfg.stage_dir("ws4", "qa-data").glob("*.parquet")):
        df = pd.read_parquet(src)
        if "text" in df:
            why = [hit(t) for t in df["text"]]
        elif "question" in df:
            why = [hit(f"{q} {a}", q) for q, a in zip(df["question"], df["answer"], strict=True)]
        else:                                                         # chat replay rows: prompt / completion JSON
            why = [hit(f"{p} {c}") for p, c in zip(df["prompt"], df["completion"], strict=True)]
        df["contaminated_by"] = why
        keep = df[df["contaminated_by"].isna()].drop(columns="contaminated_by")
        keep.to_parquet(ctx.out / src.name, index=False)
        for name, k in df["contaminated_by"].value_counts().items():
            removed.append({"file": src.name, "eval_set": name, "removed": int(k)})
        removed.append({"file": src.name, "eval_set": "(kept)", "removed": len(keep)})
    # decontamination removes different numbers of pairs per condition: re-equalise the QA files (per paper: the
    # minimum count over the paper conditions; generic: the same total), so QA runs stay count-matched
    qa = {c: pd.read_parquet(ctx.out / f"qa_{c}.parquet") for c in PAPER_CONDITIONS if (ctx.out / f"qa_{c}.parquet").exists()}
    if qa:
        per = pd.concat([d.groupby("paper_key").size().rename(c) for c, d in qa.items()], axis=1).fillna(0).min(axis=1)
        for c, d in qa.items():
            d = d.groupby("paper_key", group_keys=False).apply(lambda g: g.head(int(per.get(g.name, 0))))
            d.to_parquet(ctx.out / f"qa_{c}.parquet", index=False)
            removed.append({"file": f"qa_{c}.parquet", "eval_set": "(after re-equalising)", "removed": len(d)})
        n = int(per.sum())
        if (ctx.out / "qa_generic.parquet").exists():
            pd.read_parquet(ctx.out / "qa_generic.parquet").head(n).to_parquet(ctx.out / "qa_generic.parquet", index=False)
    rm = pd.DataFrame(removed)
    rm.to_csv(ctx.out / "removed.csv", index=False)
    n_rm = int(rm.loc[~rm["eval_set"].str.startswith("("), "removed"].sum()) if len(rm) else 0
    ctx.publish("WS4 decontamination", [
        {"type": "kpis", "items": [
            {"label": "eval sets checked", "value": len(ev), "fmt": "int", "hint": ", ".join(sorted(ev))},
            {"label": f"examples removed ({n}-gram)", "value": n_rm, "fmt": "int"},
            {"label": "files cleaned", "value": int(rm["file"].nunique()) if len(rm) else 0, "fmt": "int"}]},
        {"type": "table", "title": "Removed examples per training file and eval set",
         "rows": rm[~rm["eval_set"].str.startswith("(")].to_dict("records") if len(rm) else [],
         "columns": [{"key": "file", "label": "training file"}, {"key": "eval_set", "label": "matched eval set"},
                     {"key": "removed", "label": "removed", "fmt": "int"}]},
    ], summary=f"{n_rm} examples removed")
    return {"removed": n_rm, "eval_sets": sorted(ev)}
