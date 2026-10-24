"""Stages `eval` and `grade` (WS6).

eval   The frozen LLM answers, per arm (split over SLURM array tasks):
         ws2 / closed_book   the WS2 QA-bench question alone
         ws2 / ku_text       the gold paper's KU as text (the text point on the same axis; capped tokens)
         ws2 / emb_k<k>      the gold paper's KU embedding as k knowledge tokens (the extreme-compression point)
         ws3 / emb_k<k>      WS3 arm EMB on qabench-open: top-`retrieve_k` papers by gte-modernbert cosine over the
                             WS3 corpus, retrieve_k x k knowledge tokens; logs whether the gold paper was retrieved
       Same answer prompt as WS2 (qa.ANSWER_SYSTEM). Output: parts/<set>__<arm>.parquet.
grade  SimpleQA judge (Qwen3-8B, as WS2) over every answer + model-free exact match / token F1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..llm import model_spec, with_model
from ..qa import exact_match, grade_batch, is_abstention, token_f1
from ..runtime import Context, log, stage
from .common import assemble, ks, llm_name, prompt_pieces, setting


def arms(ctx: Context) -> list[tuple[str, str]]:
    return [("ws2", "closed_book"), ("ws2", "ku_text")] + [("ws2", f"emb_k{k}") for k in ks(ctx)] + \
        [("ws3", f"emb_k{k}") for k in ks(ctx)]


def vectors(ctx: Context, name: str) -> tuple[list[str], np.ndarray]:
    d = ctx.cfg.stage_dir("ws6", "embed")
    return pd.read_parquet(d / f"{name}_keys.parquet")["key"].tolist(), np.load(d / f"{name}.npy")


def retrieve(q: np.ndarray, corpus: np.ndarray, top: int) -> np.ndarray:
    """Indices of the top cosine matches per query (vectors are L2-normalised)."""
    s = q @ corpus.T
    return np.argsort(-s, axis=1)[:, :top]


@stage("ws6", "eval", kind="map", needs=("train-projector", "embed"), resources="gpu", env="train",
       config_keys=("ws6.eval",),
       help="answer WS2 questions closed book / with KU text / with k knowledge tokens; WS3 arm EMB (retrieved embeddings)")
def run(ctx: Context) -> dict:
    import torch

    from .embed import texts_ws2
    from .infer import generate, load_llm, load_projector
    cfg = ctx.section("ws6.eval")
    model, tok, spec = load_llm(ctx, llm_name(ctx))
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    qs = pd.read_parquet(ctx.cfg.stage_dir("ws2", "questions") / "questions.parquet", columns=["qid", "paper_key", "question", "answer"])
    w2_keys, w2_vec = vectors(ctx, "ws2")
    w2 = dict(zip(w2_keys, w2_vec, strict=True))
    t2 = texts_ws2(ctx)
    ku_txt = dict(zip(t2["key"], t2["text"], strict=True))
    cap, mx, bs = int(cfg.get("ku_text_max_tokens", 6000)), int(cfg.get("max_new_tokens", 48)), int(cfg.get("batch_size", 32))
    done = []
    for set_, arm in arms(ctx)[ctx.shard_index::ctx.shard_count]:
        dst = parts / f"{set_}__{arm}.parquet"
        if dst.exists() and not ctx.force:
            continue
        rows, seqs, kn = [], [], []
        if set_ == "ws2":
            q = qs[qs["paper_key"].isin(w2)]
            proj = None
            if arm.startswith("emb_k"):
                proj, meta = load_projector(ctx.cfg.stage_dir("ws6", "train-projector") / arm.replace("emb_", ""), model.device)
                k = meta["k"]
            for r in q.itertuples():
                if arm == "closed_book":
                    left, _ = prompt_pieces(tok, r.question, False)
                    seqs.append(assemble(left, 0, []))
                    kn.append(None)
                    n_ctx = 0
                elif arm == "ku_text":
                    ids = tok(ku_txt[r.paper_key], add_special_tokens=False)["input_ids"][:cap]
                    left, _ = prompt_pieces(tok, r.question, False, context_text=tok.decode(ids))
                    seqs.append(assemble(left, 0, []))
                    kn.append(None)
                    n_ctx = len(ids)
                else:
                    left, right = prompt_pieces(tok, r.question, True)
                    seqs.append(assemble(left, k, right))
                    with torch.no_grad():
                        kn.append(proj(torch.tensor(w2[r.paper_key][None], device=model.device))[0].to(torch.bfloat16))
                    n_ctx = k
                rows.append({"id": r.qid, "cluster": r.paper_key, "question": r.question, "gold": r.answer,
                             "ctx_tokens": n_ctx, "gold_retrieved": None})
        else:
            it = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
            it = it[it["bench"] == "qabench-open"]
            n = setting(ctx, "ws3_items", cfg.get("ws3_items"))
            it = it.head(int(n)) if n else it
            qk, qv = vectors(ctx, "ws3query")
            qmap = dict(zip(qk, qv, strict=True))
            ck, cv = vectors(ctx, "ws3corpus")
            top = int(cfg.get("retrieve_k", 3))
            proj, meta = load_projector(ctx.cfg.stage_dir("ws6", "train-projector") / arm.replace("emb_", ""), model.device)
            k = meta["k"]
            it = it[it["item_id"].isin(qmap)]
            hits = retrieve(np.stack([qmap[i] for i in it["item_id"]]), cv, top)
            for r, h in zip(it.itertuples(), hits, strict=True):
                keys = [ck[j] for j in h]
                left, right = prompt_pieces(tok, r.question, True)
                seqs.append(assemble(left, top * k, right))
                with torch.no_grad():
                    kn.append(proj(torch.tensor(cv[h], device=model.device)).reshape(top * k, -1).to(torch.bfloat16))
                rows.append({"id": r.item_id, "cluster": r.cluster, "question": r.question, "gold": r.answer,
                             "ctx_tokens": top * k, "gold_retrieved": r.gold_paper_key in keys if r.gold_paper_key else None,
                             "retrieved": ",".join(keys)})
        log(f"eval: {set_}/{arm}: {len(seqs)} prompts")
        preds = generate(model, tok, seqs, kn, mx, bs)
        pd.DataFrame(rows).assign(set=set_, arm=arm, prediction=preds, llm=spec["name"]).to_parquet(dst, index=False)
        done.append(f"{set_}/{arm}")
    n_parts = len(list(parts.glob("*.parquet")))
    ctx.publish("WS6 answers", [{"type": "kpis", "items": [
        {"label": "arms answered", "value": n_parts, "fmt": "int"},
        {"label": "arms planned", "value": len(arms(ctx)), "fmt": "int"},
        {"label": "frozen LLM", "value": spec["name"]}]}], summary=f"{n_parts} of {len(arms(ctx))} arms answered")
    return {"done": done}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("ws6", "grade", needs=("eval",), resources="gpu", env="llm", config_keys=("ws6.grade",),
       help="SimpleQA judge over every WS6 answer (+ exact match, token F1)")
def run_grade(ctx: Context) -> dict:
    cfg = ctx.section("ws6.grade")
    files = sorted((ctx.cfg.stage_dir("ws6", "eval") / "parts").glob("*.parquet"))
    if not files:
        return ctx.skip("WS6 grading", "no answers yet (run eval)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    judge = model_spec(ctx.cfg, cfg.get("judge", "qwen3-8b"))
    items = [{"question": r.question, "gold": r.gold, "predicted": r.prediction} for r in df.itertuples()]
    log(f"grade: {len(items)} answers with {judge['name']}")
    df["grade"] = with_model(judge, _grade, items, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["attempted"] = (df["grade"] != "NOT_ATTEMPTED").astype(int)
    df["abstained"] = [int(is_abstention(p)) for p in df["prediction"]]
    df["em"] = [int(exact_match(p, g)) for p, g in zip(df["prediction"], df["gold"], strict=True)]
    df["f1"] = [token_f1(p, g) for p, g in zip(df["prediction"], df["gold"], strict=True)]
    df.to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("WS6 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"},
        {"label": "judge", "value": judge["name"]},
        {"label": "CORRECT overall", "value": float(df["correct"].mean()), "fmt": "pct1"}]}], summary=f"{len(df)} answers graded")
    return {"graded": len(df)}
