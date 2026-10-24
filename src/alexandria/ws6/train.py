"""Stage `train-projector` (WS6, GPU, Alex-train env): one projector per k (knowledge tokens per paper).

The LLM stays frozen; only the projector (2-layer MLP, 768 -> hidden -> k x d_model) is trained, xRAG-style:
  stage 1  reconstruction: from the paper's KU embedding, generate the start of its executive summary
           (`recon_tokens` tokens); teaches the projector to place the KU's content in the LLM's input space;
  stage 2  QA: answer the paper's QA pairs (ws6 qa) from the embedding; loss on the answer tokens only.
AdamW, linear warmup + cosine decay, bf16 LLM / fp32 projector. Dev loss (dev papers, never trained on) every
`eval_every` steps; at the end the dev QA answers are generated and scored (normalised exact / contains match).
Split over SLURM array tasks by k. Output per k: k<k>/projector.pt, projector.json, curve.json.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pandas as pd

from ..qa import exact_match, normalize_answer
from ..runtime import Context, atomic_write_json, log, stage
from .common import RECON_QUESTION, assemble, build_batch, ks, llm_name, make_projector, prompt_pieces, setting, stage_table
from .infer import emb_rms, generate, load_llm, save_projector


def match(pred: str, gold: str) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return bool(p and g) and (exact_match(pred, gold) or (g in p and len(p.split()) <= len(g.split()) + 8))


def examples(ctx: Context, tok, split: str, recon_tokens: int) -> tuple[list[dict], list[dict]]:
    """(reconstruction examples, QA examples) of one split: {key, left, right, answer ids, gold}."""
    papers = stage_table(ctx, "papers", "papers.parquet")
    papers = papers[papers["split"] == split]
    qa = stage_table(ctx, "qa", "qa.parquet")
    qa = qa[qa["split"] == split]
    l_r, r_r = prompt_pieces(tok, RECON_QUESTION, True)
    recon = []
    for p in papers.itertuples():
        target = json.loads(p.ku).get("executive_summary") or ""
        ids = tok(target, add_special_tokens=False)["input_ids"][:recon_tokens]
        if ids:
            recon.append({"key": p.paper_key, "left": l_r, "right": r_r, "answer": ids + [tok.eos_token_id], "gold": target})
    qas = []
    for q in qa.itertuples():
        left, right = prompt_pieces(tok, q.question, True)
        qas.append({"key": q.paper_key, "left": left, "right": right, "gold": q.answer,
                    "answer": tok(q.answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]})
    return recon, qas


def lr_at(step: int, total: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def batch_loss(model, proj, vecs, batch: list[dict], k: int):
    import torch
    seqs = [assemble(b["left"], k, b["right"], b["answer"]) for b in batch]
    v = torch.tensor(np.stack([vecs[b["key"]] for b in batch]), device=model.device)
    kn = proj(v).to(torch.bfloat16)
    e, m, lab = build_batch(seqs, list(kn), model.get_input_embeddings(), pad_left=False, device=model.device)
    return model(inputs_embeds=e, attention_mask=m, labels=lab).loss


@stage("ws6", "train-projector", kind="map", needs=("embed", "qa"), resources="gpu", env="train",
       config_keys=("ws6.projector",),
       help="xRAG-style projector per k (LLM frozen): reconstruction, then QA from the KU embedding; dev loss / accuracy")
def run(ctx: Context) -> dict:
    import torch
    cfg = ctx.section("ws6.projector")
    model, tok, spec = load_llm(ctx, llm_name(ctx))
    keys = pd.read_parquet(ctx.cfg.stage_dir("ws6", "embed") / "projector_keys.parquet")["key"]
    mat = np.load(ctx.cfg.stage_dir("ws6", "embed") / "projector.npy")
    vecs = dict(zip(keys, mat, strict=True))
    rt = int(cfg.get("recon_tokens", 128))
    rec_tr, qa_tr = examples(ctx, tok, "train", rt)
    rec_dv, qa_dv = examples(ctx, tok, "dev", rt)
    bs, lr, warm = int(cfg.get("batch_size", 16)), float(cfg.get("lr", 1e-4)), int(cfg.get("warmup_steps", 20))
    plan = [("recon", rec_tr, rec_dv, int(setting(ctx, "recon_steps", cfg.get("recon_steps", 1000)))),
            ("qa", qa_tr, qa_dv, int(setting(ctx, "qa_steps", cfg.get("qa_steps", 2000))))]
    done = []
    for k in ks(ctx)[ctx.shard_index::ctx.shard_count]:
        out = ctx.out / f"k{k}"
        if (out / "projector.pt").exists() and not ctx.force:
            continue
        torch.manual_seed(int(cfg.get("seed", 0)) + k)
        rng = random.Random(int(cfg.get("seed", 0)) + k)
        d = model.get_input_embeddings().weight.shape[1]
        proj = make_projector(mat.shape[1], int(cfg.get("hidden", 2048)), k, d, emb_rms(model)).to(model.device)
        opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.01)
        curve, t0 = [], time.time()
        for phase, train, dev, steps in plan:
            log(f"train-projector: k={k} {phase}: {len(train)} train / {len(dev)} dev examples, {steps} steps")
            for step in range(steps):
                for gr in opt.param_groups:
                    gr["lr"] = lr_at(step, steps, warm, lr)
                proj.train()
                loss = batch_loss(model, proj, vecs, rng.sample(train, min(bs, len(train))), k)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                opt.step()
                row = {"phase": phase, "step": step, "loss": float(loss)}
                if dev and (step % int(cfg.get("eval_every", 50)) == 0 or step == steps - 1):
                    proj.eval()
                    with torch.no_grad():
                        dl = [float(batch_loss(model, proj, vecs, dev[i:i + bs], k)) for i in range(0, min(len(dev), 4 * bs), bs)]
                    row["dev_loss"] = float(np.mean(dl))
                curve.append(row)
        proj.eval()
        seqs = [assemble(q["left"], k, q["right"]) for q in qa_dv]
        with torch.no_grad():
            kn = list(proj(torch.tensor(np.stack([vecs[q["key"]] for q in qa_dv]), device=model.device)).to(torch.bfloat16)) if qa_dv else []
        preds = generate(model, tok, seqs, kn, 48, 32) if qa_dv else []
        dev_acc = float(np.mean([match(p, q["gold"]) for p, q in zip(preds, qa_dv, strict=True)])) if qa_dv else None
        save_projector(proj, out, {"k": k, "in_dim": int(mat.shape[1]), "hidden": int(cfg.get("hidden", 2048)),
                                   "d_model": int(d), "llm": spec["name"], "llm_hf": spec["hf"], "revision": spec.get("revision")})
        atomic_write_json(out / "curve.json", {"k": k, "curve": curve, "dev_qa_match": dev_acc, "dev_pairs": len(qa_dv),
                                               "seconds": round(time.time() - t0, 1),
                                               "dev_examples": [{"q": q["gold"], "pred": p} for q, p in list(zip(qa_dv, preds))[:20]]})
        done.append(k)
    rows = []
    for p in sorted(ctx.out.glob("k*/curve.json")):
        j = json.loads(p.read_text())
        rec = [c for c in j["curve"] if c["phase"] == "recon"]
        qa = [c for c in j["curve"] if c["phase"] == "qa"]
        last_dev = [c.get("dev_loss") for c in qa if c.get("dev_loss") is not None]
        rows.append({"k": j["k"], "recon_loss_first": rec[0]["loss"] if rec else None, "recon_loss_last": rec[-1]["loss"] if rec else None,
                     "qa_loss_first": qa[0]["loss"] if qa else None, "qa_loss_last": qa[-1]["loss"] if qa else None,
                     "dev_loss": last_dev[-1] if last_dev else None, "dev_qa_match": j["dev_qa_match"],
                     "minutes": j["seconds"] / 60})
    if rows:
        ctx.publish("WS6 projector training", [
            {"type": "kpis", "items": [{"label": f"k={r['k']} dev QA match", "value": r["dev_qa_match"], "fmt": "pct1",
                                        "hint": "dev papers (never trained on), normalised exact / contains"} for r in rows]},
            {"type": "table", "title": "Projector runs (LLM frozen)", "rows": rows, "columns": [
                {"key": "k", "label": "k tokens", "fmt": "int"},
                {"key": "recon_loss_first", "label": "recon loss first", "fmt": "float3"},
                {"key": "recon_loss_last", "label": "recon loss last", "fmt": "float3"},
                {"key": "qa_loss_first", "label": "QA loss first", "fmt": "float3"},
                {"key": "qa_loss_last", "label": "QA loss last", "fmt": "float3"},
                {"key": "dev_loss", "label": "dev QA loss", "fmt": "float3"},
                {"key": "dev_qa_match", "label": "dev QA match", "fmt": "pct1"},
                {"key": "minutes", "label": "minutes", "fmt": "float"}]},
        ], summary=", ".join(f"k={r['k']}: dev {r['dev_qa_match']:.1%}" for r in rows if r["dev_qa_match"] is not None))
    return {"trained_k": done}
