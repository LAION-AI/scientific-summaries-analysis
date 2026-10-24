"""WS6 shared pieces: settings, the projector, prompt assembly with knowledge tokens, the encoder, helpers.

A prompt with knowledge tokens is the model's own chat template around a placeholder:
    [system] [user: "Context:\n<<<\n" KN "\n>>>\n\nQuestion: ..."] [assistant: answer]
The placeholder is split out of the rendered template; the text pieces are embedded with the LLM's input embeddings and
KN is replaced by n_docs x k projected vectors. Labels are -100 everywhere except the answer tokens (and EOS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..qa import ANSWER_SYSTEM
from ..runtime import Context

PLACEHOLDER = "⁣KN⁣"           # invisible separators: never produced by a tokenizer merge with neighbours
RECON_QUESTION = "Write the summary of this study."


def setting(ctx: Context, key: str, default: Any) -> Any:
    """A debug-profile setting (profiles.debug.ws6_<key>) wins over the default."""
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"ws6_{key}", default)
    return default


def stage_table(ctx: Context, stage: str, name: str, ws: str = "ws6") -> pd.DataFrame:
    p = ctx.cfg.stage_dir(ws, stage) / name
    if not p.exists():
        raise FileNotFoundError(f"{ws}/{stage} has no {name}; run that stage first")
    return pd.read_parquet(p)


def local_snapshot(ctx: Context, spec: dict) -> str:
    """Local directory of a pinned snapshot (offline nodes: some libraries re-read configs by repo id)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(spec["hf"], revision=spec.get("revision"), cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                             local_files_only=True)


def ks(ctx: Context) -> list[int]:
    return [int(k) for k in setting(ctx, "k", ctx.section("ws6.projector").get("k", [1, 4, 16]))]


def llm_name(ctx: Context) -> str:
    return setting(ctx, "llm", ctx.section("ws6.projector").get("llm", "qwen2.5-7b"))


# ---------------------------------------------------------------- projector
def make_projector(in_dim: int, hidden: int, k: int, d_model: int, emb_rms: float = 1.0):
    """2-layer MLP: one embedding -> k vectors in the LLM's input-embedding space (xRAG uses k = 1). The output is
    RMS-normalised and scaled to the typical norm of the LLM's token embeddings, so it starts in-distribution."""
    import torch
    from torch import nn

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.k, self.d = k, d_model
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, k * d_model))
            self.scale = nn.Parameter(torch.tensor(float(emb_rms)))

        def forward(self, x):                                   # x: (n, in_dim) -> (n, k, d_model)
            y = self.net(x).view(x.shape[0], self.k, self.d)
            y = y / y.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            return y * self.scale
    return Projector()


# ---------------------------------------------------------------- prompt assembly
def prompt_pieces(tok, question: str, with_knowledge: bool, context_text: str | None = None,
                  chat_kwargs: dict | None = None) -> tuple[list[int], list[int]]:
    """Token ids before / after the knowledge placeholder (generation prompt included). With no knowledge the whole
    prompt is in `before`; with `context_text` the text is inlined instead of knowledge tokens (KU-text arm)."""
    if with_knowledge:
        user = f"Context:\n<<<\n{PLACEHOLDER}\n>>>\n\nQuestion: {question}"
    elif context_text:
        user = f"Context:\n<<<\n{context_text}\n>>>\n\nQuestion: {question}"
    else:
        user = f"Question: {question}"
    text = tok.apply_chat_template([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}],
                                   add_generation_prompt=True, tokenize=False, **(chat_kwargs or {}))
    if not with_knowledge:
        return tok(text, add_special_tokens=False)["input_ids"], []
    left, right = text.split(PLACEHOLDER)
    return tok(left, add_special_tokens=False)["input_ids"], tok(right, add_special_tokens=False)["input_ids"]


def assemble(left: list[int], n_kn: int, right: list[int], answer: list[int] | None = None) -> dict:
    """Layout of one sequence: ids with -1 at the n_kn knowledge positions, the knowledge mask, and labels (-100
    except the answer). The caller swaps the -1 slots for projected vectors."""
    ans = list(answer or [])
    ids = list(left) + [-1] * n_kn + list(right) + ans
    labels = [-100] * (len(left) + n_kn + len(right)) + ans
    kn_mask = [False] * len(left) + [True] * n_kn + [False] * (len(right) + len(ans))
    return {"ids": ids, "labels": labels, "kn_mask": kn_mask}


def build_batch(seqs: list[dict], kn: list, embed_fn, pad_left: bool, device=None):
    """Pad a list of assembled sequences into (inputs_embeds, attention_mask, labels). kn[i] is a (n_kn_i, d) tensor
    of projected vectors for sequence i (or None); embed_fn maps a LongTensor of ids to embeddings."""
    import torch
    L = max(len(s["ids"]) for s in seqs)
    embs, masks, labels = [], [], []
    for s, k in zip(seqs, kn, strict=True):
        ids = torch.tensor([max(i, 0) for i in s["ids"]], device=device)
        e = embed_fn(ids)
        if k is not None:
            pos = torch.tensor(s["kn_mask"], device=device)
            e = e.clone()
            e[pos] = k.to(e.dtype)
        pad = L - len(s["ids"])
        z = torch.zeros(pad, e.shape[-1], dtype=e.dtype, device=e.device)
        m = torch.ones(len(s["ids"]), dtype=torch.long, device=e.device)
        zm = torch.zeros(pad, dtype=torch.long, device=e.device)
        lab = torch.tensor(s["labels"], device=e.device)
        zl = torch.full((pad,), -100, dtype=torch.long, device=e.device)
        if pad_left:
            embs.append(torch.cat([z, e])); masks.append(torch.cat([zm, m])); labels.append(torch.cat([zl, lab]))
        else:
            embs.append(torch.cat([e, z])); masks.append(torch.cat([m, zm])); labels.append(torch.cat([lab, zl]))
    return torch.stack(embs), torch.stack(masks), torch.stack(labels)


# ---------------------------------------------------------------- encoder
def load_encoder(ctx: Context, device: str):
    """gte-modernbert from the local snapshot, CLS pooling, fp32 (full-precision vectors, unlike the PQ48 index)."""
    from ..llm import embedder_spec
    from ..skill.retrieval import Encoder
    spec = embedder_spec(ctx.cfg, ctx.section("ws6.embed").get("embedder", "gte-modernbert"))
    spec = {**spec, "hf": local_snapshot(ctx, spec), "revision": None,
            "batch_size": int(ctx.section("ws6.embed").get("batch_size", spec.get("batch_size", 16)))}
    enc = Encoder(spec, "cpu")                                 # loads fp32 weights (Encoder uses fp16 only on cuda)
    enc.model.to(device)                                       # fp32 on GPU too: these vectors are the input signal
    enc.device = device
    return enc


def embeddings_path(ctx: Context) -> Path:
    return ctx.cfg.stage_dir("ws6", "embed")
