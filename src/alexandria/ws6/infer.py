"""WS6 model plumbing (Alex-train env, GPU): the frozen LLM, projector load / save, batched generation from
inputs_embeds (left padding; HF generate returns only the new tokens when given embeddings)."""
from __future__ import annotations

import json
from pathlib import Path

from ..llm import model_spec
from ..runtime import Context
from .common import build_batch, local_snapshot, make_projector


def load_llm(ctx: Context, name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = model_spec(ctx.cfg, name)
    path = local_snapshot(ctx, spec)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, spec


def emb_rms(model) -> float:
    """Typical per-dimension RMS of the LLM's token embeddings (the projector's output scale at init)."""
    w = model.get_input_embeddings().weight.float()
    return float(w.pow(2).mean(-1).sqrt().mean())


def save_projector(proj, out: Path, meta: dict) -> None:
    import torch
    out.mkdir(parents=True, exist_ok=True)
    torch.save(proj.state_dict(), out / "projector.pt")
    (out / "projector.json").write_text(json.dumps(meta, indent=1))


def load_projector(path: Path, device):
    import torch
    meta = json.loads((path / "projector.json").read_text())
    proj = make_projector(meta["in_dim"], meta["hidden"], meta["k"], meta["d_model"])
    proj.load_state_dict(torch.load(path / "projector.pt", map_location="cpu"))
    return proj.to(device).eval(), meta


def generate(model, tok, seqs: list[dict], kn: list, max_new: int, batch_size: int) -> list[str]:
    import torch
    embed = model.get_input_embeddings()
    out: list[str] = []
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]["ids"]))      # length-sorted batches
    res: dict[int, str] = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        with torch.no_grad():
            e, m, _ = build_batch([seqs[i] for i in idx], [kn[i] for i in idx], embed, pad_left=True, device=model.device)
            g = model.generate(inputs_embeds=e, attention_mask=m, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for i, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True), strict=True):
            res[i] = t.strip()
    out = [res[i] for i in range(len(seqs))]
    return out
