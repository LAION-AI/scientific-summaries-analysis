"""Stages `train-sft`, `train-grpo` and `train-tools` (WS4, GPU, Alex-train env: TRL + PEFT).

One recipe for every condition (LoRA on all linear layers, same lr / schedule / sequence length / batch tokens /
steps), so the only thing that differs between runs of one model and seed is the training data:
  sft-doc   continued training on the condition's token-matched documents (+ replay), causal-LM loss, packed
  sft-qa    instruction pairs generated from the condition (+ Tulu-3 replay), loss on the answer only, packed
  grpo      GRPO (Shao et al., 2024; Dr.GRPO loss, no KL) on the condition's verifiable QA pairs as prompts,
            reward 1 if the answer matches the gold answer (normalised exact or numeric match), else 0
  tools     GRPO with the Alexandria skill as tools (Search-R1 style, Jin et al., 2025): the model may call
            `search` / `get_paper` before answering; reward as grpo; trained on generated KU questions + public
            training splits (SciQ train, PubMedQA artificial), never on HLE / GPQA
Runs are split over SLURM array tasks (`--array N`: task i trains runs[i::N]); a finished run (adapter present) is
skipped unless --force. Output per run: runs/<run>/adapter (LoRA), runs/<run>/train.json (loss curve, steps, tokens,
GPU time).
"""
from __future__ import annotations

import json
import math
import time

import pandas as pd

from ..llm import model_spec
from ..qa import exact_match, normalize_answer
from ..runtime import Context, atomic_write_json, log, stage
from .common import run_grid, setting

ANSWER_SYSTEM = "Answer the question about scientific research with a short answer only."


def _decontam(ctx: Context, name: str) -> pd.DataFrame:
    p = ctx.cfg.stage_dir("ws4", "decontam") / name
    if not p.exists():
        raise FileNotFoundError(f"ws4/decontam has no {name}; run data, qa-data, probe, decontam first")
    return pd.read_parquet(p)


def sft_dataset(ctx: Context, run: dict):
    """The training rows of one SFT run (datasets.Dataset)."""
    from datasets import Dataset
    if run["method"] == "sft-doc":
        df = _decontam(ctx, f"docs_{run['condition']}.parquet")
        return Dataset.from_dict({"text": df["text"].tolist()})
    rep = _decontam(ctx, "chat_replay.parquet")
    rows = []
    if run["condition"] == "none":
        data = []
        n_rep = len(rep)
    else:
        data = _decontam(ctx, f"qa_{run['condition']}.parquet")
        share = float(ctx.section("ws4.data").get("replay_share", 0.25))
        n_rep = int(len(data) * share / (1 - share))
        rows += [{"prompt": [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": q}],
                  "completion": [{"role": "assistant", "content": a}]}
                 for q, a in zip(data["question"], data["answer"], strict=True)]
    rows += [{"prompt": json.loads(p), "completion": [{"role": "assistant", "content": c}]}
             for p, c in zip(rep["prompt"][:n_rep], rep["completion"][:n_rep], strict=True)]
    return Dataset.from_list(rows).shuffle(seed=run["seed"])


CONTEXT_SYSTEM = ("Answer the question about scientific research from the given text, with a short answer only "
                  "(a number with its unit, a name, yes/no, or a short phrase).")


def grpo_contexts(ctx: Context, df: pd.DataFrame, cond: str, max_chars: int) -> list[str]:
    """The representation each pair was generated from (full text: a window around the evidence quote), so the
    reward is learnable: GRPO on closed-book questions about unseen papers gives ~0 reward and no signal."""
    import json as _json

    from rapidfuzz import fuzz

    from .qa_data import representation
    papers = pd.read_parquet(ctx.cfg.stage_dir("ws4", "papers") / "papers.parquet")
    order = {k: i for i, k in enumerate(papers.loc[papers["split"] == "train", "paper_key"])}
    by = {p.paper_key: {"ku": _json.loads(p.ku), "full": p.full} for p in papers.itertuples()}
    out = []
    for pk, ev in zip(df["paper_key"], df.get("evidence", pd.Series([""] * len(df))), strict=True):
        text = representation(cond, by[pk], order.get(pk, 0))
        if len(text) > max_chars:
            al = fuzz.partial_ratio_alignment(str(ev or "")[:300], text) if ev else None
            mid = (al.dest_start + al.dest_end) // 2 if al and al.score >= 70 else 0
            lo = max(0, min(mid - max_chars // 2, len(text) - max_chars))
            text = text[lo: lo + max_chars]
        out.append(text)
    return out


def grpo_dataset(ctx: Context, run: dict, sources: list[str] | None = None):
    from datasets import Dataset
    if sources is None:                         # RL variant: verifiable QA from the condition, with its source text
        df = _decontam(ctx, f"qa_{run['condition']}.parquet")
        n = ctx.section("ws4.grpo").get("prompts_per_condition")
        df = df.sample(frac=1.0, random_state=run["seed"]).head(int(n) if n else len(df))
        texts = grpo_contexts(ctx, df, run["condition"], int(ctx.section("ws4.grpo").get("context_chars", 6000)))
        return Dataset.from_list([{"prompt": [{"role": "system", "content": CONTEXT_SYSTEM},
                                              {"role": "user", "content": f"TEXT:\n{t}\n\nQuestion: {q}"}], "answer": a}
                                  for t, q, a in zip(texts, df["question"], df["answer"], strict=True)])
    else:
        from .common import load_resource
        frames = []
        for s in sources:
            if s == "ws4_qa_ku":
                frames.append(_decontam(ctx, "qa_ku.parquet")[["question", "answer"]])
            else:
                r = ctx.cfg.raw["ws4_resources"][s]
                d = load_resource(ctx, s, max_rows=20000).rename(columns={r["question"]: "question", r["answer"]: "answer"})
                frames.append(d[["question", "answer"]].astype(str))
        # equal share per source (capped at the smallest): otherwise closed-book-answerable public QA dominates and
        # the policy learns that tools are unnecessary (observed in the debug run: tool calls 0.75 -> 0)
        k = min(len(f) for f in frames)
        df = pd.concat([f.sample(frac=1.0, random_state=run["seed"]).head(k) for f in frames], ignore_index=True)
    df = df.sample(frac=1.0, random_state=run["seed"])
    return Dataset.from_list([{"prompt": [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": q}],
                               "answer": a} for q, a in zip(df["question"], df["answer"], strict=True)])


def _text(c) -> str:
    if isinstance(c, list):                                   # conversational completion: last assistant message
        return str(c[-1].get("content", "")) if c else ""
    return str(c)


def answer_match(pred: str, gold: str) -> bool:
    """Verifiable-QA match: normalised exact / numeric match, or a short answer that contains the gold answer."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    return exact_match(pred, gold) or (g in p and len(p.split()) <= len(g.split()) + 8)


def answer_reward(completions, answer, **kw) -> list[float]:
    """1.0 if the completion (its last line, or the whole of it) matches the gold answer, else 0.0."""
    out = []
    for c, a in zip(completions, answer, strict=True):
        t = _text(c).strip()
        out.append(1.0 if answer_match(t.split("\n")[-1], a) or answer_match(t, a) else 0.0)
    return out


def _model_and_tok(ctx: Context, name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = model_spec(ctx.cfg, name)
    tok = AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["hf"], revision=spec.get("revision"), dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    return model, tok, spec


def _lora(cfg: dict):
    from peft import LoraConfig
    lc = cfg.get("lora", {})
    return LoraConfig(r=int(lc.get("r", 64)), lora_alpha=int(lc.get("alpha", 128)), lora_dropout=float(lc.get("dropout", 0.05)),
                      target_modules=lc.get("target_modules", "all-linear"), task_type="CAUSAL_LM")


def _steps(ctx: Context, cfg: dict, n_tokens: int) -> int:
    ms = setting(ctx, "max_steps", cfg.get("max_steps"))
    if ms:
        return int(ms)
    return max(1, math.ceil(n_tokens * float(cfg.get("epochs", 1)) / int(cfg.get("batch_tokens", 32768))))


def _finish(ctx: Context, run: dict, trainer, out_dir, t0: float, extra: dict) -> None:
    trainer.model.save_pretrained(out_dir / "adapter")
    hist = [h for h in trainer.state.log_history if "loss" in h or "reward" in h]
    atomic_write_json(out_dir / "train.json", {**run, **extra, "steps": trainer.state.global_step,
                                               "seconds": round(time.time() - t0, 1), "log": hist})


def _runs(ctx: Context, methods: tuple[str, ...]) -> list[dict]:
    runs = [r for r in run_grid(ctx) if r["method"] in methods]
    return runs[ctx.shard_index::ctx.shard_count]


def _publish(ctx: Context, title: str, methods: tuple[str, ...]) -> None:
    rows = []
    for r in [r for r in run_grid(ctx) if r["method"] in methods]:
        p = ctx.out / "runs" / r["run"] / "train.json"
        if p.exists():
            j = json.loads(p.read_text())
            losses = [h.get("loss") for h in j["log"] if h.get("loss") is not None]
            rewards = [h.get("reward") for h in j["log"] if h.get("reward") is not None]
            rows.append({"run": r["run"], "model": r["model"], "method": r["method"], "condition": r["condition"],
                         "seed": r["seed"], "steps": j["steps"], "examples": j.get("examples"),
                         "first_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
                         "final_reward": rewards[-1] if rewards else None, "minutes": j["seconds"] / 60})
    df = pd.DataFrame(rows)
    panels = [{"type": "kpis", "items": [
        {"label": "runs finished", "value": len(df), "fmt": "int"},
        {"label": "runs planned", "value": len([r for r in run_grid(ctx) if r["method"] in methods]), "fmt": "int"},
        {"label": "GPU hours", "value": float(df["minutes"].sum() / 60) if len(df) else 0.0, "fmt": "float"}]}]
    if len(df):
        panels.append({"type": "table", "title": "Training runs", "rows": df.to_dict("records"), "columns": [
            {"key": "run", "label": "run"}, {"key": "condition", "label": "condition"}, {"key": "seed", "label": "seed", "fmt": "int"},
            {"key": "steps", "label": "steps", "fmt": "int"}, {"key": "examples", "label": "examples", "fmt": "int"},
            {"key": "first_loss", "label": "first loss", "fmt": "float3"}, {"key": "final_loss", "label": "final loss", "fmt": "float3"},
            {"key": "final_reward", "label": "final reward", "fmt": "float3"}, {"key": "minutes", "label": "minutes", "fmt": "float"}]})
    ctx.publish(title, panels, summary=f"{len(df)} runs finished")


@stage("ws4", "train-sft", kind="map", needs=("decontam",), resources="gpu", env="train", config_keys=("ws4.train",),
       help="LoRA SFT per model x format (doc / qa) x condition x seed, same recipe and budget")
def run_sft(ctx: Context) -> dict:
    from trl import SFTConfig, SFTTrainer
    cfg = ctx.section("ws4.train")
    done = []
    for run in _runs(ctx, ("sft-doc", "sft-qa")):
        out_dir = ctx.out / "runs" / run["run"]
        if (out_dir / "adapter").exists() and not ctx.force:
            continue
        ds = sft_dataset(ctx, run)
        model, tok, spec = _model_and_tok(ctx, run["model"])
        seq = int(ctx.section("ws4.data").get("doc_max_tokens", 2048))
        bs = int(cfg.get("per_device_batch", 4))
        accum = max(1, int(cfg.get("batch_tokens", 32768)) // (seq * bs))
        n_tok = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in ds["text"]) if "text" in ds.column_names \
            else len(ds) * 256
        steps = _steps(ctx, cfg, n_tok)
        log(f"train-sft: {run['run']}: {len(ds)} examples, {steps} steps")
        args = SFTConfig(output_dir=str(out_dir / "ckpt"), max_length=seq, packing=True, per_device_train_batch_size=bs,
                         gradient_accumulation_steps=accum, learning_rate=float(cfg.get("lr", 1e-4)),
                         lr_scheduler_type=cfg.get("scheduler", "cosine"), warmup_steps=float(cfg.get("warmup_ratio", 0.03)),
                         max_steps=steps, bf16=bool(cfg.get("bf16", True)), logging_steps=1, save_strategy="no",
                         report_to="none", seed=int(run["seed"]), data_seed=int(run["seed"]),
                         gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
                         **({"dataset_text_field": "text"} if run["method"] == "sft-doc" else {"completion_only_loss": True}))
        t0 = time.time()
        tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=_lora(cfg))
        tr.train()
        _finish(ctx, run, tr, out_dir, t0, {"examples": len(ds), "tokens": n_tok, "base": spec["hf"],
                                            "revision": spec.get("revision")})
        done.append(run["run"])
        del tr, model
    _publish(ctx, "WS4 SFT runs", ("sft-doc", "sft-qa"))
    return {"trained": done}


def _grpo_args(ctx: Context, cfg: dict, run: dict, out_dir):
    from trl import GRPOConfig
    extra = {"max_tool_calling_iterations": int(cfg["max_tool_calls"])} if "max_tool_calls" in cfg else {}
    return GRPOConfig(**extra, output_dir=str(out_dir / "ckpt"), learning_rate=float(cfg.get("lr", 5e-6)),
                      num_generations=int(cfg.get("num_generations", 8)),
                      per_device_train_batch_size=int(cfg.get("num_generations", 8)),
                      max_completion_length=int(cfg.get("max_completion_length", 128)), beta=float(cfg.get("beta", 0.0)),
                      loss_type=cfg.get("loss_type", "dr_grpo"), max_steps=int(setting(ctx, "max_steps", cfg.get("max_steps", 300))),
                      bf16=True, logging_steps=1, save_strategy="no", report_to="none", seed=int(run["seed"]),
                      use_vllm=bool(cfg.get("use_vllm", False)), gradient_checkpointing=True)


@stage("ws4", "train-grpo", kind="map", needs=("decontam",), resources="gpu", env="train", config_keys=("ws4.grpo",),
       help="GRPO on verifiable QA generated from each representation (same prompt count), answer-match reward")
def run_grpo(ctx: Context) -> dict:
    from trl import GRPOTrainer
    cfg, tcfg = ctx.section("ws4.grpo"), ctx.section("ws4.train")
    done = []
    for run in _runs(ctx, ("grpo",)):
        out_dir = ctx.out / "runs" / run["run"]
        if (out_dir / "adapter").exists() and not ctx.force:
            continue
        ds = grpo_dataset(ctx, run)
        model, tok, spec = _model_and_tok(ctx, run["model"])
        log(f"train-grpo: {run['run']}: {len(ds)} prompts")
        t0 = time.time()
        tr = GRPOTrainer(model=model, reward_funcs=[answer_reward], args=_grpo_args(ctx, cfg, run, out_dir),
                         train_dataset=ds, processing_class=tok, peft_config=_lora(tcfg))
        tr.train()
        _finish(ctx, run, tr, out_dir, t0, {"examples": len(ds), "base": spec["hf"], "revision": spec.get("revision")})
        done.append(run["run"])
        del tr, model
    _publish(ctx, "WS4 GRPO runs", ("grpo",))
    return {"trained": done}


def make_tools(ctx: Context):
    """The Alexandria skill as plain Python tools (TRL calls them during generation)."""
    from ..llm import embedder_spec
    from ..skill.backend import Backend
    dense = bool(ctx.section("ws4.tools").get("dense", False))      # BM25 only by default: fast rollouts
    be = Backend(ctx.cfg.stage_dir("ws3", "index"), embedder_spec(ctx.cfg, "bge-m3") if dense else None,
                 ctx.out / "tool_cache.sqlite", dense=dense, device="cpu")

    def search(query: str, k: int = 5) -> str:
        """Search scientific papers' knowledge units (structured summaries). Returns matching passages with paper ids.

        Args:
            query: what to look for, in natural language.
            k: number of results (1-10).
        """
        return be.call("search", {"query": query, "k": int(k)}, view="summary")[0]

    def get_paper(id: str) -> str:
        """Read the knowledge unit (structured summary) of one paper by its id.

        Args:
            id: the paper id returned by `search`.
        """
        return be.call("get_paper", {"id": id}, view="summary")[0]
    return [search, get_paper]


@stage("ws4", "train-tools", kind="map", needs=("decontam",), resources="gpu", env="train", config_keys=("ws4.tools",),
       help="GRPO for tool use (Search-R1 style): learn to use the Alexandria skill; never trains on HLE / GPQA")
def run_tools(ctx: Context) -> dict:
    from trl import GRPOTrainer
    cfg, tcfg = ctx.section("ws4.tools"), ctx.section("ws4.train")
    if not cfg.get("enabled"):
        return ctx.skip("WS4 tool-use RL", "ws4.tools.enabled is false")
    if not (ctx.cfg.stage_dir("ws3", "index") / "index.json").exists():
        return ctx.skip("WS4 tool-use RL", "needs the WS3 index (ws3 index stage)")
    bad = {"hle", "hle-gold", "gpqa", "gpqa_diamond"} & set(cfg.get("train_sources", []))
    if bad:
        raise ValueError(f"tool RL must never train on {sorted(bad)}")
    done = []
    for run in _runs(ctx, ("tools",)):
        out_dir = ctx.out / "runs" / run["run"]
        if (out_dir / "adapter").exists() and not ctx.force:
            continue
        ds = grpo_dataset(ctx, run, sources=list(cfg.get("train_sources", ["ws4_qa_ku"])))
        model, tok, spec = _model_and_tok(ctx, run["model"])
        log(f"train-tools: {run['run']}: {len(ds)} prompts, tools search / get_paper")
        t0 = time.time()
        tr = GRPOTrainer(model=model, reward_funcs=[answer_reward], args=_grpo_args(ctx, cfg, run, out_dir),
                         train_dataset=ds, processing_class=tok, peft_config=_lora(tcfg), tools=make_tools(ctx))
        tr.train()
        _finish(ctx, run, tr, out_dir, t0, {"examples": len(ds), "base": spec["hf"], "revision": spec.get("revision")})
        done.append(run["run"])
        del tr, model
    _publish(ctx, "WS4 tool-use RL runs", ("tools",))
    return {"trained": done}

