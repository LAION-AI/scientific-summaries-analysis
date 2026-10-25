"""Stage `train-efficiency` (X6, GPU, env train): data / compute efficiency of knowledge injection by post-training.

For each condition (docs_ku: the papers' KUs, docs_summary: prose summaries, docs_full: the papers' full text; WS4's
token-matched, decontaminated document sets) one LoRA continued-pretraining run (WS4 sft-doc recipe: packed documents,
causal-LM loss) of `train.model` runs for max(fractions) x the WS4 token budget, with a constant learning rate after
warmup so the intermediate checkpoints saved at each fraction are valid shorter-budget runs (as in WSD-style
data-scaling studies). Every checkpoint and the base model answer the WS4 knowledge probe closed book (questions on
trained papers = `train`, on held-out papers = `heldout`), graded by the SimpleQA judge.
Training tokens are counted as steps x seq x batch x accumulation (packed); training FLOPs = 6 N tokens.
Outputs: runs/<condition>/train.json (+ step*/ adapters), probe.parquet.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import time

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, with_model
from ..qa import ANSWER_SYSTEM, answer_user
from ..runtime import Context, atomic_write_json, log, stage
from .common import setting, stage_dir
from .inference import grade_frame


def budget_steps(total_tokens: int, fractions: list[float], tokens_per_step: int) -> dict[float, int]:
    return {float(f): max(1, math.ceil(float(f) * total_tokens / tokens_per_step)) for f in fractions}


def _train_child(q, hf: str, revision: str | None, texts: list[str], out_dir: str, save_steps: list[int],
                 tcfg: dict, seed: int) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer

        tok = AutoTokenizer.from_pretrained(hf, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        lc = tcfg.get("lora", {})
        lora = LoraConfig(r=int(lc.get("r", 64)), lora_alpha=int(lc.get("alpha", 128)),
                          lora_dropout=float(lc.get("dropout", 0.05)),
                          target_modules=lc.get("target_modules", "all-linear"), task_type="CAUSAL_LM")
        marks: dict[int, float] = {}
        t0 = time.time()

        class Save(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step in save_steps:
                    torch.cuda.synchronize()
                    marks[state.global_step] = time.time() - t0
                    model.save_pretrained(f"{out_dir}/step{state.global_step}")

        args = SFTConfig(output_dir=f"{out_dir}/ckpt", max_length=int(tcfg.get("seq", 2048)), packing=True,
                         per_device_train_batch_size=int(tcfg.get("per_device_batch", 4)),
                         gradient_accumulation_steps=int(tcfg.get("grad_accum", 1)),
                         learning_rate=float(tcfg.get("lr", 1e-4)), lr_scheduler_type="constant_with_warmup",
                         warmup_steps=float(tcfg.get("warmup_ratio", 0.03)), max_steps=max(save_steps), bf16=True,
                         logging_steps=1, save_strategy="no", report_to="none", seed=seed, data_seed=seed,
                         gradient_checkpointing=True, dataset_text_field="text")
        tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_dict({"text": texts}), processing_class=tok,
                        peft_config=lora, callbacks=[Save()])
        tr.train()
        hist = [h for h in tr.state.log_history if "loss" in h]
        q.put(("ok", {"marks": marks, "seconds": time.time() - t0, "steps": tr.state.global_step, "log": hist,
                      "gpu": torch.cuda.get_device_name(0)}))
    except BaseException as e:                         # noqa: BLE001  (re-raised in the parent)
        import traceback
        q.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def train_run(spec: dict, texts: list[str], out_dir, save_steps: list[int], tcfg: dict, seed: int = 0) -> dict:
    """One training run in a spawned process (frees the GPU completely before vLLM loads for the probe)."""
    c = mp.get_context("spawn")
    q = c.Queue()
    p = c.Process(target=_train_child, args=(q, spec["hf"], spec.get("revision"), texts, str(out_dir), save_steps,
                                             tcfg, seed))
    p.start()
    status, payload = q.get()
    p.join()
    if status != "ok":
        raise RuntimeError(f"training failed:\n{payload}")
    return payload


def _probe_all(llm, tok, spec, items: list[dict], adapters: list[tuple[str, str | None]]) -> list[dict]:
    from vllm.lora.request import LoRARequest
    prompts = [chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(q["question"], None)) for q in items]
    params = json_params(None, 48, 0.0, seed=0)
    rows = []
    for i, (tag, path) in enumerate(adapters, 1):
        req = LoRARequest(tag, i, path) if path else None
        outs = llm.generate(prompts, params, lora_request=req, use_tqdm=False)
        rows += [{"tag": tag, "qid": q["qid"], "prediction": (o.outputs[0].text.strip() if o.outputs else "").split("\n")[0][:300]}
                 for q, o in zip(items, outs, strict=True)]
    return rows


@stage("x6", "train-efficiency", resources="gpu", env="train", config_keys=("x6.train", "x6.judge"),
       help="knowledge-injection LoRA runs on KU / summary / full-text docs at 0.25-2x the token budget: "
            "probe accuracy vs training tokens and GPU-hours")
def run(ctx: Context) -> dict:
    tcfg = ctx.section("x6.train")
    name = setting(ctx, "train_model", tcfg.get("model", "qwen2.5-7b"))
    spec = model_spec(ctx.cfg, name)
    budget = json.loads((stage_dir(ctx, "ws4", "data") / "budget.json").read_text())
    total = int(budget["total"])
    tps = int(tcfg.get("seq", 2048)) * int(tcfg.get("per_device_batch", 4)) * int(tcfg.get("grad_accum", 1))
    fr_steps = budget_steps(total, tcfg.get("fractions", [0.25, 0.5, 1.0, 2.0]), tps)
    runs = {}
    for cond in tcfg.get("conditions", ["ku", "summary", "full"]):
        out = ctx.out / "runs" / cond
        meta = out / "train.json"
        if meta.exists() and not ctx.force:
            runs[cond] = json.loads(meta.read_text())
            continue
        docs = pd.read_parquet(stage_dir(ctx, "ws4", "decontam") / f"docs_{cond}.parquet")
        log(f"train-efficiency: {name} on docs_{cond}: {len(docs)} docs, {int(docs['tokens'].sum()):,} tokens; "
            f"checkpoints at steps {sorted(fr_steps.values())} ({tps:,} tokens/step)")
        out.mkdir(parents=True, exist_ok=True)
        res = train_run(spec, docs["text"].tolist(), out, sorted(set(fr_steps.values())), tcfg)
        runs[cond] = {"condition": cond, "model": name, "base": spec["hf"], "docs": len(docs),
                      "doc_tokens": int(docs["tokens"].sum()), "budget_tokens": total, "tokens_per_step": tps,
                      "fractions": {str(f): s for f, s in fr_steps.items()}, **res}
        atomic_write_json(meta, runs[cond])
    probe = pd.read_parquet(stage_dir(ctx, "ws4", "probe") / "probe.parquet")
    probe = probe[probe["status"] == "ok"] if "status" in probe else probe
    adapters: list[tuple[str, str | None]] = [("base", None)]
    for cond, r in runs.items():
        adapters += [(f"{cond}@{f}", str(ctx.out / "runs" / cond / f"step{s}")) for f, s in r["fractions"].items()]
    pspec = {**spec, "quantization": None, "enable_lora": True,
             "max_lora_rank": int(tcfg.get("lora", {}).get("r", 64)),
             "max_model_len": int(tcfg.get("probe_max_model_len", 4096))}
    log(f"train-efficiency: probing {len(adapters)} models on {len(probe)} questions")
    rows = with_model(pspec, _probe_all, probe[["qid", "question"]].to_dict("records"), adapters,
                      gpu_memory_utilization=0.6)
    df = pd.DataFrame(rows).merge(probe[["qid", "paper_key", "split", "question", "answer"]], on="qid")
    df = grade_frame(ctx, df, ctx.section("x6").get("judge", "qwen3-8b"), 0.85)
    df["condition"] = [t.split("@")[0] for t in df["tag"]]
    df["fraction"] = [float(t.split("@")[1]) if "@" in t else 0.0 for t in df["tag"]]

    def step_of(r) -> int:
        return int(runs[r.condition]["fractions"][str(r.fraction)]) if r.condition in runs else 0

    df["step"] = [step_of(r) for r in df.itertuples()]
    df["train_tokens"] = df["step"] * tps
    df["train_seconds"] = [float(runs[c]["marks"].get(str(s), runs[c]["marks"].get(s, float("nan"))))
                           if c in runs else 0.0 for c, s in zip(df["condition"], df["step"], strict=True)]
    df.drop(columns=["question", "answer"]).assign(model=name).to_parquet(ctx.out / "probe.parquet", index=False)
    t = df.groupby(["condition", "fraction", "split"]).agg(accuracy=("correct", "mean"), tokens=("train_tokens", "first"),
                                                          seconds=("train_seconds", "first")).reset_index()
    ctx.publish("X6 training efficiency", [
        {"type": "markdown", "text": f"Model **{name}** (LoRA r={tcfg.get('lora', {}).get('r', 64)}), "
         f"{tps:,} tokens/step, WS4 budget {total:,} tokens = 1x. Profile `{ctx.cfg.profile}`."},
        {"type": "table", "title": "Knowledge-probe accuracy per condition and training budget",
         "columns": [{"key": "condition", "label": "condition"}, {"key": "fraction", "label": "x budget", "fmt": "float"},
                     {"key": "split", "label": "probe split"}, {"key": "tokens", "label": "train tokens", "fmt": "int"},
                     {"key": "seconds", "label": "GPU-s", "fmt": "float"},
                     {"key": "accuracy", "label": "accuracy", "fmt": "pct1"}], "rows": t.to_dict("records"), "limit": 60}],
        summary=f"{len(runs)} runs, {len(adapters)} probed")
    return {"model": name, "runs": list(runs), "probed": len(adapters)}
