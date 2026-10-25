"""Stages `train` and `generate` (WS5, GPU, Alex-train env).

train     Distils the validated teacher KUs into each student (official BF16 weights), LoRA and full SFT, at every
          learning-curve size (first n of a fixed shuffled order, so the sets are nested). Prompt = the summariser
          system prompt + the paper (cut to max_source_tokens with the student's tokenizer); completion = the
          teacher's KU JSON; loss on the completion only. Output: runs/<run>/adapter (LoRA) or runs/<run>/model (full).
generate  Every system writes the KU of every TEST paper with the same prompt and JSON schema (vLLM structured
          output), timed for tokens per paper and papers per GPU-hour:
            student_zero  each student zero-shot with the current prompt
            <run>         each distilled student
            released      the dataset's existing v2.0 KU of the paper (no generation)
            extractor     the 27B-class extractor baseline (ws5.generate.extractor), when it fits the GPU
            teacher       the teacher's KU (ceiling; from `teacher`)
          Systems are split over SLURM array tasks; output parts/<system>.parquet.
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from ..data import fetch_records
from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..runtime import Context, atomic_write_json, log, stage
from ..schema import SUMMARY_FIELDS
from .common import SYSTEM, ku_schema, setting, stage_table, user_prompt


def runs(ctx: Context) -> list[dict]:
    cfg = ctx.section("ws5.train")
    out = []
    for s in setting(ctx, "students", cfg.get("students", [])):
        for m in cfg.get("methods", ["lora"]):
            for n in setting(ctx, "sizes", cfg.get("sizes", [250])):
                out.append({"run": f"{s}__{m}__n{n}", "student": s, "method": m, "size": int(n)})
    return out


def train_rows(ctx: Context, tok, n: int, max_src: int) -> list[dict]:
    papers = stage_table(ctx, "papers", "papers.parquet").set_index("paper_key")
    v = stage_table(ctx, "validate", "validated.parquet")
    v = v[(v["split"] == "train") & v["passed"]].sample(frac=1.0, random_state=0).head(n)   # nested learning curve
    rows = []
    for r in v.itertuples():
        text, _, cut = truncate_tokens(tok, papers.at[r.paper_key, "source"], max_src)
        rows.append({"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(text, cut)}],
                     "completion": [{"role": "assistant", "content": json.dumps(json.loads(r.ku), ensure_ascii=False)}]})
    return rows


@stage("ws5", "train", kind="map", needs=("validate",), resources="gpu", env="train", config_keys=("ws5.train",),
       help="distil teacher KUs into each student: LoRA vs full SFT, learning curve over training-set sizes")
def run_train(ctx: Context) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    cfg = ctx.section("ws5.train")
    max_src = int(setting(ctx, "max_source_tokens", cfg.get("max_source_tokens", 16000)))
    done, rows_pub = [], []
    for run in runs(ctx)[ctx.shard_index::ctx.shard_count]:
        out = ctx.out / "runs" / run["run"]
        if (out / "train.json").exists() and not ctx.force:
            continue
        spec = model_spec(ctx.cfg, run["student"])
        tok = AutoTokenizer.from_pretrained(spec["hf"], revision=spec.get("revision"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        data = train_rows(ctx, tok, run["size"], max_src)
        smaller = [r for r in runs(ctx) if r["student"] == run["student"] and r["method"] == run["method"]
                   and len(data) <= r["size"] < run["size"]]
        if len(data) < run["size"] and smaller:     # same data as a smaller point of the curve: do not train it twice
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out / "capped.json", {**run, "examples": len(data), "same_as": smaller[0]["run"]})
            log(f"train: {run['run']}: only {len(data)} valid teacher KUs, same data as {smaller[0]['run']}: skipped")
            continue
        if len(data) < run["size"]:
            log(f"train: {run['run']}: only {len(data)} valid teacher KUs (learning-curve point capped)")
        model = AutoModelForCausalLM.from_pretrained(spec["hf"], revision=spec.get("revision"), dtype=torch.bfloat16,
                                                     attn_implementation="sdpa")
        lora = run["method"] == "lora"
        lc = cfg.get("lora", {})
        args = SFTConfig(output_dir=str(out / "ckpt"), max_length=max_src + int(cfg.get("max_target_tokens", 4096)) + 256,
                         packing=False, completion_only_loss=True, per_device_train_batch_size=int(cfg.get("per_device_batch", 1)),
                         gradient_accumulation_steps=int(cfg.get("grad_accum", 16)),
                         learning_rate=float(cfg.get("lr_lora" if lora else "lr_full", 1e-4)), lr_scheduler_type="cosine",
                         warmup_steps=0.05, num_train_epochs=float(cfg.get("epochs", 2)),
                         max_steps=int(setting(ctx, "max_steps", cfg.get("max_steps") or -1) or -1), bf16=True,
                         gradient_checkpointing=True, logging_steps=1, save_strategy="no", report_to="none", seed=0)
        t0 = time.time()
        tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(data), processing_class=tok,
                        peft_config=LoraConfig(r=int(lc.get("r", 64)), lora_alpha=int(lc.get("alpha", 128)),
                                               lora_dropout=float(lc.get("dropout", 0.05)),
                                               target_modules=lc.get("target_modules", "all-linear"),
                                               task_type="CAUSAL_LM") if lora else None)
        tr.train()
        if lora:
            tr.model.save_pretrained(out / "adapter")
        else:
            tr.model.save_pretrained(out / "model")
            tok.save_pretrained(out / "model")
        hist = [h for h in tr.state.log_history if "loss" in h]
        atomic_write_json(out / "train.json", {**run, "examples": len(data), "steps": tr.state.global_step,
                                               "seconds": round(time.time() - t0, 1), "log": hist, "base": spec["hf"]})
        done.append(run["run"])
        del tr, model
        torch.cuda.empty_cache()
    for run in runs(ctx):
        p = ctx.out / "runs" / run["run"] / "train.json"
        if p.exists():
            j = json.loads(p.read_text())
            ls = [h["loss"] for h in j["log"]]
            rows_pub.append({**run, "examples": j["examples"], "steps": j["steps"], "first_loss": ls[0] if ls else None,
                             "final_loss": ls[-1] if ls else None, "minutes": j["seconds"] / 60})
    ctx.publish("WS5 student training", [
        {"type": "kpis", "items": [{"label": "runs finished", "value": len(rows_pub), "fmt": "int"},
                                   {"label": "runs planned", "value": len(runs(ctx)), "fmt": "int"}]},
        {"type": "table", "title": "Distillation runs", "rows": rows_pub, "columns": [
            {"key": "run", "label": "run"}, {"key": "method", "label": "method"}, {"key": "examples", "label": "examples", "fmt": "int"},
            {"key": "steps", "label": "steps", "fmt": "int"}, {"key": "first_loss", "label": "first loss", "fmt": "float3"},
            {"key": "final_loss", "label": "final loss", "fmt": "float3"}, {"key": "minutes", "label": "minutes", "fmt": "float"}]},
    ], summary=f"{len(rows_pub)} of {len(runs(ctx))} runs")
    return {"trained": done}


# ---------------------------------------------------------------- generation
def _generate(llm, tok, spec, items: list[dict], cfg: dict, lora: str | None) -> dict:
    prompts = []
    for it in items:
        text, _, cut = truncate_tokens(tok, it["source"], int(cfg["max_source_tokens"]))
        prompts.append(chat_prompt(tok, spec, SYSTEM, user_prompt(text, cut)))
    req = None
    if lora:
        from vllm.lora.request import LoRARequest
        req = LoRARequest("student", 1, lora)
    t0 = time.time()
    outs = llm.generate(prompts, json_params(ku_schema(), int(cfg["max_tokens"])), lora_request=req, use_tqdm=False)
    sec = time.time() - t0
    return {"seconds": sec, "rows": [{"paper_key": it["paper_key"], "text": o.outputs[0].text if o.outputs else None,
                                      "tokens_out": len(o.outputs[0].token_ids) if o.outputs else 0}
                                     for it, o in zip(items, outs, strict=True)]}


def systems(ctx: Context) -> list[dict]:
    cfg = ctx.section("ws5.generate")
    base = setting(ctx, "baselines", cfg.get("baselines", ["student_zero", "released", "teacher"]))
    out = []
    for b in base:
        if b == "student_zero":
            out += [{"system": f"{s}__zero", "kind": "model", "model": s} for s in setting(ctx, "students", ctx.section("ws5.train").get("students", []))]
        elif b == "extractor":
            out.append({"system": "extractor", "kind": "model", "model": cfg.get("extractor", "gemma4-31b")})
        else:
            out.append({"system": b, "kind": b})
    tr = ctx.cfg.stage_dir("ws5", "train") / "runs"
    for r in runs(ctx):
        d = tr / r["run"]
        if (d / "capped.json").exists():                 # duplicate learning-curve point (too few teacher KUs)
            continue
        out.append({"system": r["run"], "kind": "model", "model": r["student"],
                    "lora": str(d / "adapter") if r["method"] == "lora" else None,
                    "path": str(d / "model") if r["method"] == "full" else None})
    return out


def _released(ctx: Context, test: pd.DataFrame) -> list[dict]:
    recs = fetch_records(ctx.cfg, list(zip(test["file"], test["row"], strict=True)), list(SUMMARY_FIELDS))
    rows = []
    for p in test.itertuples():
        rec = recs.get((p.file, int(p.row))) or {}
        ku = {f: rec.get(f) for f in SUMMARY_FIELDS}
        if isinstance(ku.get("claims"), str):
            try:
                ku["claims"] = json.loads(ku["claims"])
            except json.JSONDecodeError:
                pass
        rows.append({"paper_key": p.paper_key, "text": json.dumps(ku, ensure_ascii=False, default=str), "tokens_out": None})
    return rows


@stage("ws5", "generate", kind="map", needs=("train",), resources="gpu", env="train", config_keys=("ws5.generate",),
       help="every system writes the KU of every test paper (students zero-shot / distilled, released, extractor, teacher)")
def run_generate(ctx: Context) -> dict:
    from huggingface_hub import snapshot_download
    cfg = {**ctx.section("ws5.generate")}
    cfg["max_source_tokens"] = int(setting(ctx, "max_source_tokens", cfg.get("max_source_tokens", 16000)))
    test = stage_table(ctx, "papers", "papers.parquet")
    test = test[test["split"] == "test"].reset_index(drop=True)
    items = [{"paper_key": p.paper_key, "source": p.source} for p in test.itertuples()]
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done, skipped = [], {}
    for s in systems(ctx)[ctx.shard_index::ctx.shard_count]:
        dst = parts / f"{s['system']}.parquet"
        if dst.exists() and not ctx.force:
            continue
        if s["kind"] == "released":
            rows, sec = _released(ctx, test), None
        elif s["kind"] == "teacher":
            if not (ctx.cfg.stage_dir("ws5", "teacher") / "teacher.parquet").exists():
                skipped[s["system"]] = "teacher not run yet"           # baselines can be scored without it
                continue
            t = stage_table(ctx, "teacher", "teacher.parquet")
            t = t[t["split"] == "test"]
            rows = t[["paper_key", "text", "tokens_out"]].to_dict("records")
            sec = float(t["gpu_seconds"].sum()) if "gpu_seconds" in t else None
        else:
            spec = model_spec(ctx.cfg, s["model"])
            trained = s.get("path") or s.get("lora")
            if trained and not os.path.exists(trained):
                skipped[s["system"]] = "not trained yet"
                continue
            local = s.get("path") or snapshot_download(spec["hf"], revision=spec.get("revision"),
                                                       cache_dir=str(ctx.cfg.site.hf_home / "hub"), local_files_only=True)
            spec = {**spec, "hf": local, "revision": None}          # offline: local snapshot / trained model dir
            spec = {**spec, "max_model_len": cfg["max_source_tokens"] + int(cfg.get("max_tokens", 4096)) + 1024}
            if s.get("lora"):
                spec = {**spec, "enable_lora": True, "max_lora_rank": int(ctx.section("ws5.train").get("lora", {}).get("r", 64))}
            try:
                res = with_model(spec, _generate, items, cfg, s.get("lora"),
                                 gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
            except RuntimeError as e:                                  # e.g. the 31B extractor on a 20 GB slice
                skipped[s["system"]] = f"failed: {str(e).splitlines()[0][:160]}"
                log(f"generate: {s['system']} failed: {str(e)[:300]}")
                continue
            rows, sec = res["rows"], res["seconds"]
        df = pd.DataFrame(rows).assign(system=s["system"], kind=s["kind"],
                                       gpu_seconds_per_paper=(sec / len(rows)) if sec else None)
        df.to_parquet(dst, index=False)
        done.append(s["system"])
    ctx.write_json(f"skipped.task{ctx.shard_index}.json", skipped)
    n = len(list(parts.glob("*.parquet")))
    ctx.publish("WS5 test-set KUs", [{"type": "kpis", "items": [
        {"label": "systems with test KUs", "value": n, "fmt": "int"},
        {"label": "test papers", "value": len(items), "fmt": "int"}]},
        {"type": "table", "title": "Systems not generated", "rows": [{"system": k, "reason": v} for k, v in skipped.items()],
         "columns": [{"key": "system", "label": "system"}, {"key": "reason", "label": "reason"}]}],
        summary=f"{n} systems")
    return {"generated": done, "skipped": skipped}
