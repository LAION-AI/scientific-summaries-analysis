"""Stages `eval` and `grade` (WS4).

eval   For every base model and every trained adapter (split over SLURM array tasks):
         1. lm-evaluation-harness (vLLM backend, LoRA adapter, chat template, zero-shot unless a task needs shots),
            closed book: the science set (main claim: GPQA-Diamond, MMLU-Pro STEM, SciQ, PubMedQA question-only, HLE
            text) and the general set (forgetting: MMLU, GSM8K, IFEval). Gated or unsupported sets are listed as not
            run (GPQA needs HF_TOKEN; HLE is not in lm-eval and gated).
         2. Knowledge probe: closed-book short answers to the probe questions on trained and held-out papers.
       Output per unit: units/<unit>/lm_eval.json (metrics with standard errors), units/<unit>/probe.parquet.
grade  SimpleQA judge (Qwen3-8B, as WS2) over every probe answer + model-free exact match / token F1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, with_model
from ..qa import ANSWER_SYSTEM, answer_user, exact_match, grade_batch, is_abstention, token_f1
from ..runtime import Context, atomic_write_json, log, stage
from .common import run_grid, setting, stage_table

# eval set -> lm-eval tasks (None = not runnable with lm-eval here)
LM_TASKS = {
    "gpqa_diamond": ["gpqa_diamond_zeroshot"],
    "mmlu_pro_stem": [f"mmlu_pro_{s}" for s in ("biology", "chemistry", "computer_science", "engineering", "math",
                                                "physics")],
    "sciq": ["sciq"],
    "pubmedqa_closed": ["pubmedqa_closedbook"],
    "hle_text": None,
    "mmlu": ["mmlu"],
    "gsm8k": ["gsm8k"],
    "ifeval": ["ifeval"],
}
GATED = {"gpqa_diamond": ("Idavidrein/gpqa", "gpqa_diamond.csv")}   # repo, a data file only access grants
# metric reported per task (first one present): MMLU-Pro answers are extracted from generations ("custom-extract");
# GSM8K uses flexible extraction (chat models rarely emit the strict "#### n" format); IFEval prompt-level strict
METRICS = ("acc,none", "exact_match,custom-extract", "exact_match,flexible-extract", "prompt_level_strict_acc,none",
           "acc_norm,none", "exact_match,strict-match")
TRAIN_STAGE = {"sft-doc": "train-sft", "sft-qa": "train-sft", "grpo": "train-grpo", "tools": "train-tools"}

PUBMEDQA_YAML = """task: pubmedqa_closedbook
dataset_path: parquet
dataset_kwargs:
  data_files:
    test: {path}
test_split: test
output_type: multiple_choice
doc_to_text: "Question: {{{{question}}}}\\nAnswer with yes, no or maybe:"
doc_to_choice: ["yes", "no", "maybe"]
doc_to_target: "{{{{['yes', 'no', 'maybe'].index(final_decision)}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def eval_units(ctx: Context) -> list[dict]:
    """Base models + every training run of the grid (whether trained yet or not)."""
    models = setting(ctx, "models", ctx.section("ws4.train").get("models", ["qwen3-4b"]))
    return [{"run": f"{m}__base", "model": m, "method": "base", "condition": "base", "seed": 0} for m in models] + \
        run_grid(ctx)


def adapter_dir(ctx: Context, unit: dict) -> Path | None:
    if unit["method"] == "base":
        return None
    return ctx.cfg.stage_dir("ws4", TRAIN_STAGE[unit["method"]]) / "runs" / unit["run"] / "adapter"


def lm_tasks(ctx: Context) -> tuple[list[str], dict[str, str]]:
    cfg = ctx.section("ws4.eval")
    hub = ctx.cfg.site.hf_home / "hub"
    tasks, skipped = [], {}
    for s in list(cfg.get("science", [])) + list(cfg.get("general", [])):
        t = LM_TASKS.get(s)
        if t is None:
            skipped[s] = "not available in lm-eval (and gated)" if s == "hle_text" else "unknown eval set"
        elif s in GATED and not any((hub / f"datasets--{GATED[s][0].replace('/', '--')}" / "snapshots").glob(f"*/{GATED[s][1]}")):
            skipped[s] = f"gated: needs HF_TOKEN and `alexandria fetch-ws4` ({GATED[s][0]})"
        else:
            tasks += t
    return tasks, skipped


def _pubmedqa_path(ctx: Context) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download("qiaojin/PubMedQA", "pqa_labeled/train-00000-of-00001.parquet", repo_type="dataset",
                                cache_dir=str(ctx.cfg.site.hf_home / "hub"), local_files_only=True))


def local_snapshot(ctx: Context, spec: dict) -> str:
    """The pinned snapshot's local directory: offline, a repo id + revision is not enough for every library call
    (vLLM re-reads the config by repo id without the revision after loading)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(spec["hf"], revision=spec.get("revision"), cache_dir=str(ctx.cfg.site.hf_home / "hub"),
                             local_files_only=True)


def run_lm_eval(ctx: Context, unit: dict, out: Path, tasks: list[str]) -> dict:
    cfg = ctx.section("ws4.eval")
    spec = model_spec(ctx.cfg, unit["model"])
    tdir = out / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "pubmedqa_closedbook.yaml").write_text(PUBMEDQA_YAML.format(path=_pubmedqa_path(ctx)))
    margs = [f"pretrained={local_snapshot(ctx, spec)}", "dtype=bfloat16",
             f"gpu_memory_utilization={cfg.get('gpu_memory_utilization', 0.85)}",
             f"max_model_len={cfg.get('max_model_len', 4096)}", "enable_prefix_caching=True"]
    ad = adapter_dir(ctx, unit)
    if ad is not None:
        margs += ["enable_lora=True", f"max_lora_rank={ctx.section('ws4.train').get('lora', {}).get('r', 64)}",
                  f"lora_local_path={ad}"]
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm", "--model_args", ",".join(margs),
           "--tasks", ",".join(tasks), "--include_path", str(tdir), "--batch_size", "auto",
           "--output_path", str(out / "lm_eval"), "--apply_chat_template", "--seed", "0"]
    limit = setting(ctx, "eval_limit", cfg.get("limit"))
    if limit:
        cmd += ["--limit", str(limit)]
    if cfg.get("num_fewshot") is not None:
        cmd += ["--num_fewshot", str(cfg["num_fewshot"])]
    log(f"eval: {unit['run']}: lm-eval {len(tasks)} tasks" + (f" (limit {limit})" if limit else ""))
    subprocess.run(cmd, check=True)
    path = sorted((out / "lm_eval").rglob("results_*.json"))[-1]
    return {"rows": extract_rows(json.loads(path.read_text())), "limit": limit, "raw": str(path)}


def extract_rows(res: dict) -> list[dict]:
    """One row per task: the first available metric of METRICS with its standard error and sample count."""
    rows = []
    for task, m in res["results"].items():
        key = next((k for k in METRICS if k in m), None)
        if key is None:
            continue
        stderr = m.get(key.replace(",", "_stderr,"))
        rows.append({"task": task, "metric": key.split(",")[0], "value": float(m[key]),
                     "stderr": float(stderr) if isinstance(stderr, (int, float)) else None,
                     "n": (res.get("n-samples", {}).get(task) or {}).get("effective")})
    return rows


def _probe(llm, tok, spec, items: list[dict], lora: str | None, rank: int) -> list[str]:
    from vllm.lora.request import LoRARequest
    prompts = [chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(q["question"], None)) for q in items]
    req = LoRARequest("adapter", 1, lora) if lora else None
    outs = llm.generate(prompts, json_params(None, 48), lora_request=req, use_tqdm=False)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]


def probe_answers(ctx: Context, unit: dict, probe: pd.DataFrame) -> pd.DataFrame:
    cfg = ctx.section("ws4.eval")
    spec = {**model_spec(ctx.cfg, unit["model"]), "max_model_len": int(cfg.get("max_model_len", 4096))}
    rank = int(ctx.section("ws4.train").get("lora", {}).get("r", 64))
    ad = adapter_dir(ctx, unit)
    if ad is not None:
        spec = {**spec, "enable_lora": True, "max_lora_rank": rank}
    items = probe[["qid", "question"]].to_dict("records")
    preds = with_model(spec, _probe, items, str(ad) if ad else None, rank,
                       gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    return probe[["qid", "paper_key", "split"]].assign(run=unit["run"], prediction=preds)


@stage("ws4", "eval", kind="map", needs=("probe",), resources="gpu", env="train", config_keys=("ws4.eval",),
       help="lm-eval closed book (science + forgetting) and the knowledge probe, per base model and adapter")
def run_eval(ctx: Context) -> dict:
    cfg = ctx.section("ws4.eval")
    tasks, skipped = lm_tasks(ctx)
    probe = stage_table(ctx, "probe", "probe.parquet") if cfg.get("knowledge", True) else pd.DataFrame()
    units = eval_units(ctx)[ctx.shard_index::ctx.shard_count]
    done, missing = [], []
    for u in units:
        out = ctx.out / "units" / u["run"]
        ad = adapter_dir(ctx, u)
        if ad is not None and not ad.exists():
            missing.append(u["run"])
            continue
        if (out / "lm_eval.json").exists() and not ctx.force:
            continue
        out.mkdir(parents=True, exist_ok=True)
        res = run_lm_eval(ctx, u, out, tasks) if tasks else {"rows": [], "limit": None}
        if len(probe):
            probe_answers(ctx, u, probe).to_parquet(out / "probe.parquet", index=False)
        atomic_write_json(out / "lm_eval.json", {**u, **res, "skipped": skipped})
        done.append(u["run"])
    ctx.write_json(f"skipped_sets.json", skipped)
    n_done = len(list((ctx.out / "units").glob("*/lm_eval.json"))) if (ctx.out / "units").exists() else 0
    ctx.publish("WS4 evaluation runs", [
        {"type": "kpis", "items": [
            {"label": "models / adapters evaluated", "value": n_done, "fmt": "int"},
            {"label": "lm-eval tasks", "value": len(tasks), "fmt": "int"},
            {"label": "adapters not trained yet", "value": len(missing), "fmt": "int"}]},
        {"type": "table", "title": "Eval sets not run", "rows": [{"set": k, "reason": v} for k, v in skipped.items()],
         "columns": [{"key": "set", "label": "eval set"}, {"key": "reason", "label": "reason"}]},
    ], summary=f"{n_done} units evaluated")
    return {"evaluated": done, "missing_adapters": missing, "skipped_sets": skipped}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("ws4", "grade", needs=("eval",), resources="gpu", env="llm", config_keys=("ws4.eval",),
       help="SimpleQA judge over every knowledge-probe answer (+ exact match, token F1)")
def run_grade(ctx: Context) -> dict:
    cfg = ctx.section("ws4.eval")
    units = ctx.cfg.stage_dir("ws4", "eval") / "units"
    frames = [pd.read_parquet(p) for p in sorted(units.glob("*/probe.parquet"))]
    if not frames:
        return ctx.skip("WS4 probe grading", "no probe answers yet (run eval)")
    df = pd.concat(frames, ignore_index=True).merge(stage_table(ctx, "probe", "probe.parquet")[["qid", "question", "answer"]],
                                                    on="qid", how="left")
    judge = model_spec(ctx.cfg, cfg.get("judge", "qwen3-8b"))
    items = [{"question": r.question, "gold": r.answer, "predicted": r.prediction} for r in df.itertuples()]
    log(f"grade: {len(items)} probe answers with {judge['name']}")
    df["grade"] = with_model(judge, _grade, items, gpu_memory_utilization=0.85)
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["abstained"] = [int(is_abstention(p)) for p in df["prediction"]]
    df["em"] = [int(exact_match(p, g)) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df["f1"] = [token_f1(p, g) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df.drop(columns=["question"]).to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("WS4 probe grading", [{"type": "kpis", "items": [
        {"label": "probe answers graded", "value": len(df), "fmt": "int"},
        {"label": "judge", "value": judge["name"]},
        {"label": "CORRECT overall", "value": float(df["correct"].mean()), "fmt": "pct1"}]}],
        summary=f"{len(df)} answers graded")
    return {"graded": len(df)}
