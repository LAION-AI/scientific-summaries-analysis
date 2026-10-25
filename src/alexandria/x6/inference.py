"""Stage `inference` (X6, GPU): what each representation costs at inference time, measured on the WS2 QA bench.

Per answer model (one SLURM array task per model) and per arm (the WS2 arms present in ws2/contexts, plus the full
text cut to N tokens, `ft@N`, the full-text accuracy-vs-tokens curve):
  latency     single requests (batch 1) on `latency_items` questions, prefix cache reset before every request:
              TTFT = time of a 1-token generation (prefill), e2e = time of the full answer, TPOT = (e2e - TTFT) /
              (output tokens - 1). Medians over items.
  throughput  every question of the arm in one batched call (as WS2 answers them: context first, so questions of one
              paper share a cached prefix; the cache is reset between arms): wall seconds, prompt / output tokens,
              GPU power sampled with nvidia-smi meanwhile (on a MIG slice it is the whole GPU's draw, or N/A).
The prompt is WS2's (qa.ANSWER_SYSTEM + answer_user); accuracy of the WS2 arms is joined from ws2/grade in `results`,
the `ft@N` answers are graded here by the same judge (SimpleQA protocol).
Outputs: answers/<model>.parquet, latency/<model>.parquet, throughput/<model>.parquet, arch/<model>.json.
"""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np
import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..qa import ANSWER_SYSTEM, answer_user, exact_match, grade_batch, token_f1
from ..runtime import Context, atomic_write_json, log, stage
from .common import arch, setting, stage_dir


class PowerSampler:
    """nvidia-smi power.draw every `every` seconds in a thread; mean watts (None when unavailable, e.g. MIG N/A)."""

    def __init__(self, every: float = 0.5):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split("\n")[0].strip()
                self.samples.append(float(out))
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join(timeout=10)

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.samples)) if self.samples else None


def _prompts(tok, spec: dict, items: list[dict], max_tokens: int) -> list[tuple[str, int]]:
    budget = int(spec.get("max_model_len", 32768)) - max_tokens - 512
    out = []
    for it in items:
        cut = min(budget, int(it["cut"])) if it.get("cut") else budget
        text, n, _ = truncate_tokens(tok, it["context"], cut) if it["context"] else ("", 0, False)
        out.append((chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user(it["question"], text)), n))
    return out


def _measure(llm, tok, spec, by_arm: dict[str, list[dict]], lat_qids: list[str], cfg: dict) -> dict:
    max_tokens = int(cfg.get("max_tokens", 64))
    full, first = json_params(None, max_tokens, 0.0, seed=0), json_params(None, 1, 0.0, seed=0)
    answers, lat, thr = [], [], []
    llm.generate([chat_prompt(tok, spec, ANSWER_SYSTEM, answer_user("Warm-up.", None))], full, use_tqdm=False)
    for arm, items in by_arm.items():
        prompts = _prompts(tok, spec, items, max_tokens)
        # latency: batch 1, cold prefix cache
        for it, (p, _) in [(it, p) for it, p in zip(items, prompts, strict=True) if it["qid"] in lat_qids]:
            llm.reset_prefix_cache()
            t0 = time.perf_counter()
            llm.generate([p], first, use_tqdm=False)
            ttft = time.perf_counter() - t0
            llm.reset_prefix_cache()
            t0 = time.perf_counter()
            o = llm.generate([p], full, use_tqdm=False)[0]
            e2e = time.perf_counter() - t0
            n_out = len(o.outputs[0].token_ids) if o.outputs else 0
            lat.append({"arm": arm, "qid": it["qid"], "prompt_tokens": len(o.prompt_token_ids or []),
                        "output_tokens": n_out, "ttft_s": ttft, "e2e_s": e2e,
                        "tpot_s": (e2e - ttft) / (n_out - 1) if n_out > 1 else None})
        # throughput: the whole arm batched (prefix sharing within the arm only)
        llm.reset_prefix_cache()
        with PowerSampler() as pw:
            t0 = time.perf_counter()
            outs = llm.generate([p for p, _ in prompts], full, use_tqdm=False)
            dt = time.perf_counter() - t0
        n_in = [len(o.prompt_token_ids or []) for o in outs]
        n_out = [len(o.outputs[0].token_ids) if o.outputs else 0 for o in outs]
        thr.append({"arm": arm, "n": len(items), "seconds": dt, "prompt_tokens": int(sum(n_in)),
                    "output_tokens": int(sum(n_out)), "power_w": pw.mean, "power_samples": len(pw.samples)})
        for it, (_, n_ctx), o, a, b in zip(items, prompts, outs, n_in, n_out, strict=True):
            text = o.outputs[0].text.strip() if o.outputs else ""
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            answers.append({"qid": it["qid"], "paper_key": it["paper_key"], "arm": arm, "context_tokens": n_ctx,
                            "prompt_tokens": a, "output_tokens": b, "prediction": text.split("\n")[0][:300]})
        log(f"inference: {spec['name']} {arm}: {len(items)} q, {dt:.1f}s, {sum(n_in) / max(dt, 1e-9):,.0f} prompt tok/s")
    return {"answers": answers, "latency": lat, "throughput": thr}


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


def grade_frame(ctx: Context, df: pd.DataFrame, judge_name: str, gpu_mem: float) -> pd.DataFrame:
    """Add SimpleQA grades (+ exact match, token F1) to a frame with question, answer, prediction."""
    judge = model_spec(ctx.cfg, judge_name)
    items = [{"question": r.question, "gold": r.answer, "predicted": r.prediction} for r in df.itertuples()]
    log(f"grading {len(items)} answers with {judge['name']}")
    df = df.copy()
    df["grade"] = with_model(judge, _grade, items, gpu_memory_utilization=gpu_mem) if items else []
    df["correct"] = (df["grade"] == "CORRECT").astype(int)
    df["attempted"] = (df["grade"] != "NOT_ATTEMPTED").astype(int)
    df["em"] = [int(exact_match(p, g)) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df["f1"] = [token_f1(p, g) for p, g in zip(df["prediction"], df["answer"], strict=True)]
    df["judge"] = judge["name"]
    return df


@stage("x6", "inference", kind="map", resources="gpu", env="llm", config_keys=("x6.inference", "x6.models", "x6.judge"),
       help="per model and arm: tokens, TTFT / TPOT / e2e latency (batch 1), batched throughput, GPU power; "
            "full text cut to N tokens graded for the accuracy-vs-tokens curve")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x6.inference")
    gpu_mem = float(cfg.get("gpu_memory_utilization", 0.85))
    qs = pd.read_parquet(stage_dir(ctx, "ws2", "questions") / "questions.parquet")
    cx = pd.read_parquet(stage_dir(ctx, "ws2", "contexts") / "contexts.parquet")
    have = set(cx["arm"])
    arms = [a for a in cfg.get("arms", []) if a in have]
    by = {(r.paper_key, r.arm): r.context for r in cx.itertuples()}
    qs = qs.sort_values(["paper_key", "qid"]).reset_index(drop=True)      # questions of one paper adjacent
    by_arm = {a: [{"qid": q.qid, "paper_key": q.paper_key, "question": q.question,
                   "context": by.get((q.paper_key, a), "")} for q in qs.itertuples()] for a in arms}
    if "full_text" in have:
        for b in cfg.get("truncation_budgets", []):
            by_arm[f"ft@{int(b)}"] = [{"qid": q.qid, "paper_key": q.paper_key, "question": q.question,
                                       "context": by.get((q.paper_key, "full_text"), ""), "cut": int(b)}
                                      for q in qs.itertuples()]
    n_lat = int(setting(ctx, "latency_items", cfg.get("latency_items", 60)))
    lat_qids = list(qs.sample(n=min(n_lat, len(qs)), random_state=0)["qid"])
    models = list(ctx.section("x6").get("models", ["qwen2.5-7b"]))[ctx.shard_index::ctx.shard_count]
    for d in ("answers", "latency", "throughput", "arch"):
        (ctx.out / d).mkdir(parents=True, exist_ok=True)
    done = []
    for name in models:
        if (ctx.out / "throughput" / f"{name}.parquet").exists() and not ctx.force:
            log(f"inference: {name} already done")
            continue
        spec = model_spec(ctx.cfg, name)
        atomic_write_json(ctx.out / "arch" / f"{name}.json", {**arch(ctx, spec), "model": name,
                                                             "quantization": spec.get("quantization"),
                                                             "max_model_len": spec.get("max_model_len")})
        log(f"inference: {name}: {len(by_arm)} arms x {len(qs)} questions; latency on {len(lat_qids)}")
        res = with_model(spec, _measure, by_arm, lat_qids, cfg, gpu_memory_utilization=gpu_mem)
        ans = pd.DataFrame(res["answers"]).assign(model=name)
        ft = ans[ans["arm"].str.startswith("ft@")].merge(qs[["qid", "question", "answer"]], on="qid")
        if len(ft):
            ft = grade_frame(ctx, ft, ctx.section("x6").get("judge", "qwen3-8b"), gpu_mem)
            ans = pd.concat([ans[~ans["arm"].str.startswith("ft@")],
                             ft.drop(columns=["question", "answer"])], ignore_index=True)
        ans.to_parquet(ctx.out / "answers" / f"{name}.parquet", index=False)
        pd.DataFrame(res["latency"]).assign(model=name).to_parquet(ctx.out / "latency" / f"{name}.parquet", index=False)
        pd.DataFrame(res["throughput"]).assign(model=name).to_parquet(ctx.out / "throughput" / f"{name}.parquet",
                                                                      index=False)
        done.append(name)
    thr = [pd.read_parquet(p) for p in sorted((ctx.out / "throughput").glob("*.parquet"))]
    rows = pd.concat(thr).to_dict("records") if thr else []
    ctx.publish("X6 inference measurements", [
        {"type": "table", "title": "Batched throughput per model and arm", "hint": "prefix cache reset between arms",
         "columns": [{"key": "model", "label": "model"}, {"key": "arm", "label": "arm"},
                     {"key": "n", "label": "questions", "fmt": "int"}, {"key": "seconds", "label": "seconds", "fmt": "float"},
                     {"key": "prompt_tokens", "label": "prompt tokens", "fmt": "int"},
                     {"key": "output_tokens", "label": "output tokens", "fmt": "int"},
                     {"key": "power_w", "label": "GPU W (nvidia-smi)", "fmt": "float"}],
         "rows": rows, "limit": 60}], summary=f"{len(thr)} models measured")
    return {"models_done": done, "arms": list(by_arm), "latency_items": len(lat_qids)}
