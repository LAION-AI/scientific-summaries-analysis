"""Stages `grade` and `results` of X3.

grade    One answer table for every system: Alexandria arms (constrained answers), OpenScholar-8B (parsed from its
         'Answer:' line), PaperQA2 (parsed likewise). Labels / option letters: exact match (the official metric of
         ScholarQABench's single-paper tasks and of LitQA2); QA-bench short answers: the SimpleQA judge (as WS2 / WS3).
results  Accuracy per benchmark x system with 95% bootstrap CIs (clusters: gold papers for the QA bench, items
         otherwise); LitQA2 also precision = correct / answered (answers other than "Insufficient information").
         Accuracy vs cost: retrieved / prompt / generated tokens, GPU seconds and wall clock per question for every
         system, the Pareto front of accuracy vs tokens per question.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..llm import model_spec, with_model
from ..qa import grade_batch
from ..runtime import Context, log, stage
from ..stats import cluster_bootstrap
from .common import parse_answer, stage_table
from .systems import final_line


def _grade(llm, tok, spec, items: list[dict]) -> list[str]:
    return grade_batch(llm, tok, spec, items)


@stage("x3", "grade", needs=("alexandria", "openscholar", "paperqa"), resources="gpu", env="llm", config_keys=("x3.grade",),
       help="one answer table for all systems; exact match for labels / letters, SimpleQA judge for short answers")
def run_grade(ctx: Context) -> dict:
    items = stage_table(ctx, "items", "items.parquet")
    items["options"] = [list(o) if o is not None else [] for o in items["options"]]
    by = {r["item_id"]: r for r in items.to_dict("records")}
    frames, missing = [], []
    for st in ("alexandria", "openscholar", "paperqa"):
        p = ctx.cfg.stage_dir("x3", st) / "answers.parquet"
        if not p.exists():
            missing.append(st)
            continue
        d = pd.read_parquet(p)
        if st == "paperqa":
            d["system"] = "PaperQA2"
            d["answer"] = [parse_answer(final_line(r), by[i]) or parse_answer(r, by[i]) for r, i in zip(d["raw"], d["item_id"], strict=True)]
            d["gold_retrieved"] = [(by[i].get("gold_paper_key") in json.loads(p_)) if by[i].get("gold_paper_key") else None
                                   for i, p_ in zip(d["item_id"], d["papers"], strict=True)]
        frames.append(d[["bench", "item_id", "system", "answer", "gold_retrieved"]])
    df = pd.concat(frames, ignore_index=True)
    df["gold"] = df["item_id"].map(lambda i: by[i]["answer"])
    df["answer_type"] = df["item_id"].map(lambda i: by[i]["answer_type"])
    df["cluster"] = df["item_id"].map(lambda i: by[i]["cluster"])
    df["abstain_option"] = df["item_id"].map(lambda i: by[i].get("abstain_option"))
    exact = df["answer_type"] != "short"
    df["correct"] = 0
    df.loc[exact, "correct"] = (df.loc[exact, "answer"].fillna("").str.strip().str.lower()
                                == df.loc[exact, "gold"].str.strip().str.lower()).astype(int)
    short = df.index[~exact]
    if len(short):
        judge = model_spec(ctx.cfg, ctx.section("x3.grade").get("judge", "qwen3-8b"))
        log(f"grade: {len(short)} short answers with {judge['name']}")
        res = with_model(judge, _grade, [{"question": by[df.at[i, "item_id"]]["question"], "gold": df.at[i, "gold"],
                                          "predicted": df.at[i, "answer"] or ""} for i in short], gpu_memory_utilization=0.85)
        df.loc[short, "correct"] = [int(g == "CORRECT") for g in res]
    df["answered"] = [int(bool(a) and a != ab) for a, ab in zip(df["answer"].fillna(""), df["abstain_option"], strict=True)]
    df.to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("X3 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"},
        {"label": "systems", "value": int(df["system"].nunique()), "fmt": "int"},
        {"label": "system stages missing", "value": ", ".join(missing) or "none"}]}], summary=f"{len(df)} answers")
    return {"graded": len(df), "missing": missing}


PARETO_SYSTEMS = ("Alexandria A0", "Alexandria R", "Alexandria S", "PaperQA2", "OpenScholar-8B LM + alexandria")
COST_COLS = ("retrieved_tokens", "tokens_in", "tokens_out", "gpu_seconds")


def pareto_front(df: pd.DataFrame, cost: str, acc: str) -> list[bool]:
    """True for rows no other row dominates (cost <= and accuracy >=, one of them strictly)."""
    c, a = df[cost].to_numpy(float), df[acc].to_numpy(float)
    return [not any((c[j] <= c[i] and a[j] >= a[i]) and (c[j] < c[i] or a[j] > a[i]) for j in range(len(c)) if j != i)
            for i in range(len(c))]


def _stage_seconds(ctx: Context, st: str) -> float | None:
    p = ctx.cfg.stage_dir("x3", st) / "manifest.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return float(m["seconds"]) if m.get("status") == "done" and m.get("seconds") else None


def cost_table(ctx: Context) -> pd.DataFrame:
    """Mean cost per question for every system, from the answer tables the system stages write. Retrieved tokens of
    the R arm (passages put in the prompt, not tool results) = its prompt tokens minus the closed-book (A0) prompt of
    the same item. Wall clock = the stage's wall time (model load, indexing, retrieval, generation) split over its
    systems by GPU seconds and divided by the questions."""
    frames = []
    for st in ("alexandria", "openscholar", "paperqa"):
        p = ctx.cfg.stage_dir("x3", st) / "answers.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if st == "paperqa":
            d["system"] = "PaperQA2"
        for c in COST_COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        if st == "alexandria":
            a0 = d[d["system"] == "Alexandria A0"].set_index("item_id")["tokens_in"]
            r = d["system"] == "Alexandria R"
            d.loc[r, "retrieved_tokens"] = (d.loc[r, "tokens_in"] - d.loc[r, "item_id"].map(a0)).clip(lower=0)
        sec = _stage_seconds(ctx, st)
        tot = d["gpu_seconds"].sum()
        d["wall_seconds"] = (sec * d["gpu_seconds"] / tot) if (sec and tot > 0) else (sec / len(d) if sec else np.nan)
        frames.append(d[["system", "item_id", *COST_COLS, "wall_seconds"]])
    if not frames:
        return pd.DataFrame(columns=["system", *COST_COLS, "wall_seconds", "total_tokens"])
    c = pd.concat(frames, ignore_index=True)
    out = c.groupby("system")[[*COST_COLS, "wall_seconds"]].mean().reset_index()
    out["total_tokens"] = out["tokens_in"] + out["tokens_out"]
    return out


def pareto(ctx: Context, wide: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """Accuracy (macro mean over the benchmarks every compared system ran) vs cost; Pareto front; figure."""
    from ..figures import plt, save, style
    sys_ = [s for s in PARETO_SYSTEMS if s in wide.index]
    if not sys_:
        return pd.DataFrame(), None
    benches = [b for b in wide.columns if wide.loc[sys_, b].notna().all()]
    acc = wide.loc[sys_, benches].mean(axis=1).rename("accuracy").reset_index()
    t = acc.merge(cost_table(ctx), on="system", how="left")
    ok = t["total_tokens"].notna() & (t["total_tokens"] > 0)
    t["pareto"] = False
    t.loc[ok, "pareto"] = pareto_front(t[ok], "total_tokens", "accuracy")
    t["benchmarks"] = ", ".join(benches)
    t.to_csv(ctx.out / "pareto.csv", index=False)
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = t[ok].sort_values("total_tokens")
    for r in d.itertuples():
        ax.scatter(r.total_tokens, r.accuracy, s=46, marker="o" if r.pareto else "x", color="C0" if r.pareto else "C3", zorder=3)
        ax.annotate(r.system.replace("OpenScholar-8B LM + alexandria", "OpenScholar-8B + Alex."), (r.total_tokens, r.accuracy),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    f = d[d["pareto"]]
    ax.step(f["total_tokens"], f["accuracy"], where="post", color="C0", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("tokens per question (prompt + generated, log)")
    ax.set_ylabel(f"accuracy (mean of {len(benches)} benchmarks)")
    figs = save(fig, ctx.out / "figures", "x3_pareto")
    return t, figs


@stage("x3", "results", needs=("grade",), config_keys=("x3.results",),
       help="accuracy per benchmark x system with bootstrap CIs; LitQA2 precision; Alexandria vs OpenScholar / PaperQA2")
def run_results(ctx: Context) -> dict:
    g = pd.read_parquet(ctx.cfg.stage_dir("x3", "grade") / "graded.parquet")
    g["gold_retrieved"] = pd.to_numeric(g["gold_retrieved"], errors="coerce")
    rows = []
    for bench, b in g.groupby("bench"):
        tbl, _ = cluster_bootstrap(b, "correct", "cluster", ["system"], n_boot=int(ctx.section("x3.results").get("n_boot", 2000)))
        extra = b.groupby("system").agg(answered=("answered", "mean"), gold=("gold_retrieved", "mean"))
        prec = b[b["answered"] == 1].groupby("system")["correct"].mean().rename("precision")
        tbl = tbl.merge(extra.reset_index(), on="system").merge(prec.reset_index(), on="system", how="left")
        tbl["bench"] = bench
        rows.append(tbl)
    acc = pd.concat(rows, ignore_index=True)
    acc.to_csv(ctx.out / "accuracy.csv", index=False)
    wide = acc.pivot_table(index="system", columns="bench", values="mean")
    wide.to_csv(ctx.out / "accuracy_wide.csv")
    kp = []
    for sysname in ("Alexandria R", "Alexandria S", "PaperQA2", "OpenScholar-8B LM + alexandria"):
        if sysname in wide.index:
            kp.append({"label": f"{sysname} (mean over benchmarks)", "value": float(wide.loc[sysname].mean()), "fmt": "pct1"})
    pt, pfig = pareto(ctx, wide)
    cost_panels = []
    if len(pt):
        front = ", ".join(pt.loc[pt["pareto"], "system"])
        kp.append({"label": "Pareto-optimal (accuracy vs tokens)", "value": front or "n/a"})
        cost_panels = [
            *([{"type": "image", "title": "Accuracy vs tokens per question (o = Pareto-optimal, x = dominated)",
                "src": pfig["png"], "alt": "accuracy vs cost Pareto plot"}] if pfig else []),
            {"type": "table", "title": f"Accuracy vs cost per question (accuracy: mean over {pt['benchmarks'].iloc[0]})",
             "rows": pt.sort_values("total_tokens").assign(pareto=lambda d: d["pareto"].map({True: "yes", False: ""})).to_dict("records"),
             "hint": "R arm retrieved tokens = its prompt minus the closed-book prompt; PaperQA2 tokens include its "
                     "evidence-summary calls; wall clock = stage time incl. model load / indexing, per question",
             "columns": [{"key": "system", "label": "system"}, {"key": "accuracy", "label": "accuracy", "fmt": "pct1"},
                         {"key": "retrieved_tokens", "label": "retrieved tokens", "fmt": "int"},
                         {"key": "tokens_in", "label": "prompt tokens", "fmt": "int"},
                         {"key": "tokens_out", "label": "generated tokens", "fmt": "int"},
                         {"key": "total_tokens", "label": "total tokens", "fmt": "int"},
                         {"key": "gpu_seconds", "label": "GPU s", "fmt": "float"},
                         {"key": "wall_seconds", "label": "wall s", "fmt": "float"},
                         {"key": "pareto", "label": "Pareto-optimal"}]}]
    ctx.publish("X3 results: Alexandria vs OpenScholar and PaperQA2", [
        {"type": "kpis", "headline": True, "items": kp},
        {"type": "table", "title": "Accuracy per benchmark and system (95% CI)", "rows": acc.sort_values(["bench", "mean"], ascending=[True, False]).to_dict("records"),
         "columns": [{"key": "bench", "label": "benchmark"}, {"key": "system", "label": "system"},
                     {"key": "mean", "label": "accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "precision", "label": "precision (answered)", "fmt": "pct1"},
                     {"key": "answered", "label": "answered", "fmt": "pct1"}, {"key": "gold", "label": "gold paper retrieved", "fmt": "pct1"}]},
        *cost_panels,
        {"type": "markdown", "text": "OpenScholar's 45M-paper datastore (0.74 TB) is not available offline: its 8B LM is fed "
                                     "Alexandria retrieval ('+ alexandria'), the benchmark's gold context ('+ gold', upper "
                                     "bound) or nothing ('+ none'). PaperQA2 reads the full texts of the WS3 corpus papers; "
                                     "PaperQA2 and Alexandria share the answer model. Every component is open-source and runs locally "
                                     "(no OpenAI or other external API): LLMs served by vLLM, all-MiniLM-L6-v2 / bge-m3 embeddings."},
    ], summary=f"{g['system'].nunique()} systems x {g['bench'].nunique()} benchmarks")
    return {"rows": len(acc)}
