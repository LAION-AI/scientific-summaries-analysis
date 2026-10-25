"""Stage `results` (X6, CPU): token & compute efficiency tables, iso-accuracy comparisons, the Pareto figure.

Inference (per model and arm): accuracy (WS2 grades for the WS2 arms, X6 grades for ft@N; 95% paper-clustered
bootstrap), mean prompt tokens, accuracy per 1k prompt tokens, GPU-seconds / $ / Wh per 1,000 correct answers (batched
wall time of the arm on one MIG slice; 95% item bootstrap of the ratio), FLOPs per query (2 N tokens), KV-cache MB per
request, p50 TTFT / TPOT / e2e latency at batch 1. Iso-accuracy: the full-text prompt length (interpolated on the
ft@N curve) needed to match the clean KU's accuracy. RAG: accuracy per retrieved-token budget, KU units vs full-text
chunks, and the budget KU units need to match the best full-text-chunk accuracy. Training: probe accuracy against
training tokens and GPU-seconds, and the tokens summary / full-text documents need to match KU documents at 1x.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..figures import color_for, save, style
from ..runtime import Context, log, stage
from ..stats import cluster_bootstrap, paired_difference
from .common import cost, flops, iso_tokens, ratio_ci, stage_dir

WS2_ARMS = ["closed_book", "full_text", "full_text_matched", "ku_raw", "ku_clean", "prose_local", "ku_local"]


def _read_dir(d) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))] if d.exists() else []
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def inference_table(ctx: Context, n_boot: int, ccfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = stage_dir(ctx, "x6", "inference")
    ans, thr, lat = _read_dir(d / "answers"), _read_dir(d / "throughput"), _read_dir(d / "latency")
    if ans.empty or thr.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    arch = {p.stem: json.loads(p.read_text()) for p in (d / "arch").glob("*.json")}
    g = pd.read_parquet(stage_dir(ctx, "ws2", "grade") / "graded.parquet")
    g = g[(g["seed"] == 0) & g["model"].isin(set(ans["model"]))][["qid", "arm", "model", "correct"]]
    g = g.drop_duplicates(["qid", "arm", "model"], keep="last")
    base = ans[~ans["arm"].str.startswith("ft@")].drop(columns=["correct"], errors="ignore").merge(
        g, on=["qid", "arm", "model"], how="inner")
    ft = ans[ans["arm"].str.startswith("ft@")]
    acc = pd.concat([base, ft[base.columns.intersection(ft.columns)]], ignore_index=True)
    acc = acc.merge(thr[["model", "arm", "n", "seconds"]], on=["model", "arm"])
    acc["gpu_s"] = acc["seconds"] / acc["n"]
    tbl, boot = cluster_bootstrap(acc, "correct", "paper_key", ["model", "arm"], n_boot=n_boot)
    rows = []
    for r in tbl.itertuples():
        a = acc[(acc["model"] == r.model) & (acc["arm"] == r.arm)]
        t = thr[(thr["model"] == r.model) & (thr["arm"] == r.arm)].iloc[0]
        l = lat[(lat["model"] == r.model) & (lat["arm"] == r.arm)] if len(lat) else pd.DataFrame()
        ar = arch.get(r.model, {})
        per1k, lo, hi = (v * 1000 for v in ratio_ci(a["gpu_s"].to_numpy(), a["correct"].to_numpy(), n_boot))
        c = cost(per1k, ccfg) if np.isfinite(per1k) else {"usd": np.nan, "wh": np.nan}
        tok = a["prompt_tokens"] + a["output_tokens"]
        rows.append({"model": r.model, "arm": r.arm, "n": r.n, "accuracy": r.mean, "lo": r.lo, "hi": r.hi,
                     "context_tokens": float(a["context_tokens"].mean()), "prompt_tokens": float(a["prompt_tokens"].mean()),
                     "output_tokens": float(a["output_tokens"].mean()),
                     "acc_per_1k_tokens": r.mean / max(a["prompt_tokens"].mean() / 1000, 1e-9),
                     "gpu_s_per_1k_correct": per1k, "gpu_s_lo": lo, "gpu_s_hi": hi,
                     "usd_per_1k_correct": c["usd"], "wh_per_1k_correct": c["wh"],
                     "tflops_per_query": flops(ar.get("params", 0), float(tok.mean())) / 1e12,
                     "kv_mb_per_request": ar.get("kv_bytes_per_token", 0) * float(tok.mean()) / 1e6,
                     "queries_per_s": float(t["n"] / t["seconds"]),
                     "prompt_tok_per_s": float(t["prompt_tokens"] / t["seconds"]),
                     "power_w": t.get("power_w"),
                     "ttft_ms_p50": float(l["ttft_s"].median() * 1000) if len(l) else None,
                     "tpot_ms_p50": float(l["tpot_s"].median() * 1000) if len(l) and l["tpot_s"].notna().any() else None,
                     "e2e_ms_p50": float(l["e2e_s"].median() * 1000) if len(l) else None})
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(WS2_ARMS)}
    out["_o"] = [order.get(a, 100 + int(a.split("@")[1]) / 1e5 if a.startswith("ft@") else 200) for a in out["arm"]]
    out = out.sort_values(["model", "_o"]).drop(columns="_o").reset_index(drop=True)
    return out, acc, boot


def iso_inference(tbl: pd.DataFrame, boot: dict, targets: list[float]) -> tuple[list[dict], list[dict]]:
    """Per model: ku_clean vs full_text (paired), and the full-text prompt length matching the clean KU."""
    iso, share = [], []
    for m, t in tbl.groupby("model"):
        tt = t.set_index("arm")
        if not {"ku_clean", "full_text"} <= set(tt.index):
            continue
        curve = t[t["arm"].str.startswith("ft@") | (t["arm"] == "full_text")].rename(
            columns={"context_tokens": "tokens", "accuracy": "acc"})[["tokens", "acc"]]
        ku, full = tt.loc["ku_clean"], tt.loc["full_text"]
        need = iso_tokens(curve, float(ku["accuracy"]))
        d = paired_difference(boot, (m, "ku_clean"), (m, "full_text"), float(ku["accuracy"]), float(full["accuracy"]))
        iso.append({"model": m, "ku_accuracy": float(ku["accuracy"]), "full_accuracy": float(full["accuracy"]),
                    "diff": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"], "p": d["p"],
                    "ku_tokens": float(ku["context_tokens"]), "full_tokens": float(full["context_tokens"]),
                    "token_ratio": float(full["prompt_tokens"] / ku["prompt_tokens"]),
                    "gpu_ratio": float(full["gpu_s_per_1k_correct"] / ku["gpu_s_per_1k_correct"]),
                    "ft_tokens_to_match_ku": need,
                    "iso_ratio": need / float(ku["context_tokens"]) if need else None})
        for s in targets:
            target = s * float(full["accuracy"])
            share.append({"model": m, "target": f"{s:.0%} of full-text accuracy", "accuracy": target,
                          "full_text_tokens_needed": iso_tokens(curve, target),
                          "ku_clean_tokens": float(ku["context_tokens"]),
                          "ku_clean_reaches": bool(float(ku["accuracy"]) >= target)})
    return iso, share


def rag_table(ctx: Context, n_boot: int, ccfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    df = _read_dir(stage_dir(ctx, "x6", "rag-budget") / "parts")
    if df.empty:
        return pd.DataFrame(), []
    df["gpu_s"] = df["batch_seconds"] / df["batch_n"]
    tbl, _ = cluster_bootstrap(df, "correct", "item_id", ["model", "kind", "budget"], n_boot=n_boot)
    rows = []
    for r in tbl.itertuples():
        a = df[(df["model"] == r.model) & (df["kind"] == r.kind) & (df["budget"] == r.budget)]
        per1k = ratio_ci(a["gpu_s"].to_numpy(), a["correct"].to_numpy(), n_boot)[0] * 1000
        rows.append({"model": r.model, "kind": r.kind, "budget": int(r.budget), "n": r.n, "accuracy": r.mean,
                     "lo": r.lo, "hi": r.hi, "tokens": float(a["context_tokens"].mean()),
                     "units": float(a["units"].mean()), "gold_in_context": float(a["gold_in_context"].mean()),
                     "gpu_s_per_1k_correct": per1k, "usd_per_1k_correct": cost(per1k, ccfg)["usd"],
                     "x2_accuracy": float(a[a["source"] == "x2-post-cutoff"]["correct"].mean())
                     if (a["source"] == "x2-post-cutoff").any() else None})
    out = pd.DataFrame(rows).sort_values(["model", "kind", "budget"]).reset_index(drop=True)
    iso = []
    for m, t in out.groupby("model"):
        s, f = t[t["kind"] == "summary"], t[t["kind"] == "fulltext"]
        if s.empty or f.empty:
            continue
        best = f.sort_values("accuracy").iloc[-1]
        need = iso_tokens(s.rename(columns={"accuracy": "acc"})[["tokens", "acc"]], float(best["accuracy"]))
        iso.append({"model": m, "best_fulltext_accuracy": float(best["accuracy"]),
                    "best_fulltext_budget": int(best["budget"]), "ku_tokens_to_match": need,
                    "token_ratio": float(best["tokens"]) / need if need else None,
                    "ku_best_accuracy": float(s["accuracy"].max())})
    return out, iso


def train_table(ctx: Context, n_boot: int, ccfg: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    p = stage_dir(ctx, "x6", "train-efficiency") / "probe.parquet"
    if not p.exists():
        return pd.DataFrame(), [], {}
    df = pd.read_parquet(p)
    runs = {}
    for f in (stage_dir(ctx, "x6", "train-efficiency") / "runs").glob("*/train.json"):
        runs[f.parent.name] = json.loads(f.read_text())
    tbl, _ = cluster_bootstrap(df, "correct", "paper_key", ["condition", "fraction", "split"], n_boot=n_boot)
    meta = df.groupby(["condition", "fraction"])[["train_tokens", "train_seconds"]].first()
    tbl["train_tokens"] = [int(meta.loc[(c, f), "train_tokens"]) for c, f in zip(tbl["condition"], tbl["fraction"], strict=True)]
    tbl["gpu_s"] = [float(meta.loc[(c, f), "train_seconds"]) for c, f in zip(tbl["condition"], tbl["fraction"], strict=True)]
    tbl["usd"] = [cost(s, ccfg)["usd"] if np.isfinite(s) else np.nan for s in tbl["gpu_s"]]
    model = str(df["model"].iloc[0]) if "model" in df else ""
    params = float(ctx.cfg.raw.get("models", {}).get(model, {}).get("size_b", 0)) * 1e9
    tbl["train_pflops"] = [6 * params * t / 1e15 for t in tbl["train_tokens"]]
    base = tbl[tbl["condition"] == "base"]
    iso = []
    for split in ("train", "heldout"):
        ku1 = tbl[(tbl["condition"] == "ku") & (tbl["fraction"] == 1.0) & (tbl["split"] == split)]
        if ku1.empty:
            continue
        target = float(ku1["mean"].iloc[0])
        for cond in ("ku", "summary", "full"):
            c = tbl[(tbl["condition"] == cond) & (tbl["split"] == split)]
            if c.empty:
                continue
            b = base[base["split"] == split]
            curve = pd.concat([pd.DataFrame({"tokens": [0], "acc": b["mean"].tolist()[:1] or [0.0]}),
                               c.rename(columns={"train_tokens": "tokens", "mean": "acc"})[["tokens", "acc"]]])
            need = iso_tokens(curve, target)
            signal = bool(len(b)) and target > float(b["hi"].iloc[0])     # KU 1x above the base model's 95% CI
            iso.append({"split": split, "condition": cond, "ku_1x_accuracy": target,
                        "base_accuracy": float(b["mean"].iloc[0]) if len(b) else None, "learning_signal": signal,
                        "tokens_to_match": need if signal else None,
                        "x_of_ku": need / float(ku1["train_tokens"].iloc[0]) if signal and need is not None else None,
                        "best_accuracy": float(c["mean"].max())})
    return tbl.rename(columns={"mean": "accuracy"}), iso, {"model": model, "runs": runs}


def figure(ctx: Context, inf: pd.DataFrame, rag: pd.DataFrame, tr: pd.DataFrame) -> dict:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    models = sorted(inf["model"].unique()) if len(inf) else []
    ax = axes[0]
    for m in models:
        t = inf[inf["model"] == m]
        col = color_for(m, models)
        curve = t[t["arm"].str.startswith("ft@") | (t["arm"] == "full_text")].sort_values("prompt_tokens")
        ax.plot(curve["prompt_tokens"], curve["accuracy"], "-", lw=1, color=col, label=f"{m}: full text cut to N")
        for arm, mk in (("ku_clean", "*"), ("ku_raw", "^"), ("full_text_matched", "s"), ("closed_book", "x"),
                        ("prose_local", "D"), ("ku_local", "P")):
            r = t[t["arm"] == arm]
            if len(r):
                ax.plot(r["prompt_tokens"], r["accuracy"], mk, color=col, ms=6 if mk == "*" else 4)
                if m == models[0]:
                    ax.annotate(arm, (float(r["prompt_tokens"].iloc[0]), float(r["accuracy"].iloc[0])), fontsize=6,
                                xytext=(3, -8), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("prompt tokens per question (log)")
    ax.set_ylabel("accuracy (judge CORRECT)")
    ax.set_title("WS2 bench: accuracy vs tokens")
    if len(models) > 1:
        ax.legend(fontsize=6)
    ax = axes[1]
    if len(rag):
        rm = sorted(rag["model"].unique())
        for m in rm:
            for kind, ls in (("summary", "-"), ("fulltext", "--")):
                t = rag[(rag["model"] == m) & (rag["kind"] == kind)].sort_values("budget")
                ax.plot(t["tokens"], t["accuracy"], ls, marker="o", ms=3, lw=1, color=color_for(m, rm),
                        label=f"{m}: {'KU units' if kind == 'summary' else 'full-text chunks'}")
        ax.set_xscale("log")
        ax.legend(fontsize=6)
    ax.set_xlabel("retrieved context tokens (log)")
    ax.set_title("RAG: accuracy vs token budget")
    ax = axes[2]
    if len(tr):
        conds = ["ku", "summary", "full"]
        for cond in conds:
            t = tr[(tr["condition"].isin([cond, "base"])) & (tr["split"] == "train")].sort_values("train_tokens")
            ax.plot(t["train_tokens"], t["accuracy"], "-o", ms=3, lw=1, color=color_for(cond, conds), label=cond)
        ax.legend(fontsize=6)
    ax.set_xlabel("training tokens")
    ax.set_title("Knowledge probe (trained papers)")
    fig.tight_layout()
    return save(fig, ctx.out / "figures", "x6_efficiency")


def _cols(*spec) -> list[dict]:
    return [{"key": k, "label": lab, **({"fmt": f} if f else {})} for k, lab, f in spec]


@stage("x6", "results", config_keys=("x6.results", "x6.cost"),
       help="efficiency tables (accuracy per token, GPU-s / $ / Wh per 1,000 correct, latency, iso-accuracy), "
            "RAG budget and training curves, storage, Pareto figure")
def run(ctx: Context) -> dict:
    cfg, ccfg = ctx.section("x6.results"), ctx.section("x6.cost")
    n_boot = int(cfg.get("n_boot", 2000))
    inf, _, boot = inference_table(ctx, n_boot, ccfg)
    iso, share = iso_inference(inf, boot, [float(t) for t in cfg.get("iso_targets", [0.8, 0.9, 1.0])]) if len(inf) else ([], [])
    rag, rag_iso = rag_table(ctx, n_boot, ccfg)
    tr, tr_iso, tr_meta = train_table(ctx, n_boot, ccfg)
    sp = stage_dir(ctx, "x6", "storage") / "storage.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    if inf.empty and rag.empty and tr.empty:
        return ctx.skip("X6 results", "no X6 measurements yet (run inference / rag-budget / train-efficiency)")
    for name, t in (("inference", inf), ("iso_inference", pd.DataFrame(iso)), ("iso_share", pd.DataFrame(share)),
                    ("rag", rag), ("rag_iso", pd.DataFrame(rag_iso)), ("train", tr), ("train_iso", pd.DataFrame(tr_iso))):
        t.to_csv(ctx.out / f"{name}.csv", index=False)
    fig = figure(ctx, inf, rag, tr)

    def mean_of(rows: list[dict], key: str) -> float | None:
        v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    kpis = []
    if iso:
        kpis += [{"label": "prompt tokens: full text / clean KU", "value": f"{mean_of(iso, 'token_ratio'):.1f}x",
                  "hint": "mean over models; WS2 bench, same prompt"},
                 {"label": "accuracy: clean KU − full text", "value": mean_of(iso, "diff"), "fmt": "pct1",
                  "hint": "; ".join(f"{r['model']}: {r['diff']:+.1%} [{r['diff_lo']:+.1%}, {r['diff_hi']:+.1%}]" for r in iso)},
                 {"label": "GPU-s per 1,000 correct: full text / clean KU", "value": f"{mean_of(iso, 'gpu_ratio'):.1f}x",
                  "hint": "batched, one 2g.20gb MIG slice"}]
        ku_rows = inf[inf["arm"] == "ku_clean"]
        ft_rows = inf[inf["arm"] == "full_text"]
        kpis.append({"label": "$ per 1,000 correct: clean KU vs full text",
                     "value": f"${ku_rows['usd_per_1k_correct'].mean():.3f} vs ${ft_rows['usd_per_1k_correct'].mean():.3f}",
                     "hint": f"{ccfg.get('usd_per_gpu_hour', 2.0)} $/A100-hour x slice share"})
        m = [r for r in iso if r["iso_ratio"]]
        kpis.append({"label": "full-text prefix length to match clean-KU accuracy",
                     "value": f"{np.mean([r['iso_ratio'] for r in m]):.1f}x the KU's tokens" if m else "not reached",
                     "hint": "interpolated on the full-text-cut-to-N curve (ft@N)"})
    if rag_iso:
        r1 = rag[rag["budget"] == 1000].groupby("kind")["accuracy"].mean()
        if {"summary", "fulltext"} <= set(r1.index):
            kpis.append({"label": "RAG at 1k tokens: KU units vs full-text chunks",
                         "value": f"{r1['summary']:.1%} vs {r1['fulltext']:.1%}"})
        m = [r for r in rag_iso if r["token_ratio"]]
        if m:
            kpis.append({"label": "RAG: tokens to match best full-text-chunk accuracy",
                         "value": f"{np.mean([r['token_ratio'] for r in m]):.1f}x fewer with KU units"})
    full_tr = [r for r in tr_iso if r["split"] == "train" and r["condition"] == "full"]
    if full_tr:
        r = full_tr[0]
        kpis.append({"label": "training tokens for full text to match KU (1x)",
                     "value": ("no learning signal (KU 1x within base CI)" if not r["learning_signal"] else
                               f"{r['x_of_ku']:.2f}x" if r["x_of_ku"] is not None else "not reached within 2x"),
                     "hint": f"knowledge probe, trained papers; {tr_meta.get('model')} LoRA, profile {ctx.cfg.profile}"})
    for r in st.get("ratios", []):
        if r["rep"] == "ku_json":
            kpis.append({"label": "storage: KU JSON / full text", "value": r["byte_ratio_vs_full"], "fmt": "pct1",
                         "hint": "median bytes per paper"})
    for k in kpis:
        k["headline"] = True

    panels = [
        {"type": "kpis", "headline": True, "items": kpis},
        {"type": "image", "title": "Accuracy vs tokens: inference, RAG budget, training", "src": fig["png"],
         "alt": "three panels: accuracy against prompt tokens, retrieved tokens and training tokens"},
        {"type": "table", "title": "Inference efficiency per model and arm",
         "hint": "accuracy: 95% paper-clustered bootstrap; GPU-s per 1,000 correct: 95% item bootstrap; "
                 "ft@N = full text cut to N tokens",
         "columns": _cols(("model", "model", None), ("arm", "arm", None), ("accuracy", "accuracy", "pct1"),
                          ("lo", "lo", "pct1"), ("hi", "hi", "pct1"), ("prompt_tokens", "prompt tokens", "int"),
                          ("acc_per_1k_tokens", "acc / 1k tokens", "float3"),
                          ("gpu_s_per_1k_correct", "GPU-s / 1k correct", "float"), ("gpu_s_lo", "lo", "float"),
                          ("gpu_s_hi", "hi", "float"), ("usd_per_1k_correct", "$ / 1k correct", "float3"),
                          ("wh_per_1k_correct", "Wh / 1k correct", "float"),
                          ("tflops_per_query", "TFLOPs / query", "float"), ("kv_mb_per_request", "KV MB / request", "float"),
                          ("ttft_ms_p50", "TTFT ms p50", "float"), ("tpot_ms_p50", "TPOT ms p50", "float"),
                          ("e2e_ms_p50", "e2e ms p50", "float"), ("queries_per_s", "q/s batched", "float")),
         "rows": inf.to_dict("records"), "limit": 100},
        {"type": "table", "title": "Clean KU vs full text (iso-accuracy)",
         "hint": "diff: paired paper-clustered bootstrap; ft tokens to match: full-text length at which the "
                 "ft@N curve reaches the clean KU's accuracy",
         "columns": _cols(("model", "model", None), ("ku_accuracy", "KU acc", "pct1"), ("full_accuracy", "full acc", "pct1"),
                          ("diff", "Δ", "pct1"), ("diff_lo", "lo", "pct1"), ("diff_hi", "hi", "pct1"),
                          ("ku_tokens", "KU tokens", "int"), ("full_tokens", "full tokens", "int"),
                          ("token_ratio", "token ratio", "float"), ("gpu_ratio", "GPU ratio", "float"),
                          ("ft_tokens_to_match_ku", "ft tokens to match KU", "int"), ("iso_ratio", "x KU tokens", "float")),
         "rows": iso},
        {"type": "table", "title": "Tokens needed for a share of full-text accuracy",
         "columns": _cols(("model", "model", None), ("target", "target", None), ("accuracy", "accuracy", "pct1"),
                          ("full_text_tokens_needed", "full-text tokens", "int"), ("ku_clean_tokens", "KU tokens", "int"),
                          ("ku_clean_reaches", "KU reaches", None)), "rows": share},
    ]
    if len(rag):
        panels += [
            {"type": "table", "title": "RAG: accuracy per retrieved-token budget",
             "hint": "summary = KU units (fields + claims), fulltext = source chunks; 95% item bootstrap",
             "columns": _cols(("model", "model", None), ("kind", "units", None), ("budget", "budget", "int"),
                              ("tokens", "mean tokens", "int"), ("accuracy", "accuracy", "pct1"), ("lo", "lo", "pct1"),
                              ("hi", "hi", "pct1"), ("gold_in_context", "gold in context", "pct1"),
                              ("x2_accuracy", "X2 post-cutoff acc", "pct1"),
                              ("gpu_s_per_1k_correct", "GPU-s / 1k correct", "float")),
             "rows": rag.to_dict("records"), "limit": 100},
            {"type": "table", "title": "RAG iso-accuracy: KU-unit tokens to match the best full-text-chunk accuracy",
             "columns": _cols(("model", "model", None), ("best_fulltext_accuracy", "best chunks acc", "pct1"),
                              ("best_fulltext_budget", "at budget", "int"), ("ku_tokens_to_match", "KU tokens", "int"),
                              ("token_ratio", "x fewer", "float"), ("ku_best_accuracy", "best KU acc", "pct1")),
             "rows": rag_iso}]
    if len(tr):
        panels += [
            {"type": "table", "title": f"Training efficiency ({tr_meta.get('model')} LoRA; profile {ctx.cfg.profile})",
             "hint": "knowledge probe closed book; train = questions on trained papers, heldout = unseen papers; "
                     "constant LR after warmup, checkpoints at each budget",
             "columns": _cols(("condition", "docs", None), ("fraction", "x budget", "float"), ("split", "split", None),
                              ("train_tokens", "tokens", "int"), ("gpu_s", "GPU-s", "float"),
                              ("train_pflops", "PFLOPs", "float"), ("accuracy", "accuracy", "pct1"),
                              ("lo", "lo", "pct1"), ("hi", "hi", "pct1")), "rows": tr.to_dict("records"), "limit": 100},
            {"type": "table", "title": "Training tokens to match KU documents at 1x",
             "columns": _cols(("split", "split", None), ("condition", "docs", None), ("ku_1x_accuracy", "KU 1x acc", "pct1"),
                              ("base_accuracy", "base acc", "pct1"), ("learning_signal", "signal", None),
                          ("tokens_to_match", "tokens", "int"), ("x_of_ku", "x KU budget", "float"),
                              ("best_accuracy", "best acc", "pct1")), "rows": tr_iso}]
    if st:
        panels += [
            {"type": "table", "title": "Storage per paper (median)",
             "columns": _cols(("rep", "representation", None), ("papers", "papers", "int"), ("bytes", "bytes", "int"),
                              ("gzip_bytes", "gzip bytes", "int"), ("tokens", "tokens", "int")), "rows": st.get("per_paper", [])},
            {"type": "table", "title": "Index size per paper",
             "columns": _cols(("index", "index", None), ("rep", "units", None), ("units_per_paper", "units / paper", "float"),
                              ("bytes_per_paper", "bytes / paper (flat + BM25)", "int"),
                              ("pq_bytes_per_paper", "bytes / paper (PQ48 + BM25)", "int"),
                              ("encode_s_per_paper", "encode s / paper", "float3")), "rows": st.get("index", [])}]
        if st.get("x1_latency"):
            panels.append({"type": "table", "title": "Query latency per representation (from X1)",
                           "columns": _cols(("rep", "units", None), ("retriever", "retriever", None),
                                            ("units", "units", "int"), ("p50_ms", "p50 ms", "float"),
                                            ("p95_ms", "p95 ms", "float"), ("qps_batch", "q/s batched", "float")),
                           "rows": st["x1_latency"]})
    panels.append({"type": "markdown", "text":
                   "**Method.** Cost = batched wall time of each arm on one 2g.20gb MIG slice (2/7 A100) × "
                   f"{ccfg.get('usd_per_gpu_hour', 2.0)} $/A100-hour × slice share; energy = GPU-s × "
                   f"{ccfg.get('gpu_tdp_watts', 400)} W TDP × slice share (estimate: MIG slices do not report their own "
                   "power; nvidia-smi gives the whole GPU, shown as measured W). FLOPs = 2 N tokens (inference), "
                   "6 N tokens (training). Latency at batch 1 with the prefix cache reset per request; throughput "
                   "batched with prefix caching within an arm (questions of one paper share the context prefix, "
                   "which favours long contexts). fp8 weights on A100 run as weight-only fp8 (Marlin)."})
    ctx.publish("X6 results: token & compute efficiency", panels,
                summary=(f"KU {mean_of(iso, 'token_ratio'):.1f}x fewer prompt tokens, Δacc {mean_of(iso, 'diff'):+.1%}"
                         if iso else "partial results"))
    log(f"results: {len(inf)} inference rows, {len(rag)} rag rows, {len(tr)} training rows")
    return {"inference_rows": len(inf), "rag_rows": len(rag), "train_rows": len(tr), "iso": iso, "rag_iso": rag_iso,
            "train_iso": tr_iso}
