"""Stages `grade` and `results` (WS3).

grade    QA-bench (short answers): SimpleQA judge (qa.GRADER_SYSTEM). Label / multiple-choice benchmarks: exact match
         of the normalised label / option letter. HLE: the official HLE judge prompt (Phan et al., 2025) with a
         structured verdict. Abstentions (LitQA2 "Insufficient information", "I don't know") are recorded.
results  Accuracy per benchmark x model x arm (mean over seeds per item) with 95% bootstrap CIs that resample clusters
         (gold papers for the QA bench, items otherwise), paired differences vs A0 (and vs W when run) with Holm
         correction, help / harm flips vs A0, gold-paper-retrieved rate, tool calls, retrieved tokens, tokens and
         GPU-seconds per question, LitQA2 precision (correct / answered), and the headline figure: delta accuracy
         over the model alone vs model size, one line per arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..figures import plt, save, style
from ..llm import chat_prompt, json_params, model_spec, parse_json, with_model
from ..qa import grade_batch, is_abstention, normalize_answer
from ..runtime import Context, log, stage
from ..stats import cluster_bootstrap, holm, paired_difference
from ..ws2.common import load_parts

HLE_JUDGE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""
HLE_SCHEMA = {"type": "object", "properties": {
    "extracted_final_answer": {"type": "string", "maxLength": 300}, "reasoning": {"type": "string", "maxLength": 800},
    "correct": {"type": "string", "enum": ["yes", "no"]}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["extracted_final_answer", "reasoning", "correct", "confidence"], "additionalProperties": False}


def exact_label(pred: str | None, gold: str, answer_type: str) -> bool:
    if answer_type == "mc":                    # option letter, read raw ("A" would be dropped as an article)
        p = str(pred or "").strip().lstrip("(").upper()
        return p[:1] == str(gold).upper() and (len(p) == 1 or not p[1:2].isalnum())
    return normalize_answer(pred) == normalize_answer(gold)


def _judge(llm, tok, spec, payload: dict) -> dict:
    out = {"simpleqa": grade_batch(llm, tok, spec, payload["simpleqa"]) if payload["simpleqa"] else []}
    if payload["hle"]:
        prompts = [chat_prompt(tok, spec, None, HLE_JUDGE.format(question=x["question"], response=x["predicted"],
                                                                  correct_answer=x["gold"])) for x in payload["hle"]]
        res = llm.generate(prompts, json_params(HLE_SCHEMA, 600), use_tqdm=False)
        out["hle"] = [((parse_json(r.outputs[0].text) or {}).get("correct") == "yes") for r in res]
    else:
        out["hle"] = []
    return out


@stage("ws3", "grade", needs=("agent",), resources="gpu", env="llm", config_keys=("ws3.grade",),
       help="grade agent answers: SimpleQA judge (QA bench), exact label / letter, official HLE judge")
def run_grade(ctx: Context) -> dict:
    cfg = ctx.section("ws3.grade")
    ans = load_parts(ctx, "agent", "ws3")
    if ans.empty:
        raise FileNotFoundError("ws3/agent has no parts; run `agent` first")
    arms = ctx.section("ws3.agent").get("arms")
    if arms:                                   # parts may hold arms dropped from the config since (e.g. K)
        ans = ans[ans["arm"].isin(arms)].reset_index(drop=True)
    items =pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
    df = ans.merge(items[["bench", "item_id", "question", "answer", "answer_type", "judge", "cluster",
                          "abstain_option", "category"]].rename(columns={"answer": "gold"}),
                   on=["bench", "item_id"], how="left")
    df["correct"] = 0
    exact = df["judge"] == "exact"
    df.loc[exact, "correct"] = [int(exact_label(p, g, t)) for p, g, t in
                                zip(df.loc[exact, "answer"], df.loc[exact, "gold"], df.loc[exact, "answer_type"], strict=True)]
    sq, hl = df.index[df["judge"] == "simpleqa"], df.index[df["judge"] == "hle"]
    payload = {"simpleqa": [{"question": df.at[i, "question"], "gold": df.at[i, "gold"],
                             "predicted": df.at[i, "answer"] or ""} for i in sq],
               "hle": [{"question": df.at[i, "question"], "gold": df.at[i, "gold"], "predicted": df.at[i, "answer"] or ""}
                       for i in hl]}
    judge = model_spec(ctx.cfg, cfg.get("judge", "qwen3-8b"))
    if len(sq) or len(hl):
        log(f"grade: {len(sq)} SimpleQA + {len(hl)} HLE answers with {judge['name']}")
        res = with_model(judge, _judge, payload, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
        df.loc[sq, "correct"] = [int(g == "CORRECT") for g in res["simpleqa"]]
        df.loc[hl, "correct"] = [int(c) for c in res["hle"]]
    df["abstained"] = [int(is_abstention(a) or (isinstance(ab, str) and str(a or "").strip()[:1].upper() == ab))
                       for a, ab in zip(df["answer"], df["abstain_option"], strict=True)]
    df["judge_model"] = judge["name"]
    df.drop(columns=["question"]).to_parquet(ctx.out / "graded.parquet", index=False)
    ctx.publish("WS3 grading", [{"type": "kpis", "items": [
        {"label": "answers graded", "value": len(df), "fmt": "int"},
        {"label": "correct", "value": float(df["correct"].mean()), "fmt": "pct1"},
        {"label": "judge (free-form)", "value": judge["name"]}]}], summary=f"{len(df)} answers")
    return {"graded": len(df)}


@stage("ws3", "results", needs=("grade",), config_keys=("ws3.results",),
       help="accuracy per benchmark x arm x model, paired tests vs A0 (Holm), flips, retrieval and cost, headline figure")
def run_results(ctx: Context) -> dict:
    cfg = ctx.section("ws3.results")
    n_boot = int(cfg.get("n_boot", 2000))
    g = pd.read_parquet(ctx.cfg.stage_dir("ws3", "grade") / "graded.parquet")
    per = g.groupby(["bench", "model", "arm", "item_id", "cluster"], as_index=False).agg(
        correct=("correct", "mean"), abstained=("abstained", "mean"), tool_calls=("tool_calls", "mean"),
        retrieved_tokens=("retrieved_tokens", "mean"), tokens=("tokens_in", "mean"), tokens_out=("tokens_out", "mean"),
        gpu_seconds=("gpu_seconds", "mean"), gold_retrieved=("gold_retrieved", "mean"))
    # items whose gold paper is in the corpus: the only ones a retrieval arm can be helped on by finding the paper
    items = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet", columns=["bench", "item_id",
                                                                                                "gold_paper_key"])
    linked = set(items.dropna(subset=["gold_paper_key"])["item_id"])
    per["gold_in_corpus"] = per["item_id"].isin(linked)
    partly = per.groupby("bench")["gold_in_corpus"].agg(lambda x: 0 < x.mean() < 1)   # all-linked (QA bench): no copy
    sub = per[per["gold_in_corpus"] & per["bench"].map(partly)].assign(bench=lambda x: x["bench"] + " [gold in corpus]")
    per = pd.concat([per, sub], ignore_index=True)
    arm_order = [a for a in (ctx.section("ws3.agent").get("arms") or []) if a in set(per["arm"])] or sorted(per["arm"].unique())
    sizes = {m: (ctx.cfg.raw.get("models") or {}).get(m, {}).get("size_b") for m in per["model"].unique()}
    acc_rows, cmp_rows, flip_rows = [], [], []
    for bench, b in per.groupby("bench"):
        tbl, boot = cluster_bootstrap(b, "correct", "cluster", ["model", "arm"], n_boot=n_boot)
        cost = b.groupby(["model", "arm"]).agg(tool_calls=("tool_calls", "mean"), retrieved=("retrieved_tokens", "mean"),
                                               tokens=("tokens", "mean"), gpu_s=("gpu_seconds", "mean"),
                                               gold=("gold_retrieved", "mean"), abstained=("abstained", "mean")).reset_index()
        tbl = tbl.merge(cost, on=["model", "arm"])
        answered = b[b["abstained"] < .5].groupby(["model", "arm"])["correct"].mean().rename("precision").reset_index()
        tbl = tbl.merge(answered, on=["model", "arm"], how="left")
        tbl["bench"] = bench
        acc_rows.append(tbl)
        pt = {(r.model, r.arm): r.mean for r in tbl.itertuples()}
        for (m, a) in pt:
            for ref in ("A0", "W"):
                if a != ref and (m, ref) in pt:
                    cmp_rows.append({"bench": bench, "model": m, "arm": a, "vs": ref,
                                     **paired_difference(boot, (m, a), (m, ref), pt[(m, a)], pt[(m, ref)])})
        base = b[b["arm"] == "A0"].set_index(["model", "item_id"])["correct"]
        for (m, a), s in b[b["arm"] != "A0"].groupby(["model", "arm"]):
            bb = base.reindex(list(zip(s["model"], s["item_id"], strict=True))).to_numpy()
            cc = s["correct"].to_numpy()
            flip_rows.append({"bench": bench, "model": m, "arm": a, "helped": float(np.nanmean((bb < .5) & (cc >= .5))),
                              "harmed": float(np.nanmean((bb >= .5) & (cc < .5)))})
    acc = pd.concat(acc_rows, ignore_index=True)
    cmp = pd.DataFrame(cmp_rows)
    if not cmp.empty:
        cmp["p_holm"] = holm(cmp["p"].tolist())
    flips = pd.DataFrame(flip_rows)
    acc.to_csv(ctx.out / "accuracy.csv", index=False)
    cmp.to_csv(ctx.out / "comparisons.csv", index=False)
    flips.to_csv(ctx.out / "flips.csv", index=False)
    breakdown = gains_by(ctx, per[~per["bench"].str.endswith("[gold in corpus]")])
    for key, tbl_ in breakdown.items():
        tbl_.to_csv(ctx.out / f"gains_by_{key}.csv", index=False)
    main = acc.pivot_table(index=["bench", "arm"], columns="model", values="mean")
    (ctx.out / "main_table.tex").write_text(main.map(lambda v: f"{v:.1%}" if pd.notna(v) else "–").to_latex(escape=True))

    # headline: delta accuracy over the model alone vs model size, one line per arm (pooled over benchmarks)
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    d = cmp[(cmp["vs"] == "A0") & ~cmp["bench"].str.endswith("[gold in corpus]")].groupby(["arm", "model"])["diff"] \
        .mean().reset_index() if not cmp.empty else pd.DataFrame()
    for arm in [a for a in arm_order if a != "A0"]:
        s = d[d["arm"] == arm].assign(size=lambda x: x["model"].map(sizes)).dropna(subset=["size"]).sort_values("size")
        if len(s):
            ax.plot(s["size"], s["diff"], marker="o", label=arm)
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xlabel("model size (B parameters)")
    ax.set_ylabel("Δ accuracy vs model alone (mean over benchmarks)")
    ax.legend(fontsize=7, ncol=2)
    fig_paths = save(fig, ctx.out / "figures", "ws3_headline_delta_vs_size")

    pilot = load_parts(ctx, "pilot", "ws3")
    gate = float(ctx.section("ws3.pilot").get("gate", 0.95))
    rel = [] if pilot.empty else [{"model": m, "well_formed": float(x["well_formed"].mean()), "n": len(x),
                                   "gate": "pass" if x["well_formed"].mean() >= gate else "FAIL"}
                                  for m, x in pilot.groupby("model")]
    pd.DataFrame(rel).to_csv(ctx.out / "pilot_reliability.csv", index=False)
    panels: list[dict] = [{"type": "kpis", "headline": True, "items": [
        {"label": "benchmarks", "value": int(acc["bench"].nunique()), "fmt": "int"},
        {"label": "models", "value": int(acc["model"].nunique()), "fmt": "int"},
        {"label": "arms", "value": int(acc["arm"].nunique()), "fmt": "int"},
        {"label": "episodes", "value": len(g), "fmt": "int"}]},
        {"type": "image", "title": "Headline: Δ accuracy over the model alone vs model size", "src": fig_paths["png"],
         "alt": "delta accuracy vs model size per arm"}]
    for bench, t in acc.groupby("bench"):
        models = sorted(t["model"].unique())
        panels.append({"type": "bar", "title": f"{bench}: accuracy per arm", "fmt": "pct1",
                       "hint": "95% cluster-bootstrap CI", "labels": [a for a in arm_order if a in set(t["arm"])],
                       "series": [{"name": m, **{k: t[t["model"] == m].set_index("arm").reindex(
                           [a for a in arm_order if a in set(t["arm"])])[c].tolist() for k, c in
                           (("data", "mean"), ("lo", "lo"), ("hi", "hi"))}} for m in models]})
    for key, tbl_ in breakdown.items():
        panels.append({"type": "table", "title": f"Gains by {key}: Δ accuracy vs the model alone (A0), all models pooled",
                       "hint": "discipline / language of the item's gold paper (WS1 features); items without a linked paper "
                               "use their benchmark's field; small groups are noisy",
                       "columns": [{"key": key, "label": key}, {"key": "arm", "label": "arm"},
                                   {"key": "items", "label": "items", "fmt": "int"}, {"key": "acc", "label": "accuracy", "fmt": "pct1"},
                                   {"key": "delta_vs_A0", "label": "Δ vs A0", "fmt": "pct1"}],
                       "rows": tbl_.to_dict("records"), "limit": 200})
    panels += [
        {"type": "table", "title": f"Tool-call reliability without constrained decoding (pilot; gate {gate:.0%})",
         "hint": "the agent arms decode JSON actions constrained to the tool schema, so a FAIL model still runs; "
                 "its agent results measure the model with constrained decoding only",
         "columns": [{"key": "model", "label": "model"}, {"key": "well_formed", "label": "well-formed", "fmt": "pct1"},
                     {"key": "n", "label": "calls", "fmt": "int"}, {"key": "gate", "label": "gate"}], "rows": rel},
        {"type": "table", "title": "Main table: accuracy and cost per benchmark, arm and model",
         "columns": [{"key": "bench", "label": "benchmark"}, {"key": "model", "label": "model"},
                     {"key": "arm", "label": "arm"}, {"key": "mean", "label": "accuracy", "fmt": "pct1"},
                     {"key": "lo", "label": "lo", "fmt": "pct1"}, {"key": "hi", "label": "hi", "fmt": "pct1"},
                     {"key": "precision", "label": "precision (answered)", "fmt": "pct1"},
                     {"key": "tool_calls", "label": "tool calls", "fmt": "float"},
                     {"key": "retrieved", "label": "retrieved tokens", "fmt": "int"},
                     {"key": "tokens", "label": "prompt tokens", "fmt": "int"},
                     {"key": "gpu_s", "label": "GPU s / question", "fmt": "float3"},
                     {"key": "gold", "label": "gold paper retrieved", "fmt": "pct1"}],
         "rows": acc.to_dict("records"), "limit": 200},
        {"type": "table", "title": "Paired differences vs the model alone (A0)",
         "hint": "cluster bootstrap; Holm over all rows",
         "columns": [{"key": "bench", "label": "benchmark"}, {"key": "model", "label": "model"},
                     {"key": "arm", "label": "arm"}, {"key": "vs", "label": "vs"},
                     {"key": "diff", "label": "Δ accuracy", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}, {"key": "p_holm", "label": "p (Holm)", "fmt": "float3"}],
         "rows": cmp.to_dict("records"), "limit": 200},
        {"type": "table", "title": "Help / harm flips vs A0",
         "columns": [{"key": "bench", "label": "benchmark"}, {"key": "model", "label": "model"},
                     {"key": "arm", "label": "arm"}, {"key": "helped", "label": "helped", "fmt": "pct1"},
                     {"key": "harmed", "label": "harmed", "fmt": "pct1"}], "rows": flips.to_dict("records"), "limit": 200},
    ]
    ctx.publish("WS3 results: knowledge skill on benchmarks", panels,
                summary=f"{acc['bench'].nunique()} benchmarks x {acc['arm'].nunique()} arms x {acc['model'].nunique()} models")
    return {"rows": len(acc)}


BENCH_FIELD = {"pubmedqa": "Biomedical (PubMedQA)", "scifact": "Biomedical (SciFact)", "litqa2": "Biology (LitQA2)",
               "gpqa": "Science (GPQA)", "hle": "Mixed (HLE)", "hle-gold": "Biology / chemistry (HLE-gold)"}


def gains_by(ctx: Context, per: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Accuracy per arm and Δ vs A0 by the discipline and the language of each item's gold paper (P2)."""
    from ..ws1.faith.common import features_table, paper_key
    items = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet", columns=["bench", "item_id", "gold_paper_key"])
    try:
        f = features_table(ctx, ["file", "row", "domain", "oa_language", "summary_lang"])
        f["paper_key"] = [paper_key(a, b) for a, b in zip(f["file"], f["row"], strict=True)]
        f["lang"] = f["oa_language"].fillna(f["summary_lang"])     # OpenAlex language, else the summary's language
        meta = f.set_index("paper_key")[["domain", "lang"]]
    except FileNotFoundError:                                         # no WS1 features: benchmark fields only
        meta = pd.DataFrame(columns=["domain", "lang"])
    items["discipline"] = items["gold_paper_key"].map(meta["domain"]).fillna(items["bench"].map(BENCH_FIELD)).fillna("unknown")
    items["language"] = items["gold_paper_key"].map(meta["lang"]).fillna("unknown (no linked paper)")
    d = per.merge(items[["item_id", "discipline", "language"]], on="item_id", how="left")
    out = {}
    for key, col in (("discipline", "discipline"), ("language", "language")):
        g = d.groupby([col, "arm"]).agg(items=("item_id", "nunique"), acc=("correct", "mean")).reset_index()
        base = g[g["arm"] == "A0"].set_index(col)["acc"]
        g["delta_vs_A0"] = g["acc"] - g[col].map(base)
        out[key] = g.rename(columns={col: key}).sort_values([key, "arm"])
    return out
