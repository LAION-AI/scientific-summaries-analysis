"""WS2 / WS3 `results` + `report` on synthetic graded outputs (no GPU): the analysis and the PDF run end to end."""
import json

import numpy as np
import pandas as pd

from alexandria.cli import _load_stages
from alexandria.config import load_config
from alexandria.runtime import STAGES, Context, run_stage


def _ctx(tmp_path, ws, stage):
    _load_stages()
    cfg = load_config(ws, "debug", [f"site.out_root={tmp_path}", f"{ws}.results.n_boot=200"])
    return Context(cfg, STAGES[(ws, stage)])


def test_ws2_results_and_report(tmp_path):
    rng = np.random.default_rng(0)
    cfg = _ctx(tmp_path, "ws2", "results").cfg
    papers = pd.DataFrame({"paper_key": [f"p{i}" for i in range(20)], "ws1_flagged": [i < 6 for i in range(20)]})
    qs = pd.DataFrame([{"qid": f"p{i}:q{j}", "paper_key": f"p{i}", "qtype": ["finding", "method"][j % 2],
                        "domain": "Physics", "ws1_flagged": i < 6, "question": "q", "answer": "a"}
                       for i in range(20) for j in range(4)])
    arms = ["closed_book", "full_text", "full_text_matched", "ku_raw", "ku_clean"]
    acc = dict(zip(arms, [0.2, 0.8, 0.55, 0.6, 0.65], strict=True))
    graded = pd.DataFrame([{"qid": q, "arm": a, "model": m, "seed": 0, "prediction": "x",
                            "correct": int(rng.random() < acc[a]), "attempted": 1, "em": 0, "f1": 0.5}
                           for q in qs["qid"] for a in arms for m in ("m1", "m2")])
    ctxs = pd.DataFrame([{"paper_key": p, "arm": a, "tokens": t, "context": "c"} for p in papers["paper_key"]
                         for a, t in zip(arms, [0, 7500, 1900, 2500, 1900], strict=True)])
    ov = pd.DataFrame([{"qid": q, "arm": "full_text", "overlap5": rng.random()} for q in qs["qid"]])
    for stage, name, df in (("papers", "papers.parquet", papers), ("questions", "questions.parquet", qs),
                            ("grade", "graded.parquet", graded), ("contexts", "contexts.parquet", ctxs),
                            ("contexts", "question_overlap.parquet", ov)):
        d = cfg.stage_dir("ws2", stage)
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / name, index=False)
    assert run_stage(_ctx(tmp_path, "ws2", "results")) == 0
    out = cfg.stage_dir("ws2", "results")
    comp = pd.read_csv(out / "comparisons.csv")
    ft = comp[(comp["arm"] == "full_text") & (comp["vs"] == "closed_book")]
    assert (ft["diff"] > 0.4).all() and (ft["p_holm"] < 0.05).all()
    assert (out / "figures" / "ws2_accuracy_vs_tokens.png").exists()
    assert len(pd.read_csv(out / "cleaning_effect.csv")) == 4                      # 2 models x flagged / not
    assert run_stage(_ctx(tmp_path, "ws2", "report")) == 0
    assert list(cfg.stage_dir("ws2", "report").glob("WS2_report_*.pdf"))


def test_ws3_results_and_report(tmp_path):
    rng = np.random.default_rng(1)
    cfg = _ctx(tmp_path, "ws3", "results").cfg
    rows = []
    for bench in ("qabench-open", "pubmedqa"):
        for i in range(30):
            for arm, p in (("A0", .3), ("R", .5), ("S", .6), ("F", .55)):
                for model in ("qwen2.5-7b", "phi4-mini"):
                    rows.append({"bench": bench, "item_id": f"{bench}:{i}", "cluster": f"c{i // 3}", "arm": arm,
                                 "model": model, "seed": 0, "correct": int(rng.random() < p), "abstained": 0,
                                 "tool_calls": 2 if arm in ("S", "F") else 0, "retrieved_tokens": 3000,
                                 "tokens_in": 5000, "tokens_out": 200, "gpu_seconds": 0.5,
                                 "gold_retrieved": True if bench == "qabench-open" and arm != "A0" else None})
    d = cfg.stage_dir("ws3", "grade")
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / "graded.parquet", index=False)
    items = pd.DataFrame([{"bench": b, "item_id": f"{b}:{i}",
                           "gold_paper_key": f"p{i}" if b == "qabench-open" or i % 2 else None}
                          for b in ("qabench-open", "pubmedqa") for i in range(30)])
    d = cfg.stage_dir("ws3", "benchmarks")
    d.mkdir(parents=True, exist_ok=True)
    items.to_parquet(d / "items.parquet", index=False)
    assert run_stage(_ctx(tmp_path, "ws3", "results")) == 0
    out = cfg.stage_dir("ws3", "results")
    acc = pd.read_csv(out / "accuracy.csv")
    assert set(acc["arm"]) == {"A0", "R", "S", "F"} and acc["lo"].le(acc["mean"]).all()
    assert set(acc["bench"]) == {"qabench-open", "pubmedqa", "pubmedqa [gold in corpus]"}   # QA bench: all linked
    assert (out / "figures" / "ws3_headline_delta_vs_size.png").exists()
    panels = json.loads((out / "dashboard.json").read_text())["panels"]
    assert any(p.get("title", "").startswith("Paired differences") for p in panels)
    assert run_stage(_ctx(tmp_path, "ws3", "report")) == 0
