"""WS4 pieces that need no GPU / model: renderings, conditions, run grid, rewards, aggregation."""
import json

import numpy as np
import pandas as pd
import pytest

KU = {"summary_title": "Graphene thermal transport", "executive_summary": "**Suspended** graphene conducts heat well.",
      "key_results": "Thermal conductivity of 5000 W/mK at room temperature.",
      "claims": json.dumps([{"description": "Graphene beats copper.", "supporting_evidence": "Raman data",
                             "contradicting_evidence": "", "implications": ""}])}


def test_clean_ku_and_every_format():
    from alexandria.ws4.common import FORMATS, clean_ku, render
    ku = clean_ku(KU)
    assert ku["executive_summary"].startswith("Suspended") and ku["claims"][0]["description"] == "Graphene beats copper."
    out = {f: render(ku, f) for f in FORMATS}
    assert out["prose"].startswith("# Graphene thermal transport") and "## Key results" in out["prose"]
    assert json.loads(out["json"])["key_results"].startswith("Thermal")
    assert "key_results:" in out["yaml"] and out["html"].startswith("<article>") and "Q: What is claimed?" in out["qa_bullets"]
    assert all("5000 W/mK" in v for v in out.values())          # same content, different surface form


def test_condition_text_diverse_rotates_formats():
    from alexandria.ws4.common import FORMATS, clean_ku, condition_text
    p = {"i": 0, "ku": clean_ku(KU), "full": "full text"}
    texts = [condition_text("diverse", p, e, list(FORMATS)) for e in range(len(FORMATS))]
    assert len(set(texts)) == len(FORMATS)
    assert condition_text("full", p, 3, list(FORMATS)) == "full text"
    assert condition_text("summary", p, 0, list(FORMATS)) == condition_text("summary", p, 5, list(FORMATS))


class _Ctx:
    def __init__(self, raw, profile="full", settings=None):
        self.cfg = type("C", (), {"raw": raw, "profile": profile, "profile_settings": settings or {}})()

    def section(self, dotted):
        cur = self.cfg.raw
        for k in dotted.split("."):
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur


def test_run_grid_and_debug_settings():
    from alexandria.ws4.common import run_grid
    raw = {"ws4": {"conditions": ["summary", "ku", "none"], "train": {"models": ["m"], "seeds": [0, 1], "formats": ["doc", "qa"]},
                   "grpo": {"conditions": ["summary"]}, "tools": {"enabled": True}}}
    runs = run_grid(_Ctx(raw))
    assert len(runs) == 2 * (2 * 3 + 1 + 1) and len({r["run"] for r in runs}) == len(runs)
    dbg = run_grid(_Ctx(raw, "debug", {"ws4_models": ["tiny"], "ws4_seeds": [0]}))
    assert {r["model"] for r in dbg} == {"tiny"} and {r["seed"] for r in dbg} == {0}


def test_answer_reward():
    from alexandria.ws4.train import answer_reward
    comps = [[{"role": "assistant", "content": "5000 W/mK"}], [{"role": "assistant", "content": "copper"}], "about 5,000 W/mK",
             "The average improvement was +140% over fastText.", "graphene " * 30]
    assert answer_reward(comps, ["5000 W/mK", "graphene", "5000 W/mK", "+140%", "graphene"]) == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_set_scores_ci_and_injection():
    from alexandria.ws4.results import ci, injection, set_scores
    unit = {"rows": [{"task": "mmlu_pro_biology", "value": .5, "stderr": .02, "n": 100},
                     {"task": "mmlu_pro_physics", "value": .3, "stderr": .02, "n": 300},
                     {"task": "sciq", "value": .9, "stderr": .01, "n": 1000}]}
    s = set_scores(unit)
    assert s["mmlu_pro_stem"][0] == pytest.approx(.35) and s["sciq"][0] == .9
    m, lo, hi = ci([.5, .6, .7], None)
    assert m == pytest.approx(.6) and lo < .6 < hi
    m1, lo1, hi1 = ci([.5], .01)
    assert hi1 - lo1 == pytest.approx(2 * 1.96 * .01)
    rng = np.random.default_rng(0)
    rows = [{"paper_key": f"{s}{i}", "split": s, "correct": int(rng.random() < p)}
            for s, p in (("train", .8), ("heldout", .3)) for i in range(40) for _ in range(3)]
    base = [{**r, "correct": int(rng.random() < .3)} for r in rows]
    e = injection(pd.DataFrame(rows), pd.DataFrame(base), n_boot=200)
    assert e["effect"] > .2 and e["lo"] < e["effect"] < e["hi"] and e["gain_train"] > e["gain_heldout"]


def test_decontam_ngrams():
    from alexandria.ws4.common import ngrams
    ev = ngrams("Which element has the highest thermal conductivity among all known natural bulk materials today", 13)
    train = "Q: which element has the highest thermal conductivity among all known natural bulk materials today? A: diamond"
    assert ngrams(train, 13) & ev
    assert not (ngrams("graphene conducts heat very well at room temperature in suspended samples measured by Raman", 13) & ev)
