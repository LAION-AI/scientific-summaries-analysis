"""WS2 structure ablation: renderings and the paired analysis (CPU only)."""
import json

import numpy as np
import pandas as pd

KU = {"summary_title": "Graphene heat", "executive_summary": "Overview.", "research_context": "Context.",
      "methodological_details": "Raman.", "key_results": "5000 W/mK.", "interpretation_implications": "Good.",
      "contradictions_limitations": "Few samples.", "three_takeaways": "One. Two. Three.",
      "claims": json.dumps([{"description": "Graphene beats copper.", "supporting_evidence": "Raman data",
                             "contradicting_evidence": "", "implications": ""}])}


def test_default_rendering_unchanged_and_ablations():
    from alexandria.ws2.common import ABLATION_GROUPS, ablation_texts, render_ku_clean
    full, _ = render_ku_clean(KU)
    assert "## key_results\n5000 W/mK." in full and "- Graphene beats copper. Evidence: Raman data" in full
    ab = ablation_texts(KU, seed=1)
    assert set(ab) == {f"ku_minus_{g}" for g in ABLATION_GROUPS} | {"ku_results_claims", "ku_flat"}
    assert "5000 W/mK" not in ab["ku_minus_results"] and "Graphene beats copper" not in ab["ku_minus_claims"]
    assert "Raman." not in ab["ku_results_claims"] and "5000 W/mK" in ab["ku_results_claims"]
    flat = ab["ku_flat"]
    assert "##" not in flat and "- " not in flat
    assert sorted(flat.split()) == sorted(render_ku_clean(KU, headings=False)[0].split())   # same content


def test_cut_at_sentence():
    from alexandria.ws2.ablation import cut_at_sentence

    class Tok:
        def __call__(self, t, add_special_tokens=False):
            return {"input_ids": t.split()}

        def decode(self, ids):
            return " ".join(ids)
    t = "One two three. Four five six. Seven eight"
    assert cut_at_sentence(Tok(), t, 7) == "One two three. Four five six."
    assert cut_at_sentence(Tok(), t, 100) == t


def test_ablation_table_pooled_and_holm():
    from alexandria.stats import cluster_bootstrap
    from alexandria.ws2.results import ablation_table
    rng = np.random.default_rng(0)
    arms = {"ku_clean": .7, "ku_local": .65, "prose_local": .5, "ku_flat": .6, "ku_minus_results": .4}
    rows = [{"model": m, "arm": a, "qid": f"p{i}:q{j}", "paper_key": f"p{i}", "correct": float(rng.random() < p)}
            for m in ("m1", "m2") for a, p in arms.items() for i in range(30) for j in range(3)]
    pq = pd.DataFrame(rows)
    tbl, boot = cluster_bootstrap(pq, "correct", "paper_key", ["model", "arm"], n_boot=300)
    cx = pd.DataFrame([{"arm": a, "tokens": 1000} for a in arms])
    abl = ablation_table(pq, tbl, boot, cx, ["m1", "m2"], 300)
    pooled = abl[abl["model"] == "all (pooled)"].set_index("arm")
    assert pooled.loc["ku_local", "vs"] == "prose_local" and pooled.loc["ku_local", "diff"] > 0
    assert pooled.loc["ku_minus_results", "diff"] < 0 and abl["p_holm"].between(0, 1).all()
