"""X5 pure functions: counterfactual numbers, KU sentence edits and checks, retrieval swap, outcome labels."""
import json

import numpy as np

KU = {"summary_title": "A sensor study", "executive_summary": "We build a graphene sensor. It reaches 92% accuracy on the test set.",
      "key_results": "Accuracy was 92% with the graphene sensor.",
      "claims": [{"description": "The graphene sensor is accurate.", "supporting_evidence": "92% accuracy on 300 samples."}]}


def test_perturb_number():
    from alexandria.x5.common import perturb_number
    assert perturb_number("92%", "s", [0.5]) == "46%"
    assert perturb_number("92%", "s", [2.0]) is None                    # a percentage above 100 is not plausible
    assert perturb_number("1,000 samples", "s", [2.0]) == "2,000 samples"
    assert perturb_number("40-50%", "s", [2.0]) == "80-100%"            # both ends of a range
    assert perturb_number("2.8 Å", "s", [2.0]) == "5.6 Å"
    for name in ("GPT-4", "CRYSTALS-Kyber", "G(z) = G0 (1+z)", "10⁶ kg"):
        assert perturb_number(name, "s", [2.0]) is None
    assert perturb_number("12", "a", [0.5, 2.0, 3.0]) == perturb_number("12", "a", [0.5, 2.0, 3.0])   # deterministic


def test_sentences_edit_and_check():
    from alexandria.x5.common import apply_edits, check, rank_candidates, replace_everywhere, sentences
    s = sentences(KU)
    assert {x["field"] for x in s} >= {"executive_summary", "key_results", "claims"}
    top = rank_candidates(s, "What accuracy does the graphene sensor reach?", "92%", 3)
    assert all("92" in t["text"] for t in top)
    edited = apply_edits(KU, [(t, t["text"].replace("92%", "46%")) for t in top])
    assert KU["key_results"].startswith("Accuracy was 92%")             # deep copy: the original is unchanged
    assert check(KU, edited, "92%", "46%") == "ok"
    assert edited["executive_summary"].startswith("We build a graphene sensor.")   # untouched sentence kept
    assert check(KU, apply_edits(KU, [(top[0], "Something else.")]), "92%", "46%") == "cf_answer_not_in_edited_ku"
    partial = apply_edits(KU, [(top[0], top[0]["text"].replace("92%", "46%"))])
    assert check(KU, partial, "92%", "46%") == "original_answer_left"
    assert check(KU, replace_everywhere(partial, "92%", "46%"), "92%", "46%") == "ok"
    assert check(KU, KU, "92%", "92 %") == "same_answer"
    assert check(KU, KU, "graphene", "graphene oxide") == "nested_answers"


def test_swap_units_copy_only():
    from alexandria.x5.common import summary_units, swap_units
    cf = json.loads(json.dumps(KU))
    cf["key_results"] = "Accuracy was 46% with the graphene sensor."
    ou, cu = summary_units(KU), summary_units(cf)
    assert [u["field"] for u in ou] == [u["field"] for u in cu]
    retrieved = [{"paper_key": "other", "field": "key_results", "text": "unrelated"},
                 {"paper_key": "g", "field": "key_results", "text": ou[[u["field"] for u in ou].index("key_results")]["text"]},
                 {"paper_key": "g", "field": "methods_missing", "text": "dropped"}]
    snap = json.dumps(retrieved)
    a, b = swap_units(retrieved, "g", ou, cu, False), swap_units(retrieved, "g", ou, cu, True)
    assert a == ["unrelated", "Accuracy was 92% with the graphene sensor."]
    assert b == ["unrelated", "Accuracy was 46% with the graphene sensor."]
    assert json.dumps(retrieved) == snap                                   # the retrieved results are not modified


def test_label_and_bootstrap():
    from alexandria.x5.common import label
    from alexandria.x5.stages import boot_ci
    assert label("CORRECT", "INCORRECT") == "original"
    assert label("INCORRECT", "CORRECT") == "counterfactual"
    assert label("CORRECT", "CORRECT") == "both"
    assert label("NOT_ATTEMPTED", "NOT_ATTEMPTED") == "abstain"
    assert label("INCORRECT", "INCORRECT") == "other"
    m, lo, hi = boot_ci(np.array([1, 0, 1, 1, 0, 1, 0, 1.0]), 500)
    assert lo <= m <= hi and abs(m - 0.625) < 1e-9
    assert all(np.isnan(boot_ci(np.array([]), 10)))
