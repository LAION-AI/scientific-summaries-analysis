"""WS1b helpers: sentence splitting, MiniCheck scoring, stratified allocation, agreement, adjudication (no GPU)."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alexandria.ws1.faith.abstract_support import recommend
from alexandria.ws1.faith.common import allocate, item_id, paper_key, split_sentences, stratified_pick
from alexandria.ws1.faith.grounding import aggregate, yes_prob
from alexandria.ws1.faith.human import adjudicate, alpha_nominal
from alexandria.ws1.faith.judge import judge_schema, minicheck_agreement, parse_answer, token_budget


def test_split_sentences_keeps_abbreviations():
    s = "Smith et al. report 81.2% accuracy. Fig. 3 shows the trend! Is it robust? Yes.\nNew block here"
    assert split_sentences(s) == ["Smith et al. report 81.2% accuracy.", "Fig. 3 shows the trend!",
                                  "Is it robust?", "Yes.", "New block here"]
    assert split_sentences("") == []
    assert split_sentences("The value 3.5 e.g. in Eq. 2 holds.") == ["The value 3.5 e.g. in Eq. 2 holds."]


def _lp(tok, p):
    return SimpleNamespace(decoded_token=tok, logprob=math.log(p))


def test_yes_prob_sums_yes_variants():
    lps = {1: _lp("Yes", 0.6), 2: _lp(" yes", 0.1), 3: _lp("No", 0.25), 4: _lp("YES", 0.02), 5: _lp("Ye", 0.01)}
    assert yes_prob(lps) == pytest.approx(0.72)
    assert yes_prob({}) == 0.0


def test_aggregate_max_over_chunks_min_over_sentences():
    sent, mn, mean = aggregate([[0.1, 0.9], [0.4, 0.3], [0.7]])
    assert sent == [0.9, 0.4, 0.7]
    assert mn == 0.4 and mean == pytest.approx(2.0 / 3)
    assert aggregate([]) == ([], None, None)


def test_allocate_floor_proportional_and_caps():
    a = allocate({"a": 1000, "b": 100, "c": 2, "d": 0}, 50, min_per=3)
    assert "d" not in a and a["c"] == 2 and sum(a.values()) == 50
    assert a["a"] > a["b"] >= 3
    assert allocate({"a": 5, "b": 5}, 100, min_per=1) == {"a": 5, "b": 5}
    assert sum(allocate({k: 10 for k in "abcdef"}, 3, min_per=1).values()) == 6   # floors win over n


def test_stratified_pick_is_deterministic():
    df = pd.DataFrame({"file": ["f"] * 200, "row": range(200), "group": ["x"] * 150 + ["y"] * 50})
    a = stratified_pick(df, 40, ["group"], seed=1)
    b = stratified_pick(df.sample(frac=1, random_state=3), 40, ["group"], seed=1)
    assert len(a) == 40 and set(a["row"]) == set(b["row"])
    assert (a["group"] == "y").sum() == 10


def test_alpha_perfect_and_chance():
    perfect = pd.DataFrame({"item_id": ["i1", "i1", "i2", "i2", "i3", "i3"], "annotator_id": ["a", "b"] * 3,
                            "label": ["supported", "supported", "unsupported", "unsupported", "contradicted",
                                      "contradicted"]})
    assert alpha_nominal(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    labs = ["supported", "unsupported", "contradicted"]
    noisy = pd.DataFrame([{"item_id": f"i{i}", "annotator_id": a, "label": labs[rng.integers(3)]}
                          for i in range(300) for a in ("a", "b")])
    assert abs(alpha_nominal(noisy)) < 0.15


def test_adjudicate_majority_and_ties():
    lab = pd.DataFrame({"item_id": ["x", "x", "x", "y", "y", "z"], "annotator_id": ["a", "b", "c", "a", "b", "a"],
                        "label": ["supported", "supported", "unsupported", "supported", "contradicted", "unsupported"],
                        "paper_key": "p", "field": "key_results", "omission": ""})
    g = adjudicate(lab).set_index("item_id")
    assert g.loc["x", "gold"] == "supported" and g.loc["x", "agreement"] == pytest.approx(2 / 3)
    assert g.loc["y", "gold"] == "disputed"
    assert g.loc["z", "gold"] == "unsupported" and g.loc["z", "n_labels"] == 1


def test_parse_answer_and_ids():
    sents = ["A holds.", "B was measured.", "The paper does not report seeds."]
    ok = parse_answer('<think>x</think>{"verdicts":[{"id":1,"evidence":"e1","verdict":"supported"},'
                      '{"id":2,"evidence":"e2","verdict":"Unsupported"},{"id":3,"evidence":"e3","verdict":"supported"}],'
                      '"omissions":["o"]}', sents)
    assert ok["label"] == "unsupported"                          # worst sentence verdict
    assert ok["unsupported_spans"] == ["B was measured."]        # spans are the flagged sentences, exactly
    assert ok["sentence_verdicts"] == ["supported", "unsupported", "supported"] and ok["omissions"] == ["o"]
    worst = parse_answer('{"verdicts":[{"id":1,"evidence":"","verdict":"unsupported"},'
                         '{"id":2,"evidence":"","verdict":"contradicted"}],"omissions":[]}', sents[:2])
    assert worst["label"] == "contradicted"
    assert parse_answer("not json", sents)["label"] == "parse_error"
    truncated = '{"verdicts":[{"id":1,"evidence":"e","verdict":"supported"}],"omissions":[]}'   # 1 of 3 sentences
    assert parse_answer(truncated, sents)["label"] == "parse_error"
    assert parse_answer('{"verdicts":[{"id":1,"evidence":"e","verdict":"supported"},{"id":2,"evidence":"e",'
                        '"verdict":"supported"},{"id":3,"evidence":"e","verdict":"supported"', sents)["label"] \
        == "parse_error"                                         # cut off at the token budget


def test_judge_schema_and_budget():
    sch = judge_schema(4)
    v = sch["properties"]["verdicts"]
    assert v["minItems"] == v["maxItems"] == 4
    assert list(v["items"]["properties"]) == ["id", "evidence", "verdict"]   # evidence before the verdict
    assert token_budget(10) > token_budget(2)


def test_minicheck_agreement():
    sent = pd.DataFrame({"paper_key": ["p"] * 4, "field": ["f"] * 4, "sent_index": [0, 1, 2, 3],
                         "verdict": ["supported", "supported", "unsupported", "contradicted"],
                         "sentence": list("abcd"), "evidence": [""] * 4, "input_type": ["full_text"] * 4})
    mc = pd.DataFrame({"paper_key": ["p"] * 4, "field": ["f"] * 4, "sent_index": [0, 1, 2, 3],
                       "prob": [0.9, 0.2, 0.1, 0.8], "supported": [True, False, False, True]})
    a = minicheck_agreement(sent, mc)
    assert a["n"] == 4 and a["agreement"] == 0.5
    assert a["table"] == {"both_supported": 1, "judge_only": 1, "minicheck_only": 1, "neither": 1}
    k = paper_key("data/arxiv/new_00300.parquet", 7)
    assert len(k) == 12 and k == paper_key("data/arxiv/new_00300.parquet", 7)
    assert item_id(k, "claims", 2) == f"{k}:claims:2" and item_id(k, "key_results") == f"{k}:key_results"


def test_recommend_thresholds():
    assert [recommend(x) for x in (0.9, 0.7, 0.5, 0.1, None)] == ["keep", "flag", "shorten", "drop", "n/a"]


def test_judge_and_minicheck_vs_gold():
    from alexandria.ws1.faith.judge import judge_vs_gold, minicheck_vs_gold
    rng = np.random.default_rng(0)
    keys = [f"p{i:03d}" for i in range(200)]
    gold = pd.DataFrame({"item_id": [f"{k}:key_results" for k in keys],
                         "gold": rng.choice(["supported", "unsupported", "contradicted"], 200, p=[.6, .3, .1])})
    judge = gold.rename(columns={"gold": "label"}).copy()
    flip = rng.random(200) < 0.2
    judge.loc[flip, "label"] = "unsupported"
    jv = judge_vs_gold(judge, gold)
    assert jv["n"] == 200 and 0.4 < jv["kappa"] < 1 and np.array(jv["confusion"]).sum() == 200
    y = gold["gold"].eq("supported").to_numpy()
    fs = pd.DataFrame({"paper_key": keys, "field": "key_results",
                       "score_min": np.clip(np.where(y, 0.8, 0.3) + rng.normal(0, 0.15, 200), 0, 1)})
    mc = minicheck_vs_gold(fs, gold)
    assert mc["auc"] > 0.9 and 0.3 < mc["best_threshold"] < 0.8
    assert np.all(np.diff(mc["isotonic"]["y"]) >= -1e-12)          # isotonic map is monotone
