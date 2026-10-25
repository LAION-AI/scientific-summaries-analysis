"""X4 pure functions: field router, topic groups, paper views, rule checks, gold recall."""
import numpy as np
import pandas as pd


def test_route_field():
    from alexandria.x4.common import route_field
    assert route_field("Which limitation does the graphene sensor study report?") == "limitations"
    assert route_field("Which study claims that X contradicts Y?") == "claims"
    assert route_field("What dataset was used to train the parser?") == "methods"
    assert route_field("Which approach achieved the highest accuracy on CIFAR-10?") == "key_results"
    assert route_field("Tell me about cats") == "any"


def test_paper_views_and_topic_groups():
    from alexandria.x4.common import paper_fields, paper_vectors, topic_groups
    units = pd.DataFrame({"kind": ["summary"] * 6 + ["fulltext"], "paper_key": ["a", "a", "b", "b", "c", "c", "a"],
                          "field": ["key_results", "claims", "key_results", "claims", "key_results", "claims", ""],
                          "text": ["ra", "ca", "rb", "cb", "rc", "cc", "full"]})
    pf = paper_fields(units)
    assert pf["a"] == {"key_results": "ra", "claims": "ca"} and "full" not in str(pf)
    emb = np.array([[1, 0], [1, 0.1], [1, 0.05], [1, 0.0], [0, 1], [0, 1]], np.float32)
    keys, vec = paper_vectors(units, emb)
    assert keys == ["a", "b", "c"] and np.allclose(np.linalg.norm(vec, axis=1), 1)
    groups = topic_groups(keys, vec, 2, 0.9, 5, np.random.default_rng(0))
    assert ["a", "b"] in groups and all(len(g) == 2 for g in groups)
    assert ["a", "c"] not in groups                                    # dissimilar: below min_sim


def test_check_item_rules():
    from alexandria.x4.common import check_item
    field = "The model fails on low-resource languages because the tokenizer splits words badly."
    ok = {"question": "Which weakness does the multilingual parser show?", "answer": "low-resource languages",
          "evidence": "fails on low-resource languages"}
    assert check_item(ok, field, ["unrelated text about chemistry"], 12, 85) == "ok"
    assert check_item({**ok, "question": "What does this paper report as a weakness?"}, field, [], 12, 85) == "not_self_contained"
    assert check_item({**ok, "question": "What limitation of the study concerns the tokenizer?"}, field, [], 12, 85) == "not_self_contained"
    assert check_item({**ok, "evidence": "works on every language perfectly"}, field, [], 12, 85) == "evidence_not_in_field"
    assert check_item(ok, field, [field], 12, 85) == "evidence_also_in_other_paper"
    assert check_item({**ok, "answer": "a " * 20}, field, [], 12, 85) == "answer_too_long"


def test_gold_recall():
    from alexandria.x4.common import gold_recall
    assert gold_recall(["a", "b"], ["b"]) == 1.0 and gold_recall(["a"], ["b", "a"]) == 0.5 and gold_recall([], []) is None
