"""X3 pieces that need no GPU: answer parsing of free-text system outputs, ScholarQABench item building."""
import json


def test_parse_answer_labels_letters_and_abstention():
    from alexandria.x3.common import parse_answer
    from alexandria.x3.systems import final_line
    lab = {"answer_type": "label", "options": ["true", "false"]}
    assert parse_answer("The claim is false given [2].", lab) == "false"
    assert parse_answer(final_line("Long answer citing [1].\nAnswer: true"), lab) == "true"
    mc = {"answer_type": "mc", "options": list("ABCDE"), "abstain_option": "E"}
    assert parse_answer("Answer: C", mc) == "C"
    assert parse_answer("(B) ciprofloxacin", mc) == "B"
    assert parse_answer("There is insufficient information in the papers.", mc) == "E"
    assert parse_answer("A short answer", {"answer_type": "short", "options": []}) == "A short answer"


def test_sqab_items(tmp_path):
    from alexandria.x3.common import sqab_items
    d = tmp_path / "scholarqabench"
    d.mkdir()
    (d / "scifact_test.jsonl").write_text(json.dumps({"input": "X causes Y.", "answer": "true",
                                                       "gold_ctx": [{"title": "T", "text": "X causes Y."}]}) + "\n")
    it = sqab_items(tmp_path, "sqab-scifact")[0]
    assert it["question"].endswith("Claim: X causes Y.") and it["options"] == ["true", "false"] and it["answer"] == "true"
    assert json.loads(it["gold_ctx"])[0]["title"] == "T"


def test_pareto_front():
    import pandas as pd

    from alexandria.x3.results import pareto_front
    d = pd.DataFrame({"system": ["cheap", "mid", "dominated", "best"], "tok": [100, 1000, 2000, 5000],
                      "acc": [0.3, 0.5, 0.4, 0.6]})
    assert pareto_front(d, "tok", "acc") == [True, True, False, True]
    tie = pd.DataFrame({"tok": [100, 100], "acc": [0.5, 0.5]})
    assert pareto_front(tie, "tok", "acc") == [True, True]            # equal points do not dominate each other
