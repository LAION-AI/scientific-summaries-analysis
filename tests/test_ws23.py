"""Unit tests for WS2 / WS3 pieces that need no GPU: retrieval, backend + cache, agent loop (scripted LLM), action
validation, QA metrics, bootstrap statistics, summary rendering, question checks, benchmark adapters."""
import json

import numpy as np
import pandas as pd
import pytest

from alexandria.qa import exact_match, is_abstention, normalize_answer, ngram_overlap, token_f1
from alexandria.skill.retrieval import BM25, rrf, tokenize, top_k
from alexandria.skill.tools import action_schema, openai_tools
from alexandria.stats import cluster_bootstrap, holm, paired_difference


def test_bm25_ranks_the_matching_document_first():
    docs = ["graphene thermal conductivity measured at room temperature",
            "protein folding with deep learning", "thermal conductivity of silicon nanowires"]
    bm = BM25.build(docs)
    s = bm.scores("graphene conductivity")
    assert int(np.argmax(s)) == 0 and s[1] == 0
    assert "the" not in tokenize("The graphene")


def test_rrf_and_top_k():
    fused = rrf([np.array([3, 1, 2]), np.array([1, 3])])
    assert max(fused, key=fused.get) in (1, 3) and fused[2] < fused[1]
    assert list(top_k(np.array([0.1, 0.9, 0.5]), 2)) == [1, 2]
    assert list(top_k(np.array([0.1, 0.9, 0.5]), 2, np.array([True, False, True]))) == [2, 0]


@pytest.fixture()
def tiny_index(tmp_path):
    units = pd.DataFrame([
        {"uid": 0, "kind": "summary", "field": "key_results", "text": "Graphene conductivity reached 5000 W/mK.",
         "paper_key": "p1", "file": "f", "row": 0, "paper_id": "2401.00001", "title": "Graphene heat", "has_text": True},
        {"uid": 1, "kind": "summary", "field": "methodological_details", "text": "Raman thermometry on suspended films.",
         "paper_key": "p1", "file": "f", "row": 0, "paper_id": "2401.00001", "title": "Graphene heat", "has_text": True},
        {"uid": 2, "kind": "summary", "field": "key_results", "text": "Protein folding accuracy improved by 10 points.",
         "paper_key": "p2", "file": "f", "row": 1, "paper_id": "x", "title": "Folding", "has_text": False},
        {"uid": 3, "kind": "fulltext", "field": "", "text": "We measured graphene with Raman spectroscopy at 300 K.",
         "paper_key": "p1", "file": "f", "row": 0, "paper_id": "2401.00001", "title": "Graphene heat", "has_text": True},
    ])
    units.to_parquet(tmp_path / "units.parquet", index=False)
    for kind, g in units.groupby("kind"):
        BM25.build(g["text"].tolist()).save(tmp_path / f"bm25_{kind}")
    (tmp_path / "index.json").write_text(json.dumps({"version": "test"}))
    return tmp_path


def test_backend_search_fields_filters_and_cache(tiny_index):
    from alexandria.skill.backend import Backend
    be = Backend(tiny_index, None, tiny_index / "cache.sqlite", dense=False)
    text, papers = be.call("search", {"query": "graphene conductivity", "field": "key_results", "k": 3}, view="summary")
    assert papers[0] == "p1" and "5000" in text and "Raman" not in text          # field filter
    _, papers_m = be.call("search", {"query": "Raman thermometry", "field": "methods"}, view="summary")
    assert papers_m == ["p1"]
    _, oa = be.call("search", {"query": "protein folding"}, view="summary", only_full_text=True)
    assert "p2" not in oa                                                         # S-OA restriction
    t2, _ = be.call("search", {"query": "graphene conductivity", "field": "key_results", "k": 3}, view="summary")
    assert t2 == text and be.hits == 1                                            # replayed from the cache
    paper, _ = be.call("get_paper", {"id": "2401.00001", "fields": ["key_results"]}, view="summary")
    assert "5000" in paper
    ft, _ = be.call("search_fulltext", {"query": "Raman 300 K"}, view="summary")
    assert "300 K" in ft
    bad, _ = be.call("search", {"query": "x", "field": "nope"}, view="summary")
    assert bad.startswith("Invalid arguments")


def test_agent_loop_with_scripted_model(tiny_index):
    from alexandria.skill.backend import Backend
    from alexandria.ws3.agent import Episode, run_agent, system_prompt
    be = Backend(tiny_index, None, None, dense=False)
    item = {"bench": "qabench-open", "item_id": "q1", "question": "Thermal conductivity of graphene?",
            "answer_type": "short", "options": [], "gold_paper_key": "p1"}
    script = [{"thought": "search", "action": "search", "arguments": {"query": "graphene conductivity"}},
              {"thought": "done", "action": "answer", "arguments": {"answer": "5000 W/mK"}}]
    calls = []

    def gen(msgs, schemas, temperature, seeds, max_tokens, n=1):
        calls.append(schemas)
        return [([json.dumps(script[len(calls) - 1])], 10, 5) for _ in msgs]

    ep = Episode(item=item, arm="S", seed=0, tools=["search", "get_paper"], view="summary", only_full_text=False,
                 messages=[{"role": "system", "content": system_prompt(item, ["search", "get_paper"], 6)},
                           {"role": "user", "content": item["question"]}])
    run_agent([ep], gen, be, lambda t: len(t.split()), lambda t, n: " ".join(t.split()[:n]),
              {"max_tool_calls": 6, "max_retrieved_tokens": 8000}, 0.0)
    assert ep.answer == "5000 W/mK" and ep.calls == 1 and "p1" in ep.papers and ep.error is None
    assert "anyOf" in calls[0][0]                                                 # tools allowed on step 1
    # budget: a 3-token budget truncates the first result and forces an answer next
    ep2 = Episode(item=item, arm="S", seed=0, tools=["search"], view="summary", only_full_text=False,
                  messages=list(ep.messages[:2]))
    calls.clear()
    run_agent([ep2], gen, be, lambda t: len(t.split()), lambda t, n: " ".join(t.split()[:n]),
              {"max_tool_calls": 6, "max_retrieved_tokens": 3}, 0.0)
    assert ep2.retrieved == 3 and "anyOf" not in calls[1][0]                      # answer-only after the budget
    # unknown tool -> episode ends with an error
    ep3 = Episode(item=item, arm="F", seed=0, tools=["search_fulltext"], view="fulltext", only_full_text=False,
                  messages=list(ep.messages[:2]))
    calls.clear()
    run_agent([ep3], gen, be, len, lambda t, n: t[:n], {"max_tool_calls": 6}, 0.0)
    assert ep3.error == "unknown_tool:search"


def test_action_schema_and_pilot_validation():
    from alexandria.ws3.agent import answer_schema, majority, valid_action
    sch = action_schema(["search"], {"type": "string"})
    assert {o["properties"]["action"]["const"] for o in sch["anyOf"]} == {"answer", "search"}
    assert "anyOf" not in action_schema(["search"], {"type": "string"}, allow_tools=False)
    assert openai_tools(["search"])[0]["function"]["name"] == "search"
    assert valid_action({"action": "search", "arguments": {"query": "x", "k": 3}}, ["search"])[0]
    assert not valid_action({"action": "search", "arguments": {"k": 3}}, ["search"])[0]
    assert valid_action({"action": "search", "arguments": {"query": "x", "field": "bad"}}, ["search"])[1] == "bad_enum"
    assert not valid_action("text", ["search"])[0]
    assert answer_schema({"answer_type": "label", "options": ["yes", "no"]})["enum"] == ["yes", "no"]
    assert majority(["5 K", "5 k", "7 K"]) == "5 K"


def test_qa_metrics():
    assert normalize_answer("The 3.2 eV.") == "3.2 ev"
    assert exact_match("about 3.2 eV", "3.2 eV") and exact_match("3.21", "3.2", rel_tol=0.01)
    assert not exact_match("3", "3.2")
    assert token_f1("graphene oxide film", "graphene film") > 0.7
    assert is_abstention("I don't know.") and not is_abstention("42 K")
    assert ngram_overlap("a b c d e f", "x a b c d e f y", n=5) == 1.0


def test_cluster_bootstrap_paired_and_holm():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(40):
        for q in range(5):
            for arm, prob in (("a", 0.8), ("b", 0.5)):
                rows.append({"paper": p, "arm": arm, "model": "m", "correct": int(rng.random() < prob)})
    df = pd.DataFrame(rows)
    tbl, boot = cluster_bootstrap(df, "correct", "paper", ["model", "arm"], n_boot=500)
    a = tbl.set_index("arm").loc["a"]
    assert a["lo"] < a["mean"] < a["hi"] and a["clusters"] == 40
    d = paired_difference(boot, ("m", "a"), ("m", "b"), a["mean"], tbl.set_index("arm").loc["b", "mean"])
    assert d["lo"] > 0 and d["p"] < 0.05
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_ku_rendering_raw_vs_clean():
    from alexandria.ws2.common import render_ku_clean, render_ku_raw
    rec = {"summary_title": "[Unknown]", "executive_summary": "**Bold** result.", "key_results": "Not provided",
           "claims": json.dumps([{"description": "X improves Y.", "supporting_evidence": "Table 2",
                                  "contradicting_evidence": "", "implications": ""}])}
    raw = render_ku_raw(rec)
    clean, issues = render_ku_clean(rec)
    assert "[Unknown]" in raw and "[Unknown]" not in clean
    assert "**" not in clean and "- X improves Y. Evidence: Table 2" in clean
    assert "Not provided" in raw and "Not provided" not in clean


def test_question_verification():
    from alexandria.ws2.questions import verify
    src = "We measured the thermal conductivity of suspended graphene to be 5000 W/mK using Raman thermometry."
    w = normalize_answer(src).split()
    grams = {tuple(w[i:i + 8]) for i in range(len(w) - 8 + 1)}
    good = {"question": "What thermal conductivity was found for suspended graphene with Raman thermometry?",
            "answer": "5000 W/mK", "evidence": "thermal conductivity of suspended graphene to be 5000 W/mK"}
    assert verify(good, src, grams, {}, []) == "ok"
    assert verify({**good, "evidence": "it was 9000 W/mK in copper wires"}, src, grams, {}, []) == "evidence_not_in_paper"
    assert verify({**good, "question": "In this paper, what was measured?"}, src, grams, {}, []) == "not_self_contained"
    copied = {**good, "question": "We measured the thermal conductivity of suspended graphene to be what value?"}
    assert verify(copied, src, grams, {}, []) == "copies_paper_text"
    assert verify(good, src, grams, {}, [good["question"].lower()]) == "duplicate"
    for trivial in ("Equation 7b", "Fig. 3", "Table 2", "Section 4.1", "Theorem 2"):
        assert verify({**good, "answer": trivial}, src, grams, {}, []) == "trivial_reference", trivial
    assert verify({**good, "answer": "Figueroa-Lopez estimator"}, src, grams, {}, []) == "ok"


def test_benchmark_mc_adapter_is_deterministic_and_graded():
    from alexandria.ws3.benchmarks import _mc
    from alexandria.ws3.results import exact_label
    t1, letters, gold = _mc("id1", "Q?", "right", ["w1", "w2", "w3"], 0, ["Insufficient information"])
    t2, _, gold2 = _mc("id1", "Q?", "right", ["w1", "w2", "w3"], 0, ["Insufficient information"])
    assert t1 == t2 and gold == gold2 and letters[-1] == "E" and f"{gold}. right" in t1
    assert exact_label(gold, gold, "mc") and exact_label(f"{gold}. right", gold, "mc")
    assert not exact_label("Z", gold, "mc") and exact_label("Yes", "yes", "label")


def test_one_ku_per_paper_with_ku_fixes():
    from alexandria.ws2.ku import apply_fixes, filled
    raw = {"summary_title": "[Unknown]", "summary_authors": "", "executive_summary": "E.",
           "claims": '[{"description": "d", "caveats": "c"}]'}
    fix = {"title_fault": "placeholder", "title_fixed": "Real title", "authors_fault": None, "json_faults":
           "claims:extra_keys=caveats", "claims_fixed": '[{"description": "d", "supporting_evidence": null, '
           '"contradicting_evidence": null, "implications": null}]'}
    fixed, applied = apply_fixes(raw, fix)
    assert fixed["summary_title"] == "Real title" and "caveats" not in fixed["claims"]
    assert applied == ["title:placeholder", "claims"] and raw["summary_title"] == "[Unknown]"
    assert apply_fixes(raw, None) == (raw, [])
    assert filled(raw) == 3


def test_gold_linking_by_doi_then_title():
    import pandas as pd

    from alexandria.ws3.gold import dois_in, link, norm_doi, norm_title
    assert dois_in("['https://doi.org/10.1128/msphere.00109-24']") == ["10.1128/msphere.00109-24"]
    assert dois_in(np.array(["https://doi.org/10.1128/msphere.00109-24"], dtype=object)) == ["10.1128/msphere.00109-24"]
    assert norm_doi("DOI: 10.1186/1471-2229-11-102.") == "10.1186/1471-2229-11-102"
    assert norm_title("Short title") is None and norm_title("A Study of X in Y, Z!") == "a study of x in y z"
    lookup = pd.DataFrame([
        {"file": "a.parquet", "row": 3, "has_text": False, "kind": "doi", "key": "10.1234/x"},
        {"file": "b.parquet", "row": 1, "has_text": True, "kind": "doi", "key": "10.1234/x"},
        {"file": "c.parquet", "row": 7, "has_text": True, "kind": "title", "key": "a study of x in y z"}])
    lookup = lookup.sort_values(["kind", "key", "has_text", "file", "row"], ascending=[True, True, False, True, True])
    items = pd.DataFrame([
        {"item_id": "1", "gold_paper_key": None, "gold_refs": [{"doi": "https://doi.org/10.1234/X"}]},
        {"item_id": "2", "gold_paper_key": None, "gold_refs": [{"doi": "10.9/none", "title": "A study of X in Y, Z"}]},
        {"item_id": "3", "gold_paper_key": None, "gold_refs": [{"title": "Unknown paper about something else"}]},
        {"item_id": "4", "gold_paper_key": "keep", "gold_refs": []}])
    out = link(items, lookup).set_index("item_id")
    assert (out.at["1", "gold_file"], out.at["1", "gold_match"]) == ("b.parquet", "doi")    # full-text row wins
    assert (out.at["2", "gold_file"], out.at["2", "gold_match"]) == ("c.parquet", "title")
    assert out.at["3", "gold_paper_key"] is None and out.at["4", "gold_paper_key"] == "keep"
