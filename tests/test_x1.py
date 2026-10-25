"""X1 pieces that need no GPU / data: metrics, fusion, unit -> paper aggregation, PQ vs flat, trace parsing, units."""
import json

import numpy as np
import pytest


def test_recall_mrr_ndcg():
    from alexandria.x1.common import first_gold_rank, mrr, ndcg_at, recall_at
    ranked = ["a", "g1", "b", "g2", "c"]
    gold = {"g1", "g2", "g3"}
    assert recall_at(ranked, gold, 1) == 0 and recall_at(ranked, gold, 2) == pytest.approx(1 / 3)
    assert recall_at(ranked, gold, 5) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == 0.5 and mrr(["x"], gold) == 0.0 and first_gold_rank(ranked, gold) == 2
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at(ranked, gold, 10) == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)
    assert np.isnan(recall_at(ranked, set(), 5))


def test_rrf_fuses_ranks_not_scores():
    from alexandria.x1.common import rrf
    assert rrf([["a", "b", "c"], ["b", "c", "a"]], k=60)[0] == "b"
    assert rrf([["a"], []], k=60) == ["a"]


def test_units_to_papers_max_score():
    from alexandria.x1.common import to_papers
    unit_paper = np.array(["p1", "p1", "p2", "p3"])
    papers, scores = to_papers(np.array([1, 2, 0, 3, -1]), np.array([0.9, 0.8, 0.7, 0.1, 0.0]), unit_paper, 10)
    assert papers == ["p1", "p2", "p3"] and scores == [0.9, 0.8, 0.1]


def test_pq_params_scaled_from_release():
    from alexandria.x1.common import pq_params
    assert pq_params(21_796_509, {}) == (4113, 32)          # ~ the release's IVF4096 with nprobe 32
    assert pq_params(4000, {}) == (1, 1)


def test_pq48_close_to_flat_on_random_vectors():
    faiss = pytest.importorskip("faiss")
    from alexandria.x1.common import build_pq
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(50, 96)).astype(np.float32)
    x = (centers[rng.integers(0, 50, 3000)] + 0.1 * rng.normal(size=(3000, 96))).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    flat = faiss.IndexFlatIP(96)
    flat.add(x)
    pq = build_pq(x, 1, m=48, bits=8)
    q = x[:50] + 0.01
    _, fi = flat.search(q, 10)
    _, pi = pq.search(q, 10)
    overlap = np.mean([len(set(fi[i]) & set(pi[i])) / 10 for i in range(50)])
    assert 0.3 < overlap <= 1.0 and pq.code_size == 48


def test_agent_query_parsing():
    from alexandria.x1.diagnosis import _agent_queries
    t = json.dumps([{"action": "search", "arguments": {"query": "MC dropout NLI", "k": 3}}, {"action": "get_paper",
                    "arguments": {"id": "x"}}, {"action": "search_fulltext", "arguments": {"query": "AUC"}}, {"action": "answer"}])
    assert _agent_queries(t) == [{"query": "MC dropout NLI", "k": 3}, {"query": "AUC", "k": 5}]
    assert _agent_queries("not json") == []


def test_units_of_a_record():
    from alexandria.x1.corpus import units_of
    rec = {"summary_title": "Graphene heat", "executive_summary": "Graphene conducts heat.", "key_results": "5000 W/mK.",
           "claims": json.dumps([{"description": "Beats copper.", "supporting_evidence": "Raman"}]),
           "text_sanitized": " ".join(["word"] * 1200)}
    u = units_of(rec, 500, 40)
    assert [f for f, _ in u["fields"]][0] == "summary_title" and u["claims"] == [("claim", "Beats copper. Raman")]
    assert len(u["chunks"]) == 3 and "Graphene conducts heat." in u["whole"][0][1]
