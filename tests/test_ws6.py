"""WS6 pieces on CPU: projector, knowledge-token assembly and label masking, split by paper, retrieval, results."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def test_projector_shapes_and_scale():
    from alexandria.ws6.common import make_projector
    p = make_projector(768, 64, 4, 32, emb_rms=0.05)
    y = p(torch.randn(3, 768))
    assert y.shape == (3, 4, 32)
    assert y.pow(2).mean(-1).sqrt().mean().item() == pytest.approx(0.05, rel=1e-3)   # starts at the LLM's token scale


def test_assemble_masks_everything_but_the_answer():
    from alexandria.ws6.common import assemble
    s = assemble([1, 2, 3], 4, [5, 6], [7, 8, 9])
    assert s["ids"] == [1, 2, 3, -1, -1, -1, -1, 5, 6, 7, 8, 9]
    assert s["labels"] == [-100] * 9 + [7, 8, 9]
    assert s["kn_mask"] == [False] * 3 + [True] * 4 + [False] * 5
    g = assemble([1, 2], 0, [])                                   # generation prompt, no knowledge, no answer
    assert g["labels"] == [-100, -100] and not any(g["kn_mask"])


def test_build_batch_places_knowledge_and_pads():
    from alexandria.ws6.common import assemble, build_batch
    emb = torch.nn.Embedding(20, 6)
    seqs = [assemble([1, 2], 2, [3], [4]), assemble([1], 0, [], [5, 6])]
    kn = [torch.full((2, 6), 9.0), None]
    for left in (False, True):
        e, m, lab = build_batch(seqs, kn, emb, pad_left=left)
        assert e.shape == (2, 6, 6) and m.sum(1).tolist() == [6, 3]
        row = e[0]
        assert torch.allclose(row[2], torch.full((6,), 9.0)) and torch.allclose(row[3], torch.full((6,), 9.0))
        assert lab[0].tolist() == [-100] * 5 + [4]
        pad = slice(0, 3) if left else slice(3, 6)
        assert (m[1, pad] == 0).all() and (lab[1, pad] == -100).all() and (e[1, pad] == 0).all()


def test_knowledge_gradient_reaches_projector():
    from alexandria.ws6.common import assemble, build_batch, make_projector
    emb = torch.nn.Embedding(20, 8)
    emb.weight.requires_grad_(False)
    proj = make_projector(16, 8, 2, 8)
    kn = proj(torch.randn(1, 16))
    e, _, _ = build_batch([assemble([1], 2, [2], [3])], [kn[0]], emb, pad_left=False)
    e.sum().backward()
    assert proj.net[0].weight.grad is not None and proj.net[0].weight.grad.abs().sum() > 0


def test_split_by_paper_and_exclusion(tmp_path):
    from alexandria.ws6.papers import excluded_papers

    class Cfg:
        def stage_dir(self, ws, st):
            return tmp_path / ws / st
    (tmp_path / "ws2" / "papers").mkdir(parents=True)
    (tmp_path / "ws3" / "benchmarks").mkdir(parents=True)
    pd.DataFrame({"paper_key": ["a", "b"]}).to_parquet(tmp_path / "ws2" / "papers" / "papers.parquet")
    pd.DataFrame({"gold_paper_key": ["b", "c", None]}).to_parquet(tmp_path / "ws3" / "benchmarks" / "items.parquet")
    ctx = type("C", (), {"cfg": Cfg()})()
    assert excluded_papers(ctx) == {"a", "b", "c"}


def test_retrieve_top_k():
    from alexandria.ws6.evaluate import retrieve
    c = np.eye(4, dtype=np.float32)
    q = np.array([[0, 0.9, 0.1, 0], [1, 0, 0, 0]], np.float32)
    assert retrieve(q, c, 2).tolist() == [[1, 2], [0, 1]] or retrieve(q, c, 2)[1, 0] == 0


def test_arm_table_paired_ci():
    from alexandria.ws6.results import arm_table
    rng = np.random.default_rng(0)
    rows = [{"set": "ws2", "arm": a, "cluster": f"p{i // 4}", "correct": int(rng.random() < p), "ctx_tokens": t,
             "attempted": 1, "em": 0, "gold_retrieved": np.nan}
            for a, p, t in (("closed_book", .05, 0), ("emb_k4", .2, 4), ("ku_text", .6, 3000)) for i in range(120)]
    tbl, boot = arm_table(pd.DataFrame(rows), 300)
    t = tbl.set_index("arm")
    assert t.loc["ku_text", "mean"] > t.loc["emb_k4", "mean"] > t.loc["closed_book", "mean"]
    assert t.loc["emb_k4", "tokens"] == 4 and (tbl["lo"] <= tbl["mean"]).all() and (tbl["mean"] <= tbl["hi"]).all()
    assert len(boot[("ws2", "emb_k4")]) == 300
