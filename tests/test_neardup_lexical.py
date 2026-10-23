"""Unit tests for the pure helpers of ws1.neardup and ws1.lexical (no data needed)."""
import numpy as np
import pytest

from alexandria.ws1.lexical import (claims_text, entity_support, field_measures, kgram_set, longest_span,
                                    number_support)
from alexandria.ws1.neardup import band_hashes, connected_components, estimated_jaccard, shingles


# ---------------------------------------------------------------- lexical: spans
def toks(s):
    return s.split()


def test_longest_span_exact_copy():
    src = toks("a b c d e f g h i j")
    assert longest_span(toks("x c d e f y"), kgram_set(src, 3), 3) == 4      # "c d e f"


def test_longest_span_whole_field_copied():
    src = toks("one two three four five six")
    assert longest_span(src, kgram_set(src, 2), 2) == 6


def test_longest_span_shorter_than_k_is_zero():
    src = toks("a b c d e")
    assert longest_span(toks("a b"), kgram_set(src, 3), 3) == 0
    assert longest_span(toks("x y z w"), kgram_set(src, 3), 3) == 0


def test_longest_span_picks_longest_of_several_runs():
    src = toks("a b c d e f g h")
    field = toks("a b c q e f g h")                                         # runs "a b c" and "e f g h"
    assert longest_span(field, kgram_set(src, 2), 2) == 4


def test_longest_span_empty():
    assert longest_span([], kgram_set(toks("a b c"), 2), 2) == 0
    assert longest_span(toks("a b c"), set(), 2) == 0


def test_field_measures_empty_field_is_null():
    m = field_measures("", "source text", set(), set(), 8)
    assert m["lcs"] is None and m["tok"] == 0


# ---------------------------------------------------------------- lexical: precision
def test_number_support_normalises_formats():
    src_nums = {"1200", "0.5", "95.3"}
    n, ok = number_support("We used 1,200 samples, p = 0.50, accuracy 95.3% and 42 layers.", src_nums)
    assert (n, ok) == (4, 3)                                                # 42 is not in the source


def test_number_support_ignores_single_digits():
    assert number_support("3 stages and 7 steps", set()) == (0, 0)


def test_entity_support_case_sensitive():
    n, ok = entity_support("We fine-tune BERT and ResNet-50 on CIFAR10.", "uses bert and ResNet-50 on CIFAR10")
    assert n == 3 and ok == 2                                               # 'BERT' absent (only 'bert')


def test_claims_text_joins_parts():
    raw = '[{"description": "A holds.", "supporting_evidence": "Table 2.", "contradicting_evidence": "", "implications": "B."}]'
    assert claims_text(raw) == "A holds. Table 2. B."
    assert claims_text("not json") == ""


# ---------------------------------------------------------------- neardup
def test_shingles():
    assert shingles("", 3) == set()
    assert shingles("Hello world", 5) == {b"hello world"}
    assert shingles("a b c d", 3) == {b"a b c", b"b c d"}


def test_band_hashes_shape_and_determinism():
    sig = np.arange(128, dtype=np.uint32)
    h = band_hashes(sig, 16)
    assert len(h) == 16 and h == band_hashes(sig.copy(), 16)
    assert all(0 <= x < 2**64 for x in h)
    with pytest.raises(ValueError):
        band_hashes(sig, 7)


def test_band_hashes_identical_band_collides():
    a = np.arange(128, dtype=np.uint32)
    b = a.copy()
    b[100:] = 0                                                             # only the last bands differ
    ha, hb = band_hashes(a, 16), band_hashes(b, 16)
    assert ha[:12] == hb[:12] and ha[12:] != hb[12:]


def test_estimated_jaccard():
    assert estimated_jaccard(np.array([1, 2, 3, 4]), np.array([1, 2, 0, 0])) == 0.5


def test_connected_components():
    nodes, labels = connected_components(np.array([1, 2, 10]), np.array([2, 3, 11]))
    comp = {int(n): int(l) for n, l in zip(nodes, labels)}
    assert comp[1] == comp[2] == comp[3] and comp[10] == comp[11] and comp[1] != comp[10]
    assert set(comp) == {1, 2, 3, 10, 11}


def test_connected_components_empty():
    nodes, labels = connected_components(np.array([], np.int64), np.array([], np.int64))
    assert nodes.size == 0 and labels.size == 0


def test_minhash_similar_texts_share_bands():
    from datasketch import MinHash
    a = "the quick brown fox jumps over the lazy dog " * 5 + "near the river bank today"
    b = "the quick brown fox jumps over the lazy dog " * 5 + "near the river bank tonight"
    ma, mb = MinHash.bulk([shingles(a, 5), shingles(b, 5)], num_perm=128, seed=1)
    sa, sb = ma.hashvalues.astype(np.uint32), mb.hashvalues.astype(np.uint32)
    assert estimated_jaccard(sa, sb) > 0.6
    assert set(band_hashes(sa, 16)) & set(band_hashes(sb, 16))
