"""Unit tests for the shared core: config, schema/normalizer, text checks, feature helpers. No real data needed."""
import json

import pyarrow as pa
import pytest

from alexandria import textutil as tu
from alexandria.config import load_config, parse_override
from alexandria.data import Shard, cast_batch, partition
from alexandria.domains import clean_label, domain_of
from alexandria.schema import CANONICAL_COLUMNS, SUMMARY_FIELDS, load_summary, resolve
from alexandria.schema.aliases import discover_candidates
from alexandria.stats import wilson
from alexandria.ws1.features import arxiv_id, input_type, to_table, year_of


# ---------------------------------------------------------------- config
def test_config_interpolation_and_override(tmp_path):
    cfg = load_config("ws1", "debug", ["site.out_root=" + str(tmp_path), "ws1.checks.min_title_similarity=75"])
    assert str(cfg.site.data_root).endswith("Scientific-Summaries")
    assert "${" not in str(cfg.site.data_root)
    assert cfg.site.out_root == tmp_path
    assert cfg.raw["ws1"]["checks"]["min_title_similarity"] == 75
    assert cfg.raw["ws1"]["checks"]["lang_model"].endswith("lid.176.bin")
    assert cfg.stage_dir("ws1", "features") == tmp_path / "debug" / "ws1" / "features"


def test_parse_override_types():
    assert parse_override("a.b=3") == ("a.b", 3)
    assert parse_override("a=[1, 2]") == ("a", [1, 2])
    with pytest.raises(ValueError):
        parse_override("novalue")


def test_fingerprint_changes_with_config():
    a = load_config("ws1", "debug").fingerprint("ws1.checks")
    b = load_config("ws1", "debug", ["ws1.checks.min_title_similarity=1"]).fingerprint("ws1.checks")
    assert a != b


# ---------------------------------------------------------------- schema
def test_schema_counts():
    assert len(CANONICAL_COLUMNS) == 44
    assert len(SUMMARY_FIELDS) == 18


@pytest.mark.parametrize("name,expected,how", [
    ("key_results", "key_results", "exact"),
    ("Key Results", "key_results", "canon"),
    ("KEY-RESULTS", "key_results", "canon"),
    ("Hyp_", "research_question_hypothesis", "alias"),
    ("hypo", "research_question_hypothesis", "alias"),
    ("tittle", None, "unknown"),
])
def test_resolve_columns(name, expected, how):
    m = resolve(name)
    assert (m.key, m.how) == (expected, how)


def test_resolve_claim_keys():
    assert resolve("contradating_evidence", "claims").key == "contradicting_evidence"
    assert resolve("contradicting evidence", "claims").how == "canon"
    assert resolve("caveats", "claims").how == "off_schema"


def test_discover_candidates_proposes_but_does_not_accept():
    out = discover_candidates({"resarch_context": 5}, "columns")
    assert out[0]["resolution"] == "unknown"
    assert out[0]["candidates"][0]["canonical"] == "research_context"


def test_load_summary_repairs_and_records_issues():
    claims = [{"description": "X improves Y.", "supporting_evidence": "Table 2", "contradating_evidence": "",
               "implications": "Z", "caveats": "small n"}]
    rec = load_summary({"paper_id": "1", "Summary Title": "[Unknown]", "hypo": "Does X improve Y?",
                        "key_results": "  Y up by 3%.  ", "claims": json.dumps(claims), "weird": 1})
    s = rec["summary"]
    assert s["summary_title"] is None                       # placeholder nulled
    assert s["research_question_hypothesis"] == "Does X improve Y?"
    assert s["key_results"] == "Y up by 3%."
    assert s["claims"][0]["contradicting_evidence"] == ""
    assert rec["extras"]["weird"] == 1
    assert rec["extras"]["claims_extras"][0] == {"caveats": "small n"}
    issues = " ".join(rec["issues"])
    for expected in ("summary_title:placeholder", "key:alias=hypo", "claims:alias=contradating_evidence",
                     "claims:off_schema_key=caveats", "key:unknown=weird"):
        assert expected in issues


def test_load_summary_bad_claims_json():
    rec = load_summary({"claims": '[{"description": "a", '})
    assert rec["summary"]["claims"] == [] and "claims:bad_json" in rec["issues"]


def test_real_value_beats_empty_duplicate():
    rec = load_summary({"key_results": "", "Key Results": "real"})
    assert rec["summary"]["key_results"] == "real"


# ---------------------------------------------------------------- text checks
@pytest.mark.parametrize("text,status", [
    ("", "empty"), ("  N/A ", "placeholder"), ("[Unknown]", "placeholder"),
    ("The authors are not provided in the text.", "not_provided"),
    ("No ethical considerations are mentioned in the paper.", "na_statement"),
    ("The paper does not mention code availability.", "na_statement"),
    ("We propose a method that improves accuracy by 3%.", "ok"),
])
def test_value_status(text, status):
    assert tu.value_status(text) == status


def test_truncation_refusal_leaks():
    assert tu.ends_mid_sentence("The model was trained on a corpus of scientific papers and then evaluated on the")
    assert not tu.ends_mid_sentence("The model was trained on a corpus of scientific papers and evaluated.")
    assert tu.refusal("As an AI language model, I cannot provide a summary.")
    assert not tu.refusal("The AI model provides a summary.")
    assert tu.key_leak("key_results: accuracy rose") == ["key_results"]
    assert tu.key_leak("the key results show") == []
    assert tu.markup_leak('```json\n{"a": 1}')


def test_numbers_entities_names():
    assert tu.numbers("We used 1,200 samples, 0.50 dropout and 3 seeds") == {"1200", "0.5"}
    assert {"BERT", "ResNet-50", "GPT-4"} <= tu.entities("BERT and ResNet-50 beat GPT-4 here")
    assert tu.name_tokens("Jalal Sarabadani, Ali Naji (Univ. X)") == {"jalal", "sarabadani", "ali", "naji"}
    assert tu.name_tokens("Sarabadani, Jalal; Naji, Ali") == {"jalal", "sarabadani", "ali", "naji"}


def test_garbled_fraction():
    clean = "The results of this study show that the method is robust and can be applied to many problems. " * 10
    garbled = "ËÔ Ö ÐÐÝ ×ÝÑÑ ØÖ ÑÓ Ð× Û Ø ÔÖ ××ÙÖ × Ô Ö Ø Ò ÜÔ Ò× ÓÒ ÖÓÑ ÓÒØÖ Ø ÓÒ " * 10
    assert tu.garbled_fraction(clean) < 0.05
    assert tu.garbled_fraction(garbled) > 0.5
    assert tu.garbled_fraction("short") is None


# ---------------------------------------------------------------- features helpers
def test_arxiv_id_and_year():
    assert arxiv_id("1901.06471") == "1901.06471"
    assert arxiv_id("hep-th/9901001v2") == "hep-th/9901001"
    assert arxiv_id("/mnt/sda/x.json", "10.48550/arXiv.2101.00001") == "2101.00001"
    assert arxiv_id("/mnt/sda/x.json") is None
    assert year_of({"source_year": "", "oa_year": 2019.0}, None) == (2019, "oa_year")
    assert year_of({}, "1901.06471") == (2019, "arxiv_id")
    assert year_of({}, "hep-th/9901001") == (1999, "arxiv_id")
    assert year_of({}, None) == (None, None)


def test_input_type_bins():
    bins = {"metadata_only": 0, "abstract_only": 150, "short_text": 700, "full_text": 2500}
    assert input_type(2000, 1950, bins) == ("metadata_only", 50)
    assert input_type(2300, 1950, bins) == ("abstract_only", 350)
    assert input_type(20000, 1950, bins)[0] == "full_text"
    assert input_type(None, 1950, bins) == ("unknown", None)


def test_to_table_stable_types_for_all_null_columns():
    t = to_table([{"refusal_fields": None, "title_sim": None, "words__key_results": 3, "refusal": False}])
    assert t.schema.field("refusal_fields").type == pa.list_(pa.string())
    assert t.schema.field("title_sim").type == pa.float64()
    assert t.schema.field("words__key_results").type == pa.int64()
    assert t.schema.field("refusal").type == pa.bool_()


# ---------------------------------------------------------------- data
def test_partition_is_disjoint_complete_and_balanced():
    shards = [Shard("g", f"data/x/{i}.parquet", None, size=(i % 7 + 1) * 100) for i in range(50)]
    parts = [partition(shards, i, 4) for i in range(4)]
    rels = [s.rel for p in parts for s in p]
    assert sorted(rels) == sorted(s.rel for s in shards)
    loads = [sum(s.size for s in p) for p in parts]
    assert max(loads) - min(loads) <= 700


def test_cast_batch_unifies_types():
    schema = pa.schema([("summary_tokens_in", pa.int64()), ("oa_is_retracted", pa.bool_()), ("oa_year", pa.float64())])
    t = pa.table({"summary_tokens_in": pa.array([1.6, None]), "oa_is_retracted": pa.array(["true", "false"]),
                  "oa_year": pa.array([None, None], pa.null())})
    out = cast_batch(t, schema)
    assert out.schema == schema
    assert out["summary_tokens_in"].to_pylist() == [2, None]
    assert out["oa_is_retracted"].to_pylist() == [True, False]


# ---------------------------------------------------------------- domains / stats
def test_domains_match_sci_jepa():
    sj = pytest.importorskip("sci_jepa.data.ku_schema")
    for label in ["Physics — Quantum Optics", "Computer Science — Machine Learning", "Astro-Physics",
                  "Medicine — Oncology", "Economics", "物理学", "", None, "Statistics — Bayesian"]:
        assert domain_of(label) == sj.domain_of(label), label


def test_clean_label_merges_variants():
    assert clean_label("Astro-Physics") == clean_label("astrophysics") == "astrophysics"
    assert clean_label("Accounting & Finance") == clean_label("Accounting and Finance")


def test_wilson():
    p, lo, hi = wilson(5, 100)
    assert p == 0.05 and 0.02 < lo < 0.05 < hi < 0.12
    assert wilson(0, 0)[0] != wilson(0, 0)[0]          # nan


# ---------------------------------------------------------------- length outliers (quality)
def test_length_outliers_and_documented():
    import numpy as np
    import pandas as pd

    from alexandria.ws1.quality import documented_vs_delivered, length_outliers
    rng = np.random.default_rng(0)
    w = rng.normal(200, 20, 200).round().astype(int)
    w[0], w[1] = 5, 3000                                          # one degenerate, one runaway field
    df = pd.DataFrame({"file": "f", "row": range(200), "input_type": "full_text", "words__executive_summary": w,
                       "status__executive_summary": "ok"})
    out = length_outliers(df, ["executive_summary"])
    assert out.loc[0, "len_short"] and out.loc[1, "len_long"]
    assert out["len_short"].sum() + out["len_long"].sum() <= 4    # the bulk of the distribution is not flagged
    assert "executive_summary" in out.loc[0, "len_fields"]
    rows = documented_vs_delivered(df, {"executive_summary": [400, 500]})
    assert rows[0]["below"] > 0.9 and rows[0]["median_vs_target"] < 0.6


def test_skip_status(tmp_path):
    from alexandria.runtime import Context, StageSpec, run_stage
    cfg = load_config("ws1", "debug", ["site.out_root=" + str(tmp_path)])
    spec = StageSpec("ws1", "t-skip", lambda ctx: ctx.skip("T", "off"), "test")
    assert run_stage(Context(cfg, spec)) == 0
    assert json.loads((tmp_path / "debug" / "ws1" / "t-skip" / "manifest.json").read_text())["status"] == "skipped"


def test_optout_list_drops_rows_and_keeps_release_row_numbers(tmp_path):
    import pyarrow as pa

    from alexandria.data import optout_ids, optout_mask
    f = tmp_path / "optout.txt"
    f.write_text("# takedown requests\n2401.00001\nhttps://doi.org/10.1234/ABC\n")
    ids = optout_ids(str(f))
    assert ids == {"2401.00001", "10.1234/abc"}
    tbl = pa.table({"paper_id": ["2401.00001", "2401.00002", "x"], "source_doi": [None, None, "10.1234/abc"],
                    "_row": [0, 1, 2]})
    kept = tbl.filter(optout_mask(tbl, ids))
    assert kept["_row"].to_pylist() == [1]
