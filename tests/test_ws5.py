"""WS5 pieces that need no GPU / API: KU schema, validation, verbatim gate, run grid, compliance guard."""
import json

import pytest

from alexandria.schema import SUMMARY_FIELDS

SOURCE = ("We measured the thermal conductivity of suspended single layer graphene using confocal Raman thermometry "
          "and found values near five thousand watts per metre kelvin at room temperature in every sample we tested.")


def _ku(**over):
    ku = {f: "" for f in SUMMARY_FIELDS}
    ku.update(summary_title="Graphene heat transport", executive_summary="Graphene conducts heat very well.",
              research_context="Heat removal matters.", research_question_hypothesis="How well does graphene conduct heat?",
              methodological_details="Raman thermometry.", key_results="Conductivity is about 5000 W/mK. It is high.",
              interpretation_implications="Useful for chips.", contradictions_limitations="Few samples.",
              three_takeaways="One. Two. Three.",
              claims=[{"description": "Graphene beats copper.", "supporting_evidence": "Raman data",
                       "contradicting_evidence": "", "implications": "heat sinks"}])
    ku.update(over)
    return ku


def test_ku_schema_covers_the_18_fields():
    from alexandria.ws5.common import ku_schema
    s = ku_schema()
    assert set(s["required"]) == set(SUMMARY_FIELDS) and s["properties"]["claims"]["type"] == "array"
    assert s["additionalProperties"] is False


def test_schema_check():
    from alexandria.ws5.common import parse_ku, schema_check
    req = ["summary_title", "executive_summary", "claims", "key_results"]
    good = schema_check(parse_ku("<think>x</think>" + json.dumps(_ku())), req)
    assert good["valid"] and good["filled"] > 0.5
    bad = schema_check(_ku(key_results="N/A", extra="x"), req)
    assert not bad["valid"] and "missing:key_results" in bad["issues"] and "extra_key:extra" in bad["issues"]
    assert schema_check(parse_ku("not json"), req)["issues"] == "unparsable"


def test_verbatim_gate():
    from alexandria.ws5.common import overlap
    clean = overlap(_ku(), SOURCE)
    copied = overlap(_ku(key_results="We measured the thermal conductivity of suspended single layer graphene using "
                                     "confocal Raman thermometry."), SOURCE)
    assert clean["max_span"] < 8 and copied["max_span"] >= 12 and copied["copied_kgram_share"] > clean["copied_kgram_share"]


def test_statements_for_the_judge():
    from alexandria.ws5.evaluate import statements
    st = statements(_ku(), 10)
    assert st[0] == "Graphene beats copper." and "Conductivity is about 5000 W/mK." in st and "It is high." not in st


class _Ctx:
    def __init__(self, raw, profile="full", settings=None):
        self.cfg = type("C", (), {"raw": raw, "profile": profile, "profile_settings": settings or {}})()
        self.written = {}

    def section(self, dotted):
        cur = self.cfg.raw
        for k in dotted.split("."):
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur

    def write_json(self, name, obj):
        self.written[name] = obj


def test_learning_curve_grid_and_debug_override():
    from alexandria.ws5.students import runs
    raw = {"ws5": {"train": {"students": ["a", "b"], "methods": ["lora", "full"], "sizes": [250, 500]}}}
    assert len(runs(_Ctx(raw))) == 8
    dbg = runs(_Ctx(raw, "debug", {"ws5_students": ["tiny"], "ws5_sizes": [8]}))
    assert [r["run"] for r in dbg] == ["tiny__lora__n8", "tiny__full__n8"]


def test_compliance_guard_blocks_large_unreviewed_teacher_jobs():
    from alexandria.ws5.teacher import _anthropic_batch
    ctx = _Ctx({"ws5": {"compliance": {"terms_checked": False, "max_unreviewed_requests": 2}}})
    with pytest.raises(PermissionError, match="terms"):
        _anthropic_batch(ctx, {}, [{"paper_key": str(i), "source": "x"} for i in range(3)])


def test_bibliographic_fields_do_not_trip_the_verbatim_gate():
    from alexandria.ws5.common import overlap
    title = "We measured the thermal conductivity of suspended single layer graphene using confocal Raman thermometry"
    assert overlap(_ku(summary_title=title), SOURCE)["max_span"] < 8
