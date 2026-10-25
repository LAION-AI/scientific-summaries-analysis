"""C1 compliance matrix: condition checks, status rules, health-identifier screen, config sanity."""
import yaml

from alexandria.c1.evidence import _countries, health_flags
from alexandria.c1.matrix import build, check, score
from alexandria.config import CONFIG_DIR


def test_check_ops_and_yaml_bool_op():
    ev = {"a": 0.01, "b": True}
    assert check({"key": "a", "op": "lt", "value": 0.05}, ev) is True
    assert check({"key": "b", "op": True}, ev) is True           # YAML `op: true` arrives as a bool
    assert check({"key": "missing", "op": "eq", "value": 0}, ev) is None


def test_score_met_partial_gap_unknown():
    req = {"met": [{"key": "r", "op": "lt", "value": 0.01}], "partial": [{"key": "r", "op": "lt", "value": 0.1}]}
    assert score(req, {"r": 0.0})[0] == "met"
    assert score(req, {"r": 0.05})[0] == "partial"
    assert score(req, {"r": 0.5})[0] == "gap"
    assert score({"met": [{"key": "x", "op": "true"}]}, {})[0] == "unknown"


def test_build_rows_carry_action_only_when_open():
    reqs = [{"id": "A", "framework": "F", "ref": "1", "requirement": "r", "met": [{"key": "k", "op": "true"}], "action": "do"},
            {"id": "B", "framework": "F", "ref": "2", "requirement": "r", "met": [{"key": "k", "op": "false"}], "action": "do",
             "legal": True}]
    df = build(reqs, {"k": True})
    assert df.set_index("id").loc["A", "action"] == "" and df.set_index("id").loc["B", "status"] == "gap"
    assert bool(df.set_index("id").loc["B", "legal_review"])


def test_health_flags_do_not_fire_on_methods():
    assert not health_flags({"methodological_details": "analysed by nano LC-MS/MS; a two-sample MR design"})["direct_id"]
    assert health_flags({"key_results": "Mrs. Smith was admitted on 03/04/2015"})["direct_id"]
    f = health_flags({"executive_summary": "We report a case of a 45-year-old woman with lupus."})
    assert f["quasi_id"] and f["case_report"]


def test_countries_from_openalex_authorships():
    assert _countries('[{"countries": ["DE"]}, {"countries": ["US", "DE"]}]') == {"DE", "US"}
    assert _countries(None) == set()


def test_config_requirements_well_formed():
    c = yaml.safe_load((CONFIG_DIR / "c1.yaml").read_text())["c1"]
    ids = [r["id"] for r in c["requirements"]]
    assert len(ids) == len(set(ids))
    for r in c["requirements"]:
        assert r["met"] and {"framework", "ref", "requirement"} <= set(r)
    assert {"GDPR", "EU AI Act"} <= {r["framework"] for r in c["requirements"]}


# ---------------------------------------------------------------- review + tdm
def test_fulltext_and_ku_decisions():
    from alexandria.c1.review import fulltext_decision, ku_decision
    assert fulltext_decision(True, "permissive", "none", False) == ("ship", "")
    assert fulltext_decision(True, "permissive", "robots_ai", False) == ("withhold", "tdm_reserved")
    assert fulltext_decision(True, None, None, False) == ("withhold", "no_licence_found")
    assert fulltext_decision(True, "nc_or_nd", None, False) == ("withhold", "nc_or_nd")
    assert fulltext_decision(False, "permissive", None, False)[0] == "none"
    assert ku_decision(True, False, 0) == "drop" and ku_decision(False, True, 2) == "hold"
    assert ku_decision(False, False, 1) == "ship_redacted" and ku_decision(False, False, 0) == "ship"


def test_pii_redact_leaves_no_hits():
    from alexandria.s1.common import pii_hits, pii_redact
    text, cats = pii_redact("Contact jane.doe@uni.example or 192.168.1.20 for data.")
    assert cats == {"email": 1, "ip_address": 1} and not pii_hits(text) and "[REDACTED:email]" in text


def test_signed_documents(tmp_path):
    from alexandria.c1.evidence import signed
    p = tmp_path / "DPIA.md"
    p.write_text("Status: DRAFT\nApproved-by: \n")
    assert not signed(p)
    p.write_text("Approved-by: J. Smith (DPO), 2026-11-02\n")
    assert signed(p) and not signed(tmp_path / "missing.md")


def test_tdmrep_and_robots():
    from alexandria.c1.tdm import combine, meta_reservation, norm_doi, record_status, robots_blocked, tdmrep_match
    rules = [{"location": "/*", "tdm-reservation": 0}, {"location": "/articles/*", "tdm-reservation": 1}]
    assert tdmrep_match(rules, "/articles/x") == 1 and tdmrep_match(rules, "/about") == 0 and tdmrep_match([], "/") is None
    assert meta_reservation('<meta name="tdm-reservation" content="1">') == 1
    assert robots_blocked("User-agent: GPTBot\nDisallow: /\n", "https://p.example/a", ["GPTBot", "CCBot"]) == ["GPTBot"]
    assert norm_doi("https://doi.org/10.1038/X") == "10.1038/x" and norm_doi("abc") is None
    base = {"resolved": True, "tdmrep": None, "header": None, "meta": None, "robots_ai": ""}
    assert record_status(combine([base, {**base, "robots_ai": "CCBot"}])) == "robots_ai"
    assert record_status(combine([{**base, "tdmrep": 1}])) == "reserved" and record_status({}) == "unknown"


def test_after_review_status_moves():
    reqs = [{"id": "L", "framework": "F", "ref": "1", "requirement": "r",
             "met": [{"key": "licenses.fulltext_nonpermissive_share", "op": "eq", "value": 0}]}]
    df = build(reqs, {"licenses.fulltext_nonpermissive_share": 0.87}, {"licenses.fulltext_nonpermissive_share": 0.0})
    assert df.loc[0, "status"] == "gap" and df.loc[0, "status_after"] == "met" and df.loc[0, "action"] == ""
