"""The release's column layout: 44 columns = 7 source metadata + 18 LLM summary fields + 3 generation metadata
+ 13 OpenAlex + 3 text columns (laion/Scientific-Summaries @ 232547b7)."""
from __future__ import annotations

SOURCE_METADATA = ["paper_id", "subset", "source_title", "source_authors", "source_year", "source_venue", "source_doi"]

# The 18 fields the summarizer LLM writes (README: "18-field structured summary").
SUMMARY_FIELDS = [
    "summary_title", "summary_authors", "field_subfield", "type_of_paper",
    "executive_summary", "research_context", "research_question_hypothesis", "methodological_details",
    "procedures_architectures", "key_results", "interpretation_implications", "contradictions_limitations",
    "claims", "data_code_availability", "robustness_ablation_notes", "ethical_considerations",
    "key_figures_tables", "three_takeaways",
]
# Free-text prose fields (the 18 minus the bibliographic labels and the JSON `claims` field).
PROSE_FIELDS = [f for f in SUMMARY_FIELDS
                if f not in ("summary_title", "summary_authors", "field_subfield", "type_of_paper", "claims")]
# Fields that are legitimately empty for many papers (the prompt asks for them "if any").
OPTIONAL_FIELDS = ["data_code_availability", "robustness_ablation_notes", "ethical_considerations"]

GENERATION_METADATA = ["summary_model", "summary_tokens_in", "summary_tokens_out"]
OPENALEX = ["openalex_id", "oa_doi", "oa_title", "oa_type", "oa_year", "oa_cited_by_count", "oa_language",
            "oa_is_retracted", "oa_is_open_access", "oa_authorships", "oa_referenced_works", "match_method",
            "has_openalex_match"]
TEXT = ["text_length", "text_sanitized", "text_raw"]

CANONICAL_COLUMNS = SOURCE_METADATA + SUMMARY_FIELDS + GENERATION_METADATA + OPENALEX + TEXT
assert len(CANONICAL_COLUMNS) == 44, len(CANONICAL_COLUMNS)

CLAIM_KEYS = ("description", "supporting_evidence", "contradicting_evidence", "implications")
