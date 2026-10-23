"""Pydantic models of one canonical record; `Summary.model_json_schema()` is the published JSON Schema."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(description="One claim the paper makes, in one or two sentences.")
    supporting_evidence: str = Field("", description="Evidence in the paper that supports the claim.")
    contradicting_evidence: str = Field("", description="Evidence in the paper that limits or contradicts it.")
    implications: str = Field("", description="What follows if the claim holds.")


class Summary(BaseModel):
    """The 18 LLM-written fields. Text fields are None when empty or a placeholder ("N/A", "[Unknown]")."""
    model_config = ConfigDict(extra="forbid")
    summary_title: str | None = None
    summary_authors: str | None = None
    field_subfield: str | None = None
    type_of_paper: str | None = None
    executive_summary: str | None = None
    research_context: str | None = None
    research_question_hypothesis: str | None = None
    methodological_details: str | None = None
    procedures_architectures: str | None = None
    key_results: str | None = None
    interpretation_implications: str | None = None
    contradictions_limitations: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    data_code_availability: str | None = None
    robustness_ablation_notes: str | None = None
    ethical_considerations: str | None = None
    key_figures_tables: str | None = None
    three_takeaways: str | None = None


class Provenance(BaseModel):
    file: str | None = None           # repo-relative release shard
    row: int | None = None            # row inside it
    group: str | None = None          # arxiv/new, arxiv/old, arxiv/titled, pubmed, ...


class CanonicalRecord(BaseModel):
    """What `load_summary` returns: identifiers, the clean summary, metadata, and every issue found on the way."""
    model_config = ConfigDict(extra="forbid")
    paper_id: str | None
    provenance: Provenance
    summary: Summary
    metadata: dict = Field(default_factory=dict, description="source_*, oa_*, generation metadata (unchanged)")
    extras: dict = Field(default_factory=dict, description="off-schema / unknown keys, never merged")
    issues: list[str] = Field(default_factory=list, description="e.g. 'claims:bad_json', 'key_results:placeholder'")
