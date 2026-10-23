"""Canonical schema of laion/Scientific-Summaries and the shared normalizer used by every workstream."""
from .aliases import canon, discover_candidates, resolve
from .fields import CANONICAL_COLUMNS, CLAIM_KEYS, OPTIONAL_FIELDS, PROSE_FIELDS, SUMMARY_FIELDS
from .model import CanonicalRecord, Claim, Summary
from .normalize import load_summary, parse_claims

__all__ = ["CANONICAL_COLUMNS", "CLAIM_KEYS", "OPTIONAL_FIELDS", "PROSE_FIELDS", "SUMMARY_FIELDS",
           "CanonicalRecord", "Claim", "Summary", "canon", "discover_candidates", "load_summary", "parse_claims",
           "resolve"]
