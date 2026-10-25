"""X4: field-level and cross-paper queries over the KU corpus. Importing registers every stage, in pipeline order:
items -> answer -> grade -> results -> report."""
from . import items, answer, results  # noqa: F401
from ..report import register_report

register_report("x4", "X4", "Field-level and cross-paper questions: what a structured corpus makes possible")
