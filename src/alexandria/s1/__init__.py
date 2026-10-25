"""S1: safety checks and evaluation of the dataset and the Alexandria skill (defensive; ids and rates only).
pii, injection, retractions (CPU) -> guard, dual-use, injection-live, uplift (GPU) -> results -> report."""
from . import scan, evals, results  # noqa: F401
from ..report import register_report

register_report("s1", "S1", "Safety checks and evaluation: PII, harmful / dual-use content, prompt injection, retractions, uplift")
