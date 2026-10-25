"""X1: retrieval evaluation of the indices. Importing registers every stage, in pipeline order:
benchmarks -> corpus -> index -> search (+ ws3-diagnosis) -> results -> report."""
from . import benchmarks, corpus, index, diagnosis, results  # noqa: F401
from ..report import register_report

register_report("x1", "X1", "Retrieval evaluation: gold-paper retrieval on LitSearch / CiteME / SciFact, BM25 vs dense vs RRF, PQ48 loss, latency")
