"""WS3: harder benchmarks via the Alexandria knowledge skill. Importing registers every stage, in pipeline order:
benchmarks -> index -> pilot -> agent -> grade -> results -> report."""
from . import benchmarks, index, agent, results  # noqa: F401
from ..report import register_report

register_report("ws3", "WS3", "Benchmarks via the Alexandria knowledge skill vs model alone, RAG, full text, summaries, KUs")
