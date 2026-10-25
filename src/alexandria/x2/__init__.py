"""X2: post-cutoff benchmark. Importing registers every stage, in pipeline order:
papers (login node) -> summarize -> questions -> verify -> index -> eval -> grade -> results -> report."""
from . import papers, questions, index, evaluate  # noqa: F401
from ..report import register_report

register_report("x2", "X2", "Post-cutoff benchmark: does the knowledge skill help when the model cannot know the answer?")
