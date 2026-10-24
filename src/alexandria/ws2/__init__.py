"""WS2: representation comparison on the QA bench. Importing registers every stage, in pipeline order:
papers -> questions -> ku -> contexts -> answer -> grade -> results -> report."""
from . import papers, questions, ku, contexts, answer, results  # noqa: F401
from ..report import register_report

register_report("ws2", "WS2", "Representation comparison on the QA bench: model alone vs full paper vs the paper's KU (raw / cleaned)")
