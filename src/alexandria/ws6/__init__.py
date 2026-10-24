"""WS6: embeddings as direct input (xRAG-style knowledge tokens; the extreme-compression point).
papers -> qa -> embed -> train-projector -> eval -> grade -> results -> report."""
from . import papers, embed, train, evaluate, results  # noqa: F401
from ..report import register_report

register_report("ws6", "WS6", "Embeddings as input: KU embeddings as k knowledge tokens for a frozen LLM vs text")
