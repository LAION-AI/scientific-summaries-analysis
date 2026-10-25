"""X6: token & compute efficiency of KUs at inference and in training. Importing registers every stage, in pipeline
order: inference -> rag-budget -> train-efficiency -> storage -> results -> report."""
from . import inference, rag, train, storage, results  # noqa: F401
from ..report import register_report

register_report("x6", "X6", "Token & compute efficiency: tokens, latency, GPU-seconds, $ and energy per correct answer, "
                "iso-accuracy, RAG token budgets and training-token efficiency of KUs vs full text")
