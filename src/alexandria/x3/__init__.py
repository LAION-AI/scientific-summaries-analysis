"""X3: comparison with existing systems. Importing registers every stage, in pipeline order:
items -> alexandria / openscholar / pqa-corpus -> paperqa -> grade -> results -> report."""
from . import common, systems, results  # noqa: F401
from ..report import register_report

register_report("x3", "X3", "Comparison with existing systems: Alexandria vs OpenScholar and PaperQA2 on shared benchmarks")
