"""WS5: teacher distillation. Importing registers every stage, in pipeline order:
papers -> teacher -> validate -> train -> generate -> mcq -> judge -> answer -> results -> report."""
from . import teacher, students, evaluate  # noqa: F401
from ..report import register_report

register_report("ws5", "WS5", "Teacher distillation: student KUs vs teacher, released v2.0 and baselines on disjoint papers")
