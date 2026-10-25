"""WS1: data quality and dataset description (+ WS1b content faithfulness). Importing registers every stage,
in pipeline order (the dashboard and `alexandria list` show them in this order)."""
from . import features, keys, lexical, neardup, quality, describe, reconcile, provenance, audit, licenses  # noqa: F401
from .faith import human, grounding, judge, abstract_support  # noqa: F401
from . import datasheet  # noqa: F401  (last: it cites every other stage's numbers)
from ..report import register_report

register_report("ws1", "WS1", "Data quality, dataset description and content faithfulness")   # last: the PDF
