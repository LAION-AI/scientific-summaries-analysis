"""X5: knowledge conflict. Importing registers every stage, in pipeline order:
items -> counterfactual -> answer -> grade -> results -> report."""
from . import stages  # noqa: F401
from ..report import register_report

register_report("x5", "X5", "Knowledge conflict: do models follow the retrieved KU or their prior?")
