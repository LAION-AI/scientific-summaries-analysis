"""C1: regulatory & ethics compliance (EU AI Act, GDPR, EU copyright / TDM, EU Trustworthy-AI guidelines / ALTAI,
Council of Europe AI Convention, OECD, UNESCO, NIST AI RMF). Importing registers every stage, in pipeline order:
tdm (login node) -> evidence -> review -> matrix -> report."""
from . import tdm, evidence, review, matrix  # noqa: F401
from ..report import register_report

register_report("c1", "C1", "Regulatory & ethics compliance: an evidence matrix for the EU AI Act, GDPR, EU copyright "
                "(TDM), EU Trustworthy-AI guidelines (ALTAI) and international AI frameworks. Not legal advice.")
