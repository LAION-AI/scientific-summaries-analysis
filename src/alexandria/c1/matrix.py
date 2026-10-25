"""Stage `matrix` (C1, CPU): the regulatory & ethics compliance matrix.

Each requirement in `c1.requirements` (EU AI Act, GDPR, EU copyright / TDM, EU Trustworthy-AI guidelines / ALTAI,
Council of Europe AI Convention, OECD, UNESCO, NIST AI RMF) is scored against the evidence map:
met (all `met` conditions hold) > partial (all `partial` conditions hold) > gap; "unknown" when a needed key was not
measured. `legal: true` items always carry "legal review" on top of the status. This is an evidence matrix for the
authors, a DPO and a lawyer, not legal advice. Outputs: matrix.csv, matrix.json, compliance_table.tex (paper appendix).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..runtime import Context, log, stage

STATUSES = ("met", "partial", "gap", "unknown")
OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
       "eq": lambda a, b: a == b, "true": lambda a, _: bool(a) is True, "false": lambda a, _: bool(a) is False,
       "exists": lambda a, _: a is not None}


def check(cond: dict, ev: dict[str, Any]) -> bool | None:
    """True / False, or None when the key was not measured."""
    v = ev.get(cond["key"])
    if v is None:
        return None
    op = cond["op"]
    op = {True: "true", False: "false"}.get(op, op) if isinstance(op, bool) else op     # YAML reads `op: true` as a bool
    return bool(OPS[op](v, cond.get("value")))


def score(req: dict, ev: dict[str, Any]) -> tuple[str, list[str]]:
    used = [c["key"] for c in req.get("met", []) + req.get("partial", [])]
    met = [check(c, ev) for c in req.get("met", [])]
    if met and all(m is True for m in met):
        return "met", used
    part = [check(c, ev) for c in req.get("partial", [])]
    if part and all(p is True for p in part):
        return "partial", used
    if any(m is None for m in met) and not any(m is False for m in met):
        return "unknown", used
    return "gap", used


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2%}" if 0 <= v <= 1 else f"{v:,.2f}"
    return str(v)


def build(reqs: list[dict], ev: dict[str, Any], after: dict[str, Any] | None = None) -> pd.DataFrame:
    """One row per requirement; `after` = the evidence once the review decisions are applied (status_after)."""
    rows = []
    for r in reqs:
        st, keys = score(r, ev)
        st_after = score(r, {**ev, **after})[0] if after is not None else st
        rows.append({"id": r["id"], "framework": r["framework"], "ref": r["ref"], "requirement": r["requirement"],
                     "status": st, "status_after": st_after, "legal_review": bool(r.get("legal")),
                     "evidence": "; ".join(f"{k} = {_fmt(ev.get(k))}" for k in dict.fromkeys(keys)) or "documentation",
                     "action": "" if st_after == "met" else r.get("action", "")})
    return pd.DataFrame(rows)


def latex(df: pd.DataFrame) -> str:
    esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")  # noqa: E731
    lines = [r"\begin{tabular}{@{}llp{6.2cm}l@{}}", r"\toprule", r"Framework & Provision & Requirement & Status \\", r"\midrule"]
    for r in df.itertuples():
        st = r.status_after + (r"$^\dagger$" if r.legal_review else "")
        lines.append(f"{esc(r.framework)} & {esc(r.ref)} & {esc(r.requirement)} & {st} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"% $^\dagger$ needs legal review whatever the evidence says."]
    return "\n".join(lines) + "\n"


@stage("c1", "matrix", needs=("evidence", "review"), config_keys=("c1.requirements",),
       help="score every AI Act / GDPR / DSM / HLEG-ALTAI / CoE / OECD / UNESCO / NIST requirement against the evidence, "
            "now and after the review decisions")
def run(ctx: Context) -> dict:
    ev = json.loads((ctx.input_dir("evidence") / "evidence.json").read_text())["evidence"]
    rp = ctx.cfg.stage_dir("c1", "review") / "review.json"
    rv = json.loads(rp.read_text()) if rp.exists() else {"now": {}, "after": None}
    ev = {**ev, **rv["now"]}
    df = build(ctx.section("c1").get("requirements", []), ev, rv["after"])
    df.to_csv(ctx.out / "matrix.csv", index=False)
    (ctx.out / "compliance_table.tex").write_text(latex(df))
    counts = {s: int((df["status"] == s).sum()) for s in STATUSES}
    after = {s: int((df["status_after"] == s).sum()) for s in STATUSES}
    by_fw = df.groupby("framework", sort=False)["status_after"].value_counts().unstack(fill_value=0).reindex(columns=STATUSES, fill_value=0)
    ctx.write_json("matrix.json", {"counts": counts, "counts_after_review": after, "legal_review": int(df["legal_review"].sum()),
                                   "by_framework": by_fw.reset_index().to_dict("records"), "rows": df.to_dict("records")})
    log(f"matrix: now {counts}, after review {after}")
    n = len(df)
    open_items = df[df["status_after"] != "met"]
    changed = df[df["status"] != df["status_after"]]
    panels = [
        {"type": "kpis", "headline": True, "items": [
            {"label": "met now", "value": counts["met"] / n if n else None, "fmt": "pct", "hint": f"{counts['met']} of {n}"},
            {"label": "met after review", "value": after["met"] / n if n else None, "fmt": "pct", "status": "good",
             "hint": f"{after['met']} of {n}, once the review decisions and redactions are applied"},
            {"label": "gaps after review", "value": after["gap"] + after["unknown"], "fmt": "int",
             "status": "bad" if after["gap"] else "good", "hint": f"now {counts['gap'] + counts['unknown']}; partial after {after['partial']}"},
            {"label": "need legal review", "value": int(df["legal_review"].sum()), "fmt": "int", "status": "warn"}]},
        {"type": "markdown", "text":
            "**Evidence matrix, not legal advice.** Each provision is mapped to measured evidence from the pipelines "
            "(S1 safety, WS1 audit / licences / datasheet, X6 compute). `met` = all conditions hold on this profile's data; "
            "items marked *legal review* need a lawyer or the DPO whatever the evidence says. The AI Act's Art. 10 is binding "
            "for high-risk systems only; it is scored here as best practice for a training dataset. GPAI duties (Art. 53 / 55) "
            "fall on the model provider that trains on the data; the dataset supplies what they need."},
        {"type": "bar", "title": "Status now vs after the review", "labels": list(STATUSES),
         "series": [{"name": "now", "data": [counts[s] for s in STATUSES]},
                    {"name": "after review", "data": [after[s] for s in STATUSES]}]},
        {"type": "bar", "title": "Status by framework (after review)", "labels": list(by_fw.index), "stacked": True, "horizontal": True,
         "series": [{"name": s, "data": [int(v) for v in by_fw[s]]} for s in STATUSES]},
        {"type": "table", "title": "What the review changes",
         "columns": [{"key": "id", "label": "id"}, {"key": "ref", "label": "provision"}, {"key": "status", "label": "now"},
                     {"key": "status_after", "label": "after review"}, {"key": "requirement", "label": "requirement"}],
         "rows": changed.to_dict("records")},
        {"type": "table", "title": "Still open after the review: what people must do before the release",
         "columns": [{"key": "id", "label": "id"}, {"key": "ref", "label": "provision"}, {"key": "status_after", "label": "status"},
                     {"key": "legal_review", "label": "legal review"}, {"key": "action", "label": "action"}],
         "rows": open_items.to_dict("records"), "pageSize": 12},
        {"type": "table", "title": "Full compliance matrix",
         "columns": [{"key": "framework", "label": "framework"}, {"key": "ref", "label": "provision"},
                     {"key": "requirement", "label": "requirement"}, {"key": "status", "label": "now"},
                     {"key": "status_after", "label": "after review"}, {"key": "legal_review", "label": "legal review"}, {"key": "evidence", "label": "evidence"}],
         "rows": df.to_dict("records"), "pageSize": 10}]
    ctx.publish("C1 compliance matrix", panels,
                summary=f"now {counts['met']} met / {counts['partial']} partial / {counts['gap']} gaps; after review "
                        f"{after['met']} / {after['partial']} / {after['gap']} of {n}")
    return {"now": counts, "after_review": after}
