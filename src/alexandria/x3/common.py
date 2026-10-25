"""X3 shared pieces: the item table, answer parsing for free-text system outputs, helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..runtime import Context, stage

SQAB = {"sqab-scifact": ("scifact_test.jsonl", ["true", "false"]), "sqab-pubmed": ("pubmed_test.jsonl", ["yes", "no"])}


def setting(ctx: Context, key: str, default: Any) -> Any:
    if ctx.cfg.profile == "debug":
        return ctx.cfg.profile_settings.get(f"x3_{key}", default)
    return default


def stage_table(ctx: Context, stage_name: str, name: str, ws: str = "x3") -> pd.DataFrame:
    p = ctx.cfg.stage_dir(ws, stage_name) / name
    if not p.exists():
        raise FileNotFoundError(f"{ws}/{stage_name} has no {name}; run that stage first")
    return pd.read_parquet(p)


def sqab_items(root: Path, bench: str) -> list[dict]:
    fname, labels = SQAB[bench]
    rows = []
    for i, line in enumerate((root / "scholarqabench" / fname).read_text().splitlines()):
        d = json.loads(line)
        q = (f"Is this scientific claim true or false? Claim: {d['input']}" if bench == "sqab-scifact" else d["input"])
        rows.append({"bench": bench, "item_id": f"{bench}:{i}", "question": q, "answer": str(d["answer"]).lower(),
                     "answer_type": "label", "options": labels, "cluster": f"{bench}:{i}", "gold_paper_key": None,
                     "category": bench, "judge": "exact", "abstain_option": None,
                     "gold_ctx": json.dumps(d.get("gold_ctx") or [], ensure_ascii=False)})
    return rows


def parse_answer(text: str | None, item: dict) -> str:
    """The label / option letter / short answer in a free-text system output."""
    t = " ".join((text or "").split())
    if item["answer_type"] == "short":
        return t
    low = t.lower()
    if item["answer_type"] == "label":
        hits = [(low.find(o), o) for o in item["options"] if re.search(rf"\b{re.escape(o)}\b", low)]
        return min(hits)[1] if hits else ""
    letters = item["options"]                                     # multiple choice: a standalone option letter
    m = re.search(r"\b(?:answer(?: is)?|option)\s*[:\-]?\s*\(?([A-Z])\)?\b", t) or re.search(r"^\(?([A-Z])[\).:\s]", t) \
        or re.search(r"\b([A-Z])\b", t)
    if m and m.group(1) in letters:
        return m.group(1)
    if re.search(r"insufficient|cannot answer|not enough information", low) and item.get("abstain_option"):
        return item["abstain_option"]
    return ""


@stage("x3", "items", config_keys=("x3.items",),
       help="shared items: LitQA2 + QA bench (same ids as WS3) and ScholarQABench SciFact / PubMedQA (with gold context)")
def run_items(ctx: Context) -> dict:
    cfg = ctx.section("x3.items")
    n = setting(ctx, "per_bench", cfg.get("per_bench"))
    seed = int(cfg.get("seed", 0))
    ws3 = pd.read_parquet(ctx.cfg.stage_dir("ws3", "benchmarks") / "items.parquet")
    frames = []
    for b in cfg.get("benchmarks", []):
        if b in SQAB:
            df = pd.DataFrame(sqab_items(Path(ctx.cfg.site.store) / "benchmarks", b))
        else:
            df = ws3[ws3["bench"] == b].copy()
            df["gold_ctx"] = None
        frames.append(df.sample(n=min(int(n), len(df)), random_state=seed) if n else df)
    items = pd.concat(frames, ignore_index=True)
    items["options"] = [list(o) if o is not None else [] for o in items["options"]]
    items.to_parquet(ctx.out / "items.parquet", index=False)
    by = items.groupby("bench").size()
    ctx.publish("X3 shared benchmark items", [{"type": "kpis", "items": [
        *[{"label": b, "value": int(v), "fmt": "int"} for b, v in by.items()],
        {"label": "ScholarQABench source", "value": "AkariAsai/ScholarQABench (single-paper tasks)"}]}],
        summary=", ".join(f"{b} {v}" for b, v in by.items()))
    return {k: int(v) for k, v in by.items()}
