"""Stages `human-sample` and `human-agreement`: the human faithfulness protocol of WS1b.

human-sample   stratified sample of papers (ws1b.human.strata, proportional with a floor per stratum) plus a small
               explicit stratum of papers whose source text is unreadable (source_garbled, a WS1 finding). Writes
                 packets.jsonl          one record per paper: source text, the configured fields, claims as items
                 annotation_template.csv one row per (paper, field / claim) item with empty label columns
                 ANNOTATION_GUIDELINES.md label definitions and examples
               ~double_annotated_share of the papers (deterministic) are marked for a second annotator.
human-agreement reads the filled-in sheets from <human-sample>/labels/*.csv (one or more per annotator), computes
               Krippendorff's alpha (nominal) on items labelled by >= 2 annotators, per field and overall, with a
               bootstrap CI over items, and writes gold.parquet (majority label per item; ties -> "disputed").
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...runtime import Context, atomic_write_parquet, log, stage
from ...stats import rate_row
from .common import HUMAN_OFF, LABELS, features_table, fetch_records, field_items, human_enabled, item_id, paper_key, \
    source_text, stable_rank, stratified_pick

CSV_COLUMNS = ["item_id", "paper_key", "paper_id", "file", "row", "double_annotated", "field", "item_index",
               "item_text", "label", "omission", "annotator_id", "notes"]

GUIDELINES = """# Annotation guidelines: faithfulness of structured paper summaries

You compare each summary item (one field of the summary, or one claim) with the **source paper text** in
`packets.jsonl` (field `source`). Judge only what the source says, not what you know about the topic.

## Labels (column `label`)

| label | use it when | example |
|---|---|---|
| `supported` | every factual statement in the item is stated in or directly follows from the source | Source: "accuracy rose from 81.2% to 86.5%". Item: "Accuracy improved by about 5 points." |
| `unsupported` | at least one statement is not in the source (invented number, method, dataset, finding, or a claim the source does not make), but nothing is contradicted | Item says "evaluated on ImageNet" and the source never mentions ImageNet. |
| `contradicted` | at least one statement conflicts with the source (wrong direction, wrong number, wrong method) | Source: "no significant effect (p = 0.41)". Item: "a significant effect was found". |

Rules:
* Label the **worst** statement in the item: one contradiction makes the item `contradicted`.
* Paraphrase, summarising and reasonable abstraction are fine (`supported`). Generic framing sentences
  ("This paper investigates ...") are `supported` when they match the paper's topic.
* An empty item cannot be labelled; leave it blank and write `empty` in `notes`.
* If the source text is unreadable (garbled characters), still label the item against what you can read: anything
  you cannot verify is `unsupported`, and write `garbled source` in `notes`.

## Omissions (column `omission`)

Free text: an important result, limitation or condition of the paper that this field should contain but does not
(e.g. "omits that the effect holds only for n < 100"). Leave empty when nothing important is missing.

## Workflow

1. Copy `annotation_template.csv` to `labels/<your_id>.csv` and fill in `annotator_id` on every row you label.
2. Rows with `double_annotated = 1` are labelled by two annotators independently; do not discuss them until both
   are done. Agreement is measured on these rows (Krippendorff's alpha).
3. Keep the file as UTF-8 CSV; do not change `item_id`.
"""


@stage("ws1", "human-sample", needs=("features",), config_keys=("ws1b.human",),
       help="stratified paper sample + annotation packets for human faithfulness labels")
def run_sample(ctx: Context) -> dict:
    if not human_enabled(ctx):
        return ctx.skip("Human annotation sample", HUMAN_OFF)
    cfg = ctx.section("ws1b.human")
    n = int(cfg.get("n_papers", 500))
    seed = int(cfg.get("seed", 0))
    strata = list(cfg.get("strata", ["group", "input_type", "domain"]))
    fields = list(cfg.get("fields", []))
    share = float(cfg.get("double_annotated_share", 0.2))
    min_per = int(cfg.get("min_per_stratum", 3))
    garbled_share = float(cfg.get("garbled_share", 0.02))

    feats = features_table(ctx, ["file", "row", "paper_id", "group", "input_type", "domain", "has_text",
                                 "source_garbled"])
    usable = feats[feats["has_text"].fillna(False)]
    garbled = usable[usable["source_garbled"].fillna(False)]
    clean = usable[~usable["source_garbled"].fillna(False)]
    n_garbled = min(len(garbled), max(1, round(n * garbled_share))) if len(garbled) else 0
    pick = pd.concat([
        stratified_pick(clean, n - n_garbled, strata, seed, min_per).assign(stratum_kind="main"),
        stratified_pick(garbled, n_garbled, [], seed).assign(stratum_kind="garbled_source"),
    ])
    pick["paper_key"] = [paper_key(f, r) for f, r in zip(pick["file"], pick["row"], strict=True)]
    order = sorted(pick["paper_key"], key=lambda k: stable_rank(seed, "double", k))
    double = set(order[:max(1, round(len(order) * share))]) if len(order) else set()
    pick["double_annotated"] = pick["paper_key"].isin(double)
    log(f"human-sample: {len(pick)} papers ({n_garbled} garbled-source), {len(double)} double-annotated")

    recs = fetch_records(ctx.cfg, zip(pick["file"], pick["row"], strict=True),
                         ["paper_id", "summary_title", "text_raw", "text_sanitized", *fields])
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "labels").mkdir(exist_ok=True)
    n_items = 0
    with open(ctx.out / "packets.jsonl", "w") as fj, open(ctx.out / "annotation_template.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for p in pick.itertuples():
            rec = recs.get((p.file, int(p.row)))
            if rec is None:
                continue
            items = []
            for f in fields:
                texts = field_items(rec, f)
                for i, t in enumerate(texts):
                    iid = item_id(p.paper_key, f, i if f == "claims" else None)
                    items.append({"item_id": iid, "field": f, "item_index": i, "text": t})
                    w.writerow({"item_id": iid, "paper_key": p.paper_key, "paper_id": p.paper_id, "file": p.file,
                                "row": int(p.row), "double_annotated": int(p.double_annotated), "field": f,
                                "item_index": i, "item_text": t, "label": "", "omission": "", "annotator_id": "",
                                "notes": ""})
            n_items += len(items)
            fj.write(json.dumps({"paper_key": p.paper_key, "paper_id": p.paper_id, "file": p.file, "row": int(p.row),
                                 "group": p.group, "input_type": p.input_type, "domain": p.domain,
                                 "stratum_kind": p.stratum_kind, "double_annotated": bool(p.double_annotated),
                                 "title": rec.get("summary_title"), "source": source_text(rec), "items": items},
                                ensure_ascii=False) + "\n")
    (ctx.out / "ANNOTATION_GUIDELINES.md").write_text(GUIDELINES)
    atomic_write_parquet(ctx.out / "sample.parquet",
                         __import__("pyarrow").Table.from_pandas(pick.reset_index(drop=True), preserve_index=False))

    strata_cols = [c for c in strata if c in pick.columns]
    tab = (pick.groupby(["stratum_kind", *strata_cols]).size().reset_index(name="papers")
           .sort_values("papers", ascending=False))
    pop = usable.groupby(strata_cols).size().reset_index(name="population") if strata_cols else None
    if pop is not None:
        tab = tab.merge(pop, on=strata_cols, how="left")
    ctx.publish("Human annotation sample", [
        {"type": "kpis", "items": [
            {"label": "papers", "value": len(pick), "fmt": "int"},
            {"label": "items to label", "value": n_items, "fmt": "int"},
            {"label": "double-annotated papers", "value": len(double), "fmt": "int"},
            {"label": "garbled-source papers", "value": n_garbled, "fmt": "int"}]},
        {"type": "table", "title": "Strata", "hint": "proportional allocation with a floor per stratum",
         "columns": [{"key": "stratum_kind", "label": "kind"}] + [{"key": c, "label": c} for c in strata_cols]
         + [{"key": "papers", "label": "sampled", "fmt": "int"}, {"key": "population", "label": "eligible", "fmt": "int"}],
         "rows": tab.fillna("").to_dict("records"), "limit": 200},
        {"type": "markdown", "text": f"Packets: `{ctx.out / 'packets.jsonl'}`. Annotators copy "
                                     f"`annotation_template.csv` to `labels/<id>.csv`; then run `alexandria ws1 human-agreement`."},
    ], summary=f"{len(pick)} papers, {n_items} items")
    return {"papers": len(pick), "items": n_items, "double_annotated": len(double), "garbled": n_garbled}


# ---------------------------------------------------------------- agreement
def read_labels(folder: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(folder.glob("*.csv")):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "annotator_id" not in df or (df["annotator_id"] == "").all():
            df["annotator_id"] = p.stem
        df["annotator_id"] = df["annotator_id"].replace("", p.stem)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].str.strip().str.lower()
    return df[df["label"].isin(LABELS)]


def reliability_matrix(labels: pd.DataFrame) -> np.ndarray:
    """annotators x items matrix of label codes (nan = not labelled), as krippendorff.alpha expects."""
    wide = labels.pivot_table(index="annotator_id", columns="item_id", values="label", aggfunc="first")
    code = {lab: i for i, lab in enumerate(LABELS)}
    return wide.apply(lambda col: col.map(code)).to_numpy(dtype=float)


def _alpha(data: np.ndarray) -> float:
    import krippendorff
    if data.shape[0] < 2 or int(((~np.isnan(data)).sum(axis=0) >= 2).sum()) == 0:
        return float("nan")
    if len(np.unique(data[~np.isnan(data)])) < 2:
        return 1.0                                          # perfect agreement on a single category
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement="nominal"))


def alpha_nominal(labels: pd.DataFrame) -> float:
    """Krippendorff's alpha (nominal) from long-format rows (item_id, annotator_id, label)."""
    return float("nan") if labels.empty else _alpha(reliability_matrix(labels))


def alpha_ci(labels: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Point alpha and a percentile bootstrap 95% CI, resampling items (matrix columns) with replacement."""
    if labels.empty:
        return float("nan"), float("nan"), float("nan")
    data = reliability_matrix(labels)
    point = _alpha(data)
    if data.shape[1] < 5 or np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [a for a in (_alpha(data[:, rng.integers(0, data.shape[1], data.shape[1])]) for _ in range(n_boot))
             if not np.isnan(a)]
    if not boots:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def adjudicate(labels: pd.DataFrame) -> pd.DataFrame:
    """Majority label per item; ties -> 'disputed'. Keeps n_labels and the agreement share."""
    rows = []
    for iid, g in labels.groupby("item_id"):
        counts = g["label"].value_counts()
        top = counts[counts == counts.max()]
        gold = top.index[0] if len(top) == 1 else "disputed"
        first = g.iloc[0]
        rows.append({"item_id": iid, "paper_key": first.get("paper_key"), "field": first.get("field"),
                     "gold": gold, "n_labels": int(len(g)), "agreement": float(counts.max() / len(g)),
                     "omission": "; ".join(o for o in g.get("omission", pd.Series(dtype=str)) if o)})
    return pd.DataFrame(rows, columns=["item_id", "paper_key", "field", "gold", "n_labels", "agreement", "omission"])


@stage("ws1", "human-agreement", needs=("human-sample",), config_keys=("ws1b.human",),
       help="Krippendorff's alpha + adjudicated gold labels from the filled-in annotation sheets")
def run_agreement(ctx: Context) -> dict:
    if not human_enabled(ctx):
        return ctx.skip("Human agreement", HUMAN_OFF)
    labels_dir = ctx.input_dir("human-sample") / "labels"
    labels = read_labels(labels_dir)
    if labels.empty:
        ctx.publish("Human agreement", [{"type": "markdown", "text":
                    f"No labels yet. Put filled-in sheets in `{labels_dir}` and re-run this stage."}],
                    summary="waiting for labels")
        return {"labels": 0}
    labels = labels.drop_duplicates(["item_id", "annotator_id"], keep="last")
    multi = labels[labels.groupby("item_id")["annotator_id"].transform("nunique") >= 2]
    a, lo, hi = alpha_ci(multi)
    per_field = []
    for f, g in multi.groupby("field"):
        fa, flo, fhi = alpha_ci(g)
        per_field.append({"field": f, "items": g["item_id"].nunique(), "alpha": fa, "lo": flo, "hi": fhi})
    gold = adjudicate(labels)
    atomic_write_parquet(ctx.out / "gold.parquet", __import__("pyarrow").Table.from_pandas(gold, preserve_index=False))
    dist = labels.groupby(["field", "label"]).size().unstack(fill_value=0).reindex(columns=list(LABELS), fill_value=0)
    gold_rates = [{"label": lab, **rate_row((gold["gold"] == lab).sum(), len(gold))} for lab in (*LABELS, "disputed")]
    ctx.write_json("agreement.json", {"alpha": a, "lo": lo, "hi": hi, "per_field": per_field,
                                      "annotators": labels["annotator_id"].nunique(), "items": len(gold)})
    ctx.publish("Human agreement", [
        {"type": "kpis", "items": [
            {"label": "Krippendorff α (nominal)", "value": a, "fmt": "float3",
             "hint": f"95% CI {lo:.3f}–{hi:.3f}" if not np.isnan(lo) else "too few double-labelled items for a CI",
             "status": "good" if a >= 0.667 else "warn"},
            {"label": "annotators", "value": int(labels["annotator_id"].nunique()), "fmt": "int"},
            {"label": "labelled items", "value": len(gold), "fmt": "int"},
            {"label": "double-labelled items", "value": int(multi["item_id"].nunique()), "fmt": "int"}]},
        {"type": "table", "title": "Agreement per field", "columns": [
            {"key": "field", "label": "field"}, {"key": "items", "label": "items", "fmt": "int"},
            {"key": "alpha", "label": "α", "fmt": "float3"}, {"key": "lo", "label": "lo", "fmt": "float3"},
            {"key": "hi", "label": "hi", "fmt": "float3"}], "rows": per_field},
        {"type": "bar", "title": "Labels per field", "stacked": True, "horizontal": True,
         "labels": list(dist.index), "series": [{"name": lab, "data": dist[lab].tolist()} for lab in LABELS],
         "fmt": "int"},
        {"type": "table", "title": "Gold label rates", "columns": [
            {"key": "label", "label": "gold"}, {"key": "k", "label": "items", "fmt": "int"},
            {"key": "rate", "label": "rate", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
            {"key": "hi", "label": "hi", "fmt": "pct1"}], "rows": gold_rates},
    ], summary=f"α = {a:.3f} on {multi['item_id'].nunique()} double-labelled items")
    return {"labels": len(labels), "items": len(gold), "alpha": a}
