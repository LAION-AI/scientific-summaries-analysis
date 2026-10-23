"""Stages `judge`, `judge-report` and `calibrate`: a sentence-level LLM judge with the human label set, and its
calibration on human gold labels.

judge      Qwen3-8B (ws1b.judge.model) served by vLLM. Each (paper, field) section is split into numbered sentences
           (the same splitter as `grounding`; claims: one item per claim). The model returns, per sentence, a short
           piece of evidence and then a verdict (supported / unsupported / contradicted), as JSON constrained by a
           per-item schema with exactly one entry per sentence (vLLM structured outputs). The section label is the
           worst sentence verdict, computed in code, so label and flagged spans cannot disagree. The prompt has
           explicit rules for statements *about* the paper ("the paper does not report X"), which limitation
           sections are made of. Output budget scales with the number of sentences; answers cut off at the budget
           are retried once with twice the budget. Papers: the same stratified sample as `grounding` (same seed and
           n), so judge and MiniCheck can be compared sentence by sentence. temperature 0; Qwen3 thinking off.
           Item ids equal the human protocol's (paper_key:field). Batches in parts/ make the stage resumable.
judge-report  label rates per field and input type, sentence-level rates, and agreement with MiniCheck.
calibrate  judge vs adjudicated human gold (confusion matrix, Cohen's kappa, per-class precision / recall), and the
           MiniCheck score against human labels (ROC AUC, best-F1 threshold, isotonic calibration map), when the
           inputs exist. Skipped while ws1b.human.enabled is false.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa

from ...papers import paper_table, with_paper_info
from ...runtime import Context, atomic_write_parquet, log, stage
from .common import HUMAN_OFF, LABELS, fetch_records, field_items, human_enabled, item_id, source_text, \
    split_sentences
from .grounding import sample_papers

SEVERITY = {"supported": 0, "unsupported": 1, "contradicted": 2}


def judge_schema(n: int, evidence_chars: int = 240, omission_chars: int = 200) -> dict:
    """JSON schema for one section of n sentences: exactly n {id, evidence, verdict} entries (evidence first, so
    the model points at the paper before it decides), plus up to 3 omissions. Lengths are bounded so an answer
    cannot run past its token budget by copying paragraphs."""
    return {
        "type": "object",
        "properties": {
            "verdicts": {"type": "array", "minItems": n, "maxItems": n, "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "minimum": 1, "maximum": n},
                               "evidence": {"type": "string", "maxLength": evidence_chars},
                               "verdict": {"type": "string", "enum": list(LABELS)}},
                "required": ["id", "evidence", "verdict"], "additionalProperties": False}},
            "omissions": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": omission_chars}},
        },
        "required": ["verdicts", "omissions"],
        "additionalProperties": False,
    }


SYSTEM = """You check, sentence by sentence, whether a section of an AI-written summary of a scientific paper is faithful to that paper. Judge only against the PAPER text given, not against your own knowledge.

Verdict for each numbered sentence:
- supported: everything it states is in the paper or directly follows from it. Paraphrase, summarising and generic framing that matches the paper are fine.
- unsupported: it states something the paper does not contain (an invented number, method, dataset, result or claim), but nothing in it conflicts with the paper.
- contradicted: it conflicts with the paper (wrong number, wrong direction of an effect, wrong method, wrong conclusion).

Sentences about the paper itself, common in limitation sections, are checked the same way:
- "The paper does not report / address X" is supported if the paper indeed does not, and contradicted only if the paper does report X.
- A limitation, caveat or open question is supported if the paper states it or it follows directly from what the paper did (e.g. "evaluated only on dataset D" when D is the only dataset used). A critique the paper gives no basis for is unsupported.
- Describing a conflict, contradiction or limitation does not make a sentence contradicted. Only a sentence that misstates the paper is contradicted.
- If the paper text is truncated, a statement that may concern the missing part is unsupported, not contradicted.

For each sentence write "evidence" first (a short quote or pointer from the paper, or what is missing or conflicting; at most 25 words), then the "verdict". Then list up to 3 "omissions": important results, limitations or conditions of the paper that this section should mention but does not (at most 25 words each). Answer with JSON only."""

USER = """PAPER ({completeness}):
<<<
{source}
>>>

SUMMARY SECTION "{field}", numbered sentences:
{sentences}"""


def sampling_params(schema: dict, max_tokens: int, temperature: float = 0.0):
    """SamplingParams with a JSON-schema constraint, for both vLLM APIs:
    >= 0.10.2: SamplingParams(structured_outputs=StructuredOutputsParams(json=...));
    older:     SamplingParams(guided_decoding=GuidedDecodingParams(json=...))."""
    from vllm import SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=schema))


def token_budget(n_sentences: int, per_sentence: int = 100, base: int = 400) -> int:
    return base + per_sentence * n_sentences


def parse_answer(text: str, sentences: list[str]) -> dict:
    """Parse one section's JSON into per-sentence verdicts and the section label (worst verdict).
    Anything malformed or incomplete -> label 'parse_error' (counted, never silently dropped)."""
    n = len(sentences)
    s = (text or "").strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    err = {"label": "parse_error", "sentence_verdicts": [], "sentence_evidence": [], "unsupported_spans": [],
           "omissions": [], "rationale": s[-300:]}
    try:
        v = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return err
    verdicts, evidence = [None] * n, [""] * n
    for e in v.get("verdicts") or []:
        try:
            i = int(e.get("id")) - 1
        except (TypeError, ValueError):
            continue
        lab = str(e.get("verdict", "")).strip().lower()
        if 0 <= i < n and lab in LABELS:
            verdicts[i], evidence[i] = lab, str(e.get("evidence", ""))[:300]
    if any(x is None for x in verdicts):
        return err
    worst = max(verdicts, key=SEVERITY.__getitem__)
    bad = [i for i, x in enumerate(verdicts) if x != "supported"]
    return {"label": worst, "sentence_verdicts": verdicts, "sentence_evidence": evidence,
            "unsupported_spans": [sentences[i] for i in bad],
            "omissions": [str(x) for x in v.get("omissions") or []][:3],
            "rationale": " | ".join(f"[{i + 1}] {evidence[i]}" for i in bad)[:600]}


def section_sentences(rec: dict, field: str, max_sentences: int) -> tuple[list[str], bool]:
    """Sentences of one section (claims: one per claim), capped; returns (sentences, capped?)."""
    sents: list[str] = []
    for item in field_items(rec, field):
        sents.extend([item] if field == "claims" else split_sentences(item))
    return sents[:max_sentences], len(sents) > max_sentences


@stage("ws1", "judge", needs=("features",), resources="gpu", env="llm", config_keys=("ws1b.judge",),
       help="Qwen3-8B sentence-level judge: supported / unsupported / contradicted per sentence and field (vLLM)")
def run(ctx: Context) -> dict:
    from vllm import LLM
    cfg = ctx.section("ws1b.judge")
    fields = list(cfg.get("fields", ["executive_summary", "key_results"]))
    max_src = int(cfg.get("max_source_tokens", 32000))
    max_sent = int(cfg.get("max_sentences", 40))
    batch_papers = int(cfg.get("batch_papers", 32))
    thinking = bool(cfg.get("enable_thinking", False))
    temperature = float(cfg.get("temperature", 0.0))
    per_sent = int(cfg.get("tokens_per_sentence", 100)) + (300 if thinking else 0)

    papers = sample_papers(ctx, int(cfg.get("n_papers", 10000)), int(cfg.get("seed", 0)))
    parts = ctx.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    batches = [papers.iloc[i:i + batch_papers] for i in range(0, len(papers), batch_papers)]
    todo = [(k, b) for k, b in enumerate(batches) if ctx.force or not (parts / f"batch_{k:05d}.parquet").exists()]
    log(f"judge: {len(papers)} papers, {len(batches)} batches, {len(todo)} to do")
    if not todo:
        return {"papers": len(papers), "batches": len(batches), "done": 0}

    llm = LLM(model=cfg.get("model", "Qwen/Qwen3-8B"), max_model_len=int(cfg.get("max_model_len", 40960)),
              gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)), enable_prefix_caching=True,
              revision=cfg.get("revision"), tokenizer_revision=cfg.get("revision"), seed=0)
    tok = llm.get_tokenizer()
    n_items = n_err = n_retry = 0
    for k, batch in todo:
        recs = fetch_records(ctx.cfg, zip(batch["file"], batch["row"], strict=True),
                             ["paper_id", "text_raw", "text_sanitized", *fields])
        prompts, meta = [], []
        for p in batch.itertuples():
            rec = recs.get((p.file, int(p.row)))
            if rec is None:
                continue
            ids = tok(source_text(rec), add_special_tokens=False)["input_ids"]
            truncated = len(ids) > max_src
            src = tok.decode(ids[:max_src])
            for f in fields:
                sents, capped = section_sentences(rec, f, max_sent)
                if not sents:
                    continue
                msg = [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": USER.format(
                           completeness=f"truncated to its first {max_src} tokens" if truncated else "complete text",
                           source=src, field=f, sentences="\n".join(f"{i}. {x}" for i, x in enumerate(sents, 1)))}]
                prompts.append(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False,
                                                       enable_thinking=thinking))
                meta.append({"item_id": item_id(p.paper_key, f), "paper_key": p.paper_key, "paper_id": p.paper_id,
                             "file": p.file, "row": int(p.row), "group": p.group, "input_type": p.input_type,
                             "domain": p.domain, "field": f, "source_truncated": truncated,
                             "sentences": sents, "n_sentences": len(sents), "sentences_capped": capped})
        answers: list[dict] = [{} for _ in meta]
        pending = list(range(len(meta)))
        for attempt in range(2):                       # second pass: answers cut off at the budget, twice the budget
            if not pending:
                break
            params = [sampling_params(judge_schema(meta[i]["n_sentences"]),
                                      token_budget(meta[i]["n_sentences"], per_sent) * (attempt + 1), temperature)
                      for i in pending]
            outs = llm.generate([prompts[i] for i in pending], params, use_tqdm=False)
            again = []
            for i, out in zip(pending, outs, strict=True):
                o = out.outputs[0] if out.outputs else None
                ans = parse_answer(o.text if o else "", meta[i]["sentences"])
                answers[i] = {**ans, "finish_reason": getattr(o, "finish_reason", None), "attempts": attempt + 1}
                if ans["label"] == "parse_error" and attempt == 0:
                    again.append(i)
            n_retry += len(again)
            pending = again
        rows = [{**m, **a, "n_unsupported": sum(v == "unsupported" for v in a["sentence_verdicts"]),
                 "n_contradicted": sum(v == "contradicted" for v in a["sentence_verdicts"])}
                for m, a in zip(meta, answers, strict=True)]
        n_items += len(rows)
        n_err += sum(r["label"] == "parse_error" for r in rows)
        atomic_write_parquet(parts / f"batch_{k:05d}.parquet", pa.Table.from_pylist(rows) if rows else
                             pa.table({"item_id": pa.array([], pa.string())}))
        log(f"judge: batch {k + 1}/{len(batches)} ({len(rows)} items)")
    return {"papers": len(papers), "batches": len(batches), "done": len(todo), "items": n_items,
            "retried": n_retry, "parse_errors": n_err}


# ---------------------------------------------------------------- calibrate
def _load_parts(ctx: Context, stage_name: str) -> pd.DataFrame:
    import duckdb
    try:
        parts = ctx.parts(stage_name)
    except FileNotFoundError:
        return pd.DataFrame()
    if not parts:
        return pd.DataFrame()
    return duckdb.sql(f"select * from read_parquet({[str(p) for p in parts]!r}, union_by_name=true)").df()


def judge_vs_gold(judge: pd.DataFrame, gold: pd.DataFrame) -> dict:
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
    m = judge.merge(gold[gold["gold"].isin(LABELS)], on="item_id", how="inner")
    m = m[m["label"].isin(LABELS)]
    if m.empty:
        return {"n": 0}
    cm = confusion_matrix(m["gold"], m["label"], labels=list(LABELS))
    rep = classification_report(m["gold"], m["label"], labels=list(LABELS), output_dict=True, zero_division=0)
    return {"n": len(m), "kappa": float(cohen_kappa_score(m["gold"], m["label"], labels=list(LABELS))),
            "accuracy": float((m["gold"] == m["label"]).mean()), "confusion": cm.tolist(),
            "per_class": {lab: {k: float(rep[lab][k]) for k in ("precision", "recall", "f1-score", "support")}
                          for lab in LABELS}}


def minicheck_vs_gold(field_scores: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Binary: gold supported vs not; score = MiniCheck min sentence score of the field."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    fs = field_scores.assign(item_id=[item_id(k, f) for k, f in zip(field_scores["paper_key"], field_scores["field"],
                                                                     strict=True)])
    m = fs.merge(gold[gold["gold"].isin(LABELS)], on="item_id", how="inner")
    if m.empty or m["gold"].eq("supported").nunique() < 2:
        return {"n": int(len(m))}
    y = m["gold"].eq("supported").to_numpy(dtype=int)
    s = m["score_min"].to_numpy(dtype=float)
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1[:-1])) if len(thr) else 0
    iso = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    grid = np.linspace(0, 1, 21)
    return {"n": int(len(m)), "auc": float(roc_auc_score(y, s)), "best_f1": float(f1[best]),
            "best_threshold": float(thr[best]) if len(thr) else 0.5,
            "isotonic": {"x": grid.tolist(), "y": iso.predict(grid).tolist()}}


@stage("ws1", "calibrate", needs=("judge", "human-agreement"), config_keys=("ws1b.judge", "ws1b.grounding", "ws1b.human"),
       help="judge and MiniCheck vs human gold labels: kappa, confusion, AUC, best threshold, isotonic map")
def run_calibrate(ctx: Context) -> dict:
    if not human_enabled(ctx):
        return ctx.skip("Judge calibration", HUMAN_OFF)
    gold_path = ctx.cfg.stage_dir("ws1", "human-agreement") / "gold.parquet"
    if not gold_path.exists():
        ctx.publish("Judge calibration", [{"type": "markdown", "text":
                    "Waiting for human labels: run `human-sample`, collect the sheets, then `human-agreement`."}],
                    summary="waiting for human labels")
        return {"gold": 0}
    gold = pd.read_parquet(gold_path)
    judge = _load_parts(ctx, "judge")
    jv = judge_vs_gold(judge, gold) if not judge.empty else {"n": 0}
    mc: dict = {"n": 0}
    fs_path = ctx.cfg.stage_dir("ws1", "grounding-report") / "field_scores.parquet"
    if fs_path.exists():
        mc = minicheck_vs_gold(pd.read_parquet(fs_path), gold)
    ctx.write_json("calibration.json", {"judge": jv, "minicheck": mc})
    panels: list[dict] = [{"type": "kpis", "items": [
        {"label": "gold items", "value": int(len(gold)), "fmt": "int"},
        {"label": "judge vs gold (n)", "value": jv.get("n", 0), "fmt": "int"},
        {"label": "judge Cohen's κ", "value": jv.get("kappa"), "fmt": "float3"},
        {"label": "MiniCheck AUC", "value": mc.get("auc"), "fmt": "float3",
         "hint": f"best F1 {mc['best_f1']:.3f} at {mc['best_threshold']:.2f}" if "best_f1" in mc else "no overlap yet"}]}]
    if jv.get("n"):
        panels.append({"type": "table", "title": "Judge confusion matrix (rows = human gold)",
                       "columns": [{"key": "gold", "label": "gold"}] + [{"key": lab, "label": lab, "fmt": "int"}
                                                                         for lab in LABELS],
                       "rows": [{"gold": g, **dict(zip(LABELS, r, strict=True))}
                                for g, r in zip(LABELS, jv["confusion"], strict=True)]})
        panels.append({"type": "table", "title": "Judge per class",
                       "columns": [{"key": "label", "label": "label"}, {"key": "precision", "label": "P", "fmt": "float3"},
                                   {"key": "recall", "label": "R", "fmt": "float3"},
                                   {"key": "f1-score", "label": "F1", "fmt": "float3"},
                                   {"key": "support", "label": "n", "fmt": "int"}],
                       "rows": [{"label": k, **v} for k, v in jv["per_class"].items()]})
    if "isotonic" in mc:
        panels.append({"type": "line", "title": "MiniCheck score → P(human: supported) (isotonic)", "fmt": "float",
                       "labels": [f"{x:.2f}" for x in mc["isotonic"]["x"]],
                       "series": [{"name": "calibrated", "data": mc["isotonic"]["y"]}]})
    ctx.publish("Judge calibration", panels, summary=f"κ = {jv.get('kappa', float('nan')):.3f} on {jv.get('n', 0)} items")
    return {"gold": int(len(gold)), "judge_n": jv.get("n", 0), "minicheck_n": mc.get("n", 0)}


# also report the judge's own label rates (no gold needed) as part of the judge stage's dashboard
def judge_rates(judge: pd.DataFrame) -> list[dict]:
    from ...stats import rate_row
    rows = []
    for (f, it), g in judge.groupby(["field", "input_type"], dropna=False):
        for lab in (*LABELS, "parse_error"):
            rows.append({"field": f, "input_type": it, "label": lab, **rate_row(int((g["label"] == lab).sum()), len(g))})
    return rows


def explode_sentences(judge: pd.DataFrame) -> pd.DataFrame:
    """One row per judged sentence: paper_key, field, sent_index, sentence, verdict, evidence."""
    ok = judge[judge["label"].isin(LABELS)]
    rows = []
    for r in ok.itertuples():
        for i, (s, v, e) in enumerate(zip(r.sentences, r.sentence_verdicts, r.sentence_evidence, strict=True)):
            rows.append({"paper_key": r.paper_key, "field": r.field, "input_type": r.input_type, "group": r.group,
                         "sent_index": i, "sentence": s, "verdict": v, "evidence": e})
    return pd.DataFrame(rows)


def minicheck_agreement(sent: pd.DataFrame, mc: pd.DataFrame) -> dict:
    """Judge vs MiniCheck on the same sentences (supported vs not): n, agreement, Cohen's kappa, 2x2 counts."""
    from sklearn.metrics import cohen_kappa_score
    if sent.empty or mc.empty:
        return {"n": 0}
    m = sent.merge(mc[["paper_key", "field", "sent_index", "prob", "supported"]],
                   on=["paper_key", "field", "sent_index"], how="inner")
    if m.empty:
        return {"n": 0}
    j = m["verdict"].eq("supported")
    c = m["supported"].astype(bool)
    both_kinds = j.nunique() > 1 or c.nunique() > 1
    m = m.assign(j=j, c=c)
    by_field = [{"field": f, "n": len(g), "judge_only": int((g.j & ~g.c).sum()), "minicheck_only": int((~g.j & g.c).sum()),
                 "agreement": float((g.j == g.c).mean())} for f, g in m.groupby("field")]
    return {"n": int(len(m)), "agreement": float((j == c).mean()),
            "kappa": float(cohen_kappa_score(j, c)) if both_kinds else None,
            "table": {"both_supported": int((j & c).sum()), "judge_only": int((j & ~c).sum()),
                      "minicheck_only": int((~j & c).sum()), "neither": int((~j & ~c).sum())},
            "by_field": by_field}


@stage("ws1", "judge-report", needs=("judge",), config_keys=("ws1b.judge",),
       help="LLM-judge label rates by field and input type, sentence-level rates, agreement with MiniCheck")
def run_judge_report(ctx: Context) -> dict:
    from ...stats import rate_row
    from .grounding import load_sentences
    judge = _load_parts(ctx, "judge")
    if judge.empty or "label" not in judge:
        ctx.publish("LLM judge", [{"type": "markdown", "text": "No judge results yet."}])
        return {"items": 0}
    rates = judge_rates(judge)
    pivot = judge.groupby(["field", "label"]).size().unstack(fill_value=0)
    labs = [lab for lab in (*LABELS, "parse_error") if lab in pivot.columns]
    sent = explode_sentences(judge)
    sent_rows = [{"field": f, **{lab: float((g["verdict"] == lab).mean()) for lab in LABELS}, "n": len(g)}
                 for f, g in sent.groupby("field")] if not sent.empty else []
    try:
        mc = load_sentences(ctx, "grounding")
    except FileNotFoundError:
        mc = pd.DataFrame()
    agree = minicheck_agreement(sent, mc)
    ok = judge[judge["label"].isin(LABELS)]
    pp = ok.groupby(["paper_key", "file", "row", "input_type"], dropna=False).agg(
        sections=("label", "size"), unsupported=("label", lambda x: int((x == "unsupported").sum())),
        contradicted=("label", lambda x: int((x == "contradicted").sum())),
        flagged_sentences=("n_unsupported", "sum")).reset_index()
    pp["flagged_sentences"] += ok.groupby("paper_key")["n_contradicted"].sum().reindex(pp["paper_key"]).to_numpy()
    flagged_fields = ok[ok["label"] != "supported"].groupby("paper_key")["field"].agg(lambda x: ", ".join(sorted(x)))
    pp["fields"] = pp["paper_key"].map(flagged_fields).fillna("")
    per_paper = with_paper_info(ctx, pp.sort_values(["flagged_sentences", "contradicted"], ascending=False))
    per_paper.to_csv(ctx.out / "per_paper.csv", index=False)
    ctx.write_json("rates.json", rates)
    ctx.write_json("minicheck_agreement.json", {k: v for k, v in agree.items() if k != "disagreements"})
    n_sent = len(sent)
    s_sup = rate_row(int((sent["verdict"] == "supported").sum()), n_sent) if n_sent else {"rate": None}
    panels = [
        {"type": "kpis", "items": [
            {"label": "sections judged", "value": len(judge), "fmt": "int"},
            {"label": "sentences judged", "value": n_sent, "fmt": "int"},
            {"label": "sentences supported", "value": s_sup["rate"], "fmt": "pct1",
             "hint": f"95% CI {s_sup['lo']:.1%}–{s_sup['hi']:.1%}" if n_sent else ""},
            {"label": "sections unsupported", "value": float((judge["label"] == "unsupported").mean()), "fmt": "pct1"},
            {"label": "sections contradicted", "value": float((judge["label"] == "contradicted").mean()), "fmt": "pct1"},
            {"label": "parse errors", "value": int((judge["label"] == "parse_error").sum()), "fmt": "int",
             "status": "good" if not (judge["label"] == "parse_error").any() else "warn",
             "hint": f"{int((judge.get('attempts', pd.Series([1])) > 1).sum())} sections retried with a larger budget"},
            {"label": "agreement with MiniCheck", "value": agree.get("agreement"), "fmt": "pct1",
             "hint": f"κ = {agree['kappa']:.2f} on {agree['n']} shared sentences" if agree.get("kappa") is not None
             else f"{agree.get('n', 0)} shared sentences"}]},
        {"type": "bar", "title": "Section labels per field", "hint": "section label = worst sentence verdict",
         "stacked": True, "horizontal": True, "fmt": "int",
         "labels": list(pivot.index), "series": [{"name": lab, "data": pivot[lab].tolist()} for lab in labs]},
        {"type": "bar", "title": "Sentence verdicts per field", "stacked": True, "horizontal": True, "fmt": "pct",
         "labels": [r["field"] for r in sent_rows],
         "series": [{"name": lab, "data": [r[lab] for r in sent_rows]} for lab in LABELS]},
    ]
    if agree.get("n"):
        t = agree["table"]
        panels.append({"type": "table", "title": "Judge vs MiniCheck on the same sentences",
                       "hint": "MiniCheck supported = P(support) > 0.5; judge supported = verdict 'supported'",
                       "columns": [{"key": "judge", "label": "judge \\ MiniCheck"},
                                   {"key": "s", "label": "supported", "fmt": "int"},
                                   {"key": "n", "label": "not supported", "fmt": "int"}],
                       "rows": [{"judge": "supported", "s": t["both_supported"], "n": t["judge_only"]},
                                {"judge": "not supported", "s": t["minicheck_only"], "n": t["neither"]}]})
        panels.append({"type": "table", "title": "Judge vs MiniCheck disagreements by field",
                       "columns": [{"key": "field", "label": "field"}, {"key": "n", "label": "shared sentences", "fmt": "int"},
                                   {"key": "judge_only", "label": "only judge: supported", "fmt": "int"},
                                   {"key": "minicheck_only", "label": "only MiniCheck: supported", "fmt": "int"},
                                   {"key": "agreement", "label": "agreement", "fmt": "pct1"}],
                       "rows": agree["by_field"]})
    panels += [
        {"type": "table", "title": "Section label rates by field and input type",
         "columns": [{"key": "field", "label": "field"}, {"key": "input_type", "label": "input"},
                     {"key": "label", "label": "label"}, {"key": "n", "label": "n", "fmt": "int"},
                     {"key": "rate", "label": "rate", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
                     {"key": "hi", "label": "hi", "fmt": "pct1"}], "rows": rates, "limit": 300},
        paper_table("Papers judged", per_paper,
                    [("input_type", "input", None), ("sections", "sections", "int"),
                     ("unsupported", "unsupported sections", "int"), ("contradicted", "contradicted sections", "int"),
                     ("flagged_sentences", "flagged sentences", "int"), ("fields", "fields with a flagged sentence", None)],
                    hint="papers with the most flagged sentences first; full list in per_paper.csv"),
    ]
    ctx.publish("LLM judge", panels, summary=f"{len(judge)} sections, {n_sent} sentences judged")
    return {"items": len(judge), "sentences": n_sent, "minicheck_agreement": agree.get("agreement")}
