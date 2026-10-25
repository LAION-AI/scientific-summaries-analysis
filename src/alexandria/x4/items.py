"""Stage `items` (X4, GPU): the field-level / cross-paper benchmark, built from the WS3 corpus.

Two item types, each tied to one KU field (limitations / methods / results / claims):
  single  one paper: a question about what that paper's field says (e.g. the limitation it reports)
  cross   a topic group (a seed paper + its nearest neighbours by mean KU embedding): a question whose answer is in the
          field of exactly ONE paper of the group, so it can only be answered by comparing that field across papers
          (e.g. "Among studies of <topic>, which approach reports <property> as a limitation?")
Only papers with full text are used, so the full-text arms can find the same information. A generator writes each
item from the field texts (JSON: question, short answer, exact evidence quote, the answering paper); rule checks
(short answer, self-contained, answer not in the question, evidence quote found in the target field and not in the
other papers) and a verifier from another family (supported, unique within the group / vs. a random other paper,
self-contained) decide what is kept. The verifier's "answerable from general knowledge" flag is stored but does not
filter (an 8B verifier is unreliable at it); the closed-book arm A0 measures it instead (results: paper-required slice). Every candidate and its outcome is stored;
a human-verification packet is written for a random subset (filled later; not required to run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, parse_json, with_model
from ..runtime import Context, log, stage
from .common import check_item, paper_fields, paper_vectors, topic_groups

LABEL = {"limitations": "limitations reported", "methods": "methods and procedures", "results": "key results",
         "claims": "claims with their evidence"}

SINGLE_SYSTEM = """You write exam questions that test what one part of a scientific paper's structured summary says.
You see the paper's title and ONE field of its summary. Write one question that can only be answered with that field's content.
Rules: the question is self-contained (name the topic, system or method concretely; never write "this paper", "the study" or "the authors"); it does not contain the answer; the answer is short (a name, a number with unit, or a phrase of at most 10 words); "evidence" is an exact quote (at most 40 words) from the field that contains the answer.
Answer with JSON only, exactly these keys: {"question": ..., "answer": ..., "evidence": ...}"""
CROSS_SYSTEM = """You write questions that require comparing several scientific papers on the same topic.
You see the same summary field for {n} papers (P1..P{n}). Write ONE question whose answer is stated in the field of exactly ONE of these papers, and that a reader can only answer by checking this field across the papers (for example: which approach, system or setting reports a given property, limitation, result or claim).
Rules: describe the shared topic and the property asked for, but do NOT name the answering paper's system in the question; never write "this paper", "P1" or "the authors"; the answer is the short name of the method, system, material or setting of the answering paper (at most 10 words); "paper" is its number; "evidence" is an exact quote (at most 40 words) from that paper's field that shows the answer.
Answer with JSON only, exactly these keys: {{"question": ..., "answer": ..., "paper": <number>, "evidence": ...}}"""
VERIFY_SYSTEM = """You check an exam question about scientific papers before it enters a benchmark. You see one or more texts (fields of paper summaries), the question, the gold answer and which text is said to contain it.
Judge: supported = the indicated text states the gold answer; unique = no OTHER shown text also answers the question (equally well or with a different answer); self_contained = the question names its topic concretely and does not refer to "this paper" or to text labels; general_knowledge = a well-read expert could answer it correctly WITHOUT any of the texts. Answer with JSON only."""
VERIFY_SCHEMA = {"type": "object", "properties": {k: {"type": "boolean"} for k in
                 ("supported", "unique", "self_contained", "general_knowledge")},
                 "required": ["supported", "unique", "self_contained", "general_knowledge"], "additionalProperties": False}


FILTER = ("supported", "unique", "self_contained")                     # verifier flags that must be true


def _gen_schema(n: int | None) -> dict:
    props = {"question": {"type": "string", "maxLength": 400}, "answer": {"type": "string", "maxLength": 120},
             "evidence": {"type": "string", "maxLength": 400}}
    req = ["question", "answer", "evidence"]
    if n:
        props["paper"] = {"type": "integer", "minimum": 1, "maximum": n}
        req.append("paper")
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


def _generate(llm, tok, spec, cands: list[dict]) -> list[str]:
    prompts, params = [], []
    for c in cands:
        if c["type"] == "single":
            user = (f"TITLE: {c['titles'][0]}\nFIELD ({LABEL[c['field']]}):\n<<<\n{c['texts'][0]}\n>>>\n\n"
                    "Write one question.")
            prompts.append(chat_prompt(tok, spec, SINGLE_SYSTEM, user))
            params.append(json_params(_gen_schema(None), 400))
        else:
            n = len(c["texts"])
            body = "\n\n".join(f"P{i + 1} ({c['titles'][i]}), field: {LABEL[c['field']]}\n<<<\n{t}\n>>>"
                               for i, t in enumerate(c["texts"]))
            prompts.append(chat_prompt(tok, spec, CROSS_SYSTEM.format(n=n), body + "\n\nWrite one question."))
            params.append(json_params(_gen_schema(n), 400))
    outs = llm.generate(prompts, params, use_tqdm=False)
    texts = [o.outputs[0].text if o.outputs else "" for o in outs]
    # constrained decoding can run into whitespace until max_tokens (unparseable): retry those unconstrained
    bad = [i for i, t in enumerate(texts) if parse_json(t) is None]
    if bad:
        log(f"items: {len(bad)} unparseable generations, retrying without the JSON constraint")
        for i, o in zip(bad, llm.generate([prompts[i] for i in bad], json_params(None, 700), use_tqdm=False), strict=True):
            texts[i] = o.outputs[0].text if o.outputs else ""
    return texts


def _verify(llm, tok, spec, jobs: list[dict]) -> list[str]:
    prompts = []
    for j in jobs:
        body = "\n\n".join(f"TEXT {i + 1}:\n<<<\n{t}\n>>>" for i, t in enumerate(j["texts"]))
        prompts.append(chat_prompt(tok, spec, VERIFY_SYSTEM, f"{body}\n\nQUESTION: {j['question']}\nGOLD ANSWER: "
                                   f"{j['answer']}\nSAID TO BE IN: TEXT {j['target'] + 1}"))
    outs = llm.generate(prompts, json_params(VERIFY_SCHEMA, 80), use_tqdm=False)
    return [o.outputs[0].text if o.outputs else "" for o in outs]


@stage("x4", "items", resources="gpu", env="llm", config_keys=("x4.items",),
       help="field-level and cross-paper questions from the WS3 corpus: generator + rule checks + verifier (another family)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("x4.items")
    settings = ctx.cfg.profile_settings if ctx.cfg.profile == "debug" else {}
    n_single = int(settings.get("x4_single", cfg.get("n_single", 400)))
    n_cross = int(settings.get("x4_cross", cfg.get("n_cross", 600)))
    idx = ctx.cfg.stage_dir("ws3", "index")
    units = pd.read_parquet(idx / "units.parquet")
    emb = np.load(idx / "emb_summary.npy", mmap_mode="r")
    pf = paper_fields(units)
    meta = units.drop_duplicates("paper_key").set_index("paper_key")
    fields = dict(cfg.get("fields", {}))
    cap = int(cfg.get("max_field_chars", 2500))

    def ftext(pk: str, f: str) -> str:
        return "\n".join(pf.get(pk, {}).get(u, "") for u in fields[f]).strip()[:cap]

    ok_papers = [pk for pk in pf if bool(meta.at[pk, "has_text"]) and all(ftext(pk, f) for f in fields)]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    names = list(fields)
    cands = []
    for i, pk in enumerate(rng.choice(ok_papers, size=min(n_single, len(ok_papers)), replace=False)):
        f = names[i % len(names)]
        cands.append({"type": "single", "field": f, "papers": [pk], "titles": [str(meta.at[pk, "title"])],
                      "texts": [ftext(pk, f)]})
    keys, vec = paper_vectors(units, emb, center=bool(cfg.get("center_vectors", True)))
    keep = [k in set(ok_papers) for k in keys]
    keys = [k for k, m in zip(keys, keep, strict=True) if m]
    vec = vec[np.array(keep)]
    groups = topic_groups(keys, vec, int(cfg.get("group_size", 3)), float(cfg.get("min_neighbour_sim", 0.55)), n_cross, rng)
    for i, g in enumerate(groups):
        f = names[i % len(names)]
        cands.append({"type": "cross", "field": f, "papers": g, "titles": [str(meta.at[p, "title"]) for p in g],
                      "texts": [ftext(p, f) for p in g]})
    log(f"items: {sum(c['type'] == 'single' for c in cands)} single + {len(groups)} cross candidates from {len(ok_papers)} papers")

    gen = model_spec(ctx.cfg, cfg.get("generator", "qwen3-8b"))
    raw = with_model(gen, _generate, cands, gpu_memory_utilization=0.85)
    rows, jobs = [], []
    for i, (c, txt) in enumerate(zip(cands, raw, strict=True)):
        q = parse_json(txt) or {}
        t = (int(q.get("paper", 1)) - 1) if c["type"] == "cross" else 0
        t = t if 0 <= t < len(c["papers"]) else 0
        others = [x for j, x in enumerate(c["texts"]) if j != t]
        if c["type"] == "single":                                   # a random other paper's same field, for uniqueness
            o = rng.choice([p for p in ok_papers if p != c["papers"][0]])
            others = [ftext(o, c["field"])]
        status = check_item(q, c["texts"][t], others, int(cfg.get("max_answer_words", 12)),
                            float(cfg.get("min_evidence_match", 85)))
        row = {"item_id": f"x4:{c['type']}:{i}", "type": c["type"], "field": c["field"], "question": str(q.get("question", "")).strip(),
               "answer": str(q.get("answer", "")).strip(), "evidence": str(q.get("evidence", "")), "gold_paper_key": c["papers"][t],
               "gold_papers": [c["papers"][t]], "group": c["papers"], "status": status, "raw": txt[:2000]}
        rows.append(row)
        if status == "ok":
            texts = [c["texts"][t], *others] if c["type"] == "single" else c["texts"]
            jobs.append({"row": len(rows) - 1, "texts": texts, "question": row["question"], "answer": row["answer"],
                         "target": 0 if c["type"] == "single" else t})
    ver = model_spec(ctx.cfg, cfg.get("verifier", "granite3.3-8b"))
    log(f"items: verifying {len(jobs)} of {len(rows)} with {ver['name']}")
    verdicts = with_model(ver, _verify, jobs, gpu_memory_utilization=0.85) if jobs else []
    for j, v in zip(jobs, verdicts, strict=True):
        d = parse_json(v) or {}
        failed = [k for k in FILTER if d.get(k) is not True]
        rows[j["row"]]["status"] = f"verifier:{failed[0]}" if failed else "ok"
        rows[j["row"]].update({f"verify_{k}": d.get(k) for k in VERIFY_SCHEMA["required"]})
    allc = pd.DataFrame(rows)
    allc.to_parquet(ctx.out / "all_generated.parquet", index=False)
    items = allc[allc["status"] == "ok"].drop(columns=["raw"]).copy()
    items["bench"], items["answer_type"], items["cluster"] = "x4", "short", items["item_id"]
    items["options"] = [[] for _ in range(len(items))]
    items.to_parquet(ctx.out / "items.parquet", index=False)
    pk = items.sample(n=min(int(cfg.get("human_packet", 60)), len(items)), random_state=0)
    pk[["item_id", "type", "field", "question", "answer", "evidence", "gold_paper_key"]].assign(
        correct="", unambiguous="", needs_field="", notes="").to_csv(ctx.out / "verify.csv", index=False)
    with open(ctx.out / "verify.md", "w") as fh:
        fh.write("# X4 human verification\nFor each item: is the answer correct and unique given the paper(s)? "
                 "Does it need the named field? Fill the columns in verify.csv.\n\n")
        for r in pk.itertuples():
            fh.write(f"## {r.item_id} ({r.type}, {r.field})\nQ: {r.question}\nA: {r.answer}\nEvidence: {r.evidence}\n"
                     f"Gold paper: {r.gold_paper_key} ({meta.at[r.gold_paper_key, 'title']})\n\n")
    st = allc["status"].value_counts()
    by = items.groupby(["type", "field"]).size().reset_index(name="items")
    ctx.publish("X4 benchmark items", [
        {"type": "kpis", "items": [
            {"label": "items kept", "value": len(items), "fmt": "int"},
            {"label": "single-paper (field)", "value": int((items["type"] == "single").sum()), "fmt": "int"},
            {"label": "cross-paper", "value": int((items["type"] == "cross").sum()), "fmt": "int"},
            {"label": "kept share", "value": len(items) / max(len(allc), 1), "fmt": "pct1"},
            {"label": "generator / verifier", "value": f"{gen['name']} / {ver['name']}"}]},
        {"type": "table", "title": "Kept items by type and field", "rows": by.to_dict("records"),
         "columns": [{"key": "type", "label": "type"}, {"key": "field", "label": "field"}, {"key": "items", "label": "items", "fmt": "int"}]},
        {"type": "bar", "title": "Outcome of every generated candidate", "horizontal": True, "fmt": "int",
         "labels": list(st.index), "series": [{"name": "candidates", "data": st.tolist()}]},
    ], summary=f"{len(items)} items ({(items['type'] == 'cross').sum()} cross-paper) of {len(allc)} candidates")
    return {"items": len(items), "candidates": len(allc), "groups": len(groups)}
