"""Stages `questions` (GPU) and `verify` of X2.

questions  A generator from a model family used nowhere else in the project (OpenAI gpt-oss-20b) writes short-answer
           questions from each new paper's FULL TEXT with the WS2 rules (specific to the paper, self-contained so they
           work in open-corpus mode, paraphrased, short answer, exact evidence quote) and the WS2 checks (evidence in
           the paper, no >= 8-word copy, not trivial, no duplicates), then the same model's critic pass. gpt-oss is a
           reasoning model in the harmony format: it is run unconstrained and the JSON is read from its final channel.
verify     A human-verification packet for a random subset: CSV (columns to fill: answerable from the paper?, answer
           correct?, self-contained?, notes) + Markdown with question, answer, evidence and the arXiv link.
"""
from __future__ import annotations

import pandas as pd

from ..llm import chat_prompt, json_params, model_spec, truncate_tokens, with_model
from ..qa import normalize_answer
from ..runtime import Context, log, stage
from ..ws2.questions import CRITIC_SCHEMA, CRITIC_SYSTEM, SYSTEM, USER, _norm, verify
from .common import parse_final_json, setting, stage_table

JSON_HINT = ('\n\nReturn JSON only: {"questions": [{"question": "...", "answer": "...", "type": "finding|numeric_result|'
             'method|setup|comparison|limitation|definition", "evidence": "<exact quote>"}]}')
CRITIC_HINT = ('\n\nReturn JSON only: {"identifies_study": true|false, "single_answer": true|false, '
               '"not_trivial": true|false, "needs_paper": true|false}')


def _generate(llm, tok, spec, papers: list[dict], cfg: dict) -> list[dict]:
    k = int(cfg.get("generate_per_paper", 6))
    max_src = min(int(cfg.get("max_source_tokens", 12000)),
                  int(spec.get("max_model_len", 24576)) - int(cfg.get("max_tokens", 6000)) - 1024)
    prompts = []
    for p in papers:
        src, _, cut = truncate_tokens(tok, p["full"], max_src)
        prompts.append(chat_prompt(tok, spec, SYSTEM + JSON_HINT, USER.format(
            completeness=f"first {max_src} tokens" if cut else "complete", source=src, k=k)))
    outs = llm.generate(prompts, json_params(None, int(cfg.get("max_tokens", 6000))), use_tqdm=False)
    gen = [{"paper_key": p["paper_key"], "raw": o.outputs[0].text if o.outputs else "",
            "questions": [q for q in ((parse_final_json(o.outputs[0].text if o.outputs else "") or {}).get("questions") or [])
                          if isinstance(q, dict) and q.get("question")]} for p, o in zip(papers, outs, strict=True)]
    flat = [(gi, qi, q) for gi, g in enumerate(gen) for qi, q in enumerate(g["questions"])]
    crit = llm.generate([chat_prompt(tok, spec, CRITIC_SYSTEM + CRITIC_HINT,
                                     f"Question: {q['question']}\nGold answer: {q.get('answer', '')}\nEvidence: "
                                     f"{q.get('evidence', '')}") for _, _, q in flat],
                        json_params(None, 2000), use_tqdm=False)
    for (gi, qi, _), o in zip(flat, crit, strict=True):
        gen[gi]["questions"][qi]["_critic"] = parse_final_json(o.outputs[0].text if o.outputs else "") or {}
    return gen


@stage("x2", "questions", needs=("papers",), resources="gpu", env="llm", config_keys=("x2.questions",),
       help="post-cutoff QA from the full text by a family used nowhere else (gpt-oss), WS2 rules + checks + critic")
def run_questions(ctx: Context) -> dict:
    cfg = ctx.section("x2.questions")
    papers = stage_table(ctx, "papers", "papers.parquet")
    items = papers[["paper_key", "full"]].to_dict("records")
    gen = model_spec(ctx.cfg, cfg.get("generator", "gpt-oss-20b"))
    log(f"questions: {len(items)} papers with {gen['name']}")
    raw = with_model(gen, _generate, items, cfg, gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)))
    src = dict(zip(papers["paper_key"], papers["full"], strict=True))
    per, n_copy, rows = int(setting(ctx, "per_paper", cfg.get("per_paper", 2))), int(cfg.get("copy_ngram", 8)), []
    for r in raw:
        w = normalize_answer(src[r["paper_key"]]).split()
        grams = {tuple(w[i:i + n_copy]) for i in range(len(w) - n_copy + 1)}
        seen, kept = [], 0
        if not r["questions"]:
            rows.append({"qid": f"{r['paper_key']}:none", "paper_key": r["paper_key"], "question": "", "answer": "",
                         "qtype": None, "evidence": "", "status": "no_parsable_output", "raw_tail": r["raw"][-300:]})
        for j, q in enumerate(r["questions"]):
            reason = verify(q, src[r["paper_key"]], grams, cfg, seen)
            crit = q.get("_critic") or {}
            if reason == "ok":
                failed = [k for k in CRITIC_SCHEMA["required"] if crit.get(k) is not True]
                reason = f"critic:{failed[0]}" if failed else ("over_cap" if kept >= per else "ok")
            if reason == "ok":
                seen.append(_norm(q["question"]))
                kept += 1
            rows.append({"qid": f"{r['paper_key']}:q{j}", "paper_key": r["paper_key"], "question": q["question"].strip(),
                         "answer": str(q.get("answer", "")).strip(), "qtype": q.get("type"),
                         "evidence": q.get("evidence", ""), "status": reason, "raw_tail": None})
    allq = pd.DataFrame(rows)
    allq.to_parquet(ctx.out / "all_generated.parquet", index=False)
    qs = allq[allq["status"] == "ok"].drop(columns="raw_tail").reset_index(drop=True)
    qs["generator"] = gen["name"]
    qs.to_parquet(ctx.out / "questions.parquet", index=False)
    st = allq["status"].value_counts()
    ctx.publish("X2 post-cutoff questions", [
        {"type": "kpis", "items": [
            {"label": "questions kept", "value": len(qs), "fmt": "int"},
            {"label": "papers with questions", "value": int(qs["paper_key"].nunique()), "fmt": "int"},
            {"label": "kept share", "value": len(qs) / max(len(allq), 1), "fmt": "pct1"},
            {"label": "generator", "value": f"{gen['name']} (family used nowhere else)"}]},
        {"type": "bar", "title": "Verification outcome", "horizontal": True, "fmt": "int",
         "labels": list(st.index), "series": [{"name": "questions", "data": st.tolist()}]},
    ], summary=f"{len(qs)} questions on {qs['paper_key'].nunique()} papers")
    return {"questions": len(qs)}


@stage("x2", "verify", needs=("questions",), config_keys=("x2.verify",),
       help="human-verification packet for a random subset of the post-cutoff questions")
def run_verify(ctx: Context) -> dict:
    n = int(setting(ctx, "verify", ctx.section("x2.verify").get("n", 100)))
    qs = stage_table(ctx, "questions", "questions.parquet")
    papers = stage_table(ctx, "papers", "papers.parquet")[["paper_key", "arxiv_id", "title"]]
    s = qs.sample(n=min(n, len(qs)), random_state=0).merge(papers, on="paper_key")
    s["link"] = "https://arxiv.org/abs/" + s["arxiv_id"]
    cols = ["qid", "title", "link", "question", "answer", "evidence"]
    s[cols].assign(answerable_from_paper="", answer_correct="", self_contained="", notes="") \
        .to_csv(ctx.out / "verify.csv", index=False)
    with open(ctx.out / "verify.md", "w") as fh:
        fh.write("# X2 human verification\nFor each question: is it answerable from the paper, is the answer correct, "
                 "is it self-contained (names the study)? Fill the columns in verify.csv.\n\n")
        for r in s.itertuples():
            fh.write(f"## {r.qid}\n[{r.title}]({r.link})\n\n**Q:** {r.question}\n\n**A:** {r.answer}\n\n"
                     f"**Evidence:** {r.evidence}\n\n")
    done = ctx.out / "verify_filled.csv"                          # put the filled-in sheet here to get the rates
    panels = [{"type": "kpis", "items": [{"label": "questions in the packet", "value": len(s), "fmt": "int",
                                          "hint": "verify.csv / verify.md; fill and save as verify_filled.csv"}]}]
    if done.exists():
        f = pd.read_csv(done)
        panels.append({"type": "kpis", "items": [
            {"label": c.replace("_", " "), "value": float(f[c].astype(str).str.lower().isin(["1", "y", "yes", "true"]).mean()),
             "fmt": "pct1"} for c in ("answerable_from_paper", "answer_correct", "self_contained") if c in f]})
    ctx.publish("X2 human verification", panels, summary=f"{len(s)} questions to verify")
    return {"packet": len(s)}
