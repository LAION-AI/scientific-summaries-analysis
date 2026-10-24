"""Benchmarks for WS3: download (login node) and the `benchmarks` stage that builds one item table.

    alexandria fetch-benchmarks          # login node (compute nodes are offline) -> $STORE/benchmarks/

Item table columns: bench, item_id, question, answer, answer_type (short | label | mc), options (labels / letters),
cluster (bootstrap unit: the gold paper for the QA bench, the item otherwise), gold_refs (DOIs / titles of the
paper(s) the item is about), gold_paper_key / gold_file / gold_row / gold_match (the linked dataset row, see
ws3.gold; None = gold paper not in the dataset), category, judge (simpleqa | exact | hle).

  qabench-open  WS2 questions in open-corpus mode: the gold paper must be found in the index (bridges WS2 and WS3).
  pubmedqa      PubMedQA pqa_labeled, question only (no abstract): yes / no / maybe.
  litqa2        LAB-Bench LitQA2: multiple choice incl. "Insufficient information", options shuffled per item (seeded).
  scifact       SciFact dev claims with a SUPPORT / CONTRADICT label: true / false.
  hle           Humanity's Last Exam, text-only, graded with the official HLE judge prompt (gated: needs HF_TOKEN).
  hle-gold      futurehouse/hle-gold-bio-chem (gated).
  gpqa          GPQA-Diamond, options shuffled per item (gated).
"""
from __future__ import annotations

import json
import os
import random
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

from ..runtime import Context, log, stage
from .gold import dois_in, id_lookup, link

SOURCES = {
    "pubmedqa": {"repo": "qiaojin/PubMedQA", "files": ["pqa_labeled/train-00000-of-00001.parquet"], "gated": False},
    "litqa2": {"repo": "futurehouse/lab-bench", "files": ["LitQA2/train-00000-of-00001.parquet"], "gated": False},
    "scifact": {"url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz", "gated": False},
    "hle": {"repo": "cais/hle", "files": ["data/test-00000-of-00001.parquet"], "gated": True},
    "hle-gold": {"repo": "futurehouse/hle-gold-bio-chem", "files": ["data/test-00000-of-00001.parquet"], "gated": True},
    "gpqa": {"repo": "Idavidrein/gpqa", "files": ["gpqa_diamond.csv"], "gated": True},
}
INSUFFICIENT = "Insufficient information to answer the question"


def bench_dir(cfg) -> Path:
    return Path(cfg.site.store) / "benchmarks"


def fetch(cfg, names: list[str] | None = None) -> dict[str, str]:
    """Download benchmark files (public ones always; gated ones when HF_TOKEN grants access)."""
    from huggingface_hub import hf_hub_download
    out, root = {}, bench_dir(cfg)
    for name, src in SOURCES.items():
        if names and name not in names:
            continue
        dst = root / name
        dst.mkdir(parents=True, exist_ok=True)
        try:
            if "url" in src:
                tgz = dst / "data.tar.gz"
                if not tgz.exists():
                    urllib.request.urlretrieve(src["url"], tgz)
                with tarfile.open(tgz) as t:
                    for m in t.getmembers():
                        if Path(m.name).name in ("claims_dev.jsonl", "corpus.jsonl"):
                            m.name = Path(m.name).name
                            t.extract(m, dst)
            else:
                for f in src["files"]:
                    p = hf_hub_download(src["repo"], f, repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                                        cache_dir=str(cfg.site.hf_home / "hub"))
                    (dst / Path(f).name).unlink(missing_ok=True)
                    (dst / Path(f).name).symlink_to(p)
            if name == "pubmedqa":
                _fetch_pubmed_meta(dst)
            out[name] = "ok"
        except Exception as e:                       # noqa: BLE001  (gated / offline: reported, not fatal)
            out[name] = f"not available: {type(e).__name__}" + (" (gated: accept the terms on huggingface.co "
                                                                 "and set HF_TOKEN)" if src.get("gated") else "")
        log(f"fetch-benchmarks: {name}: {out[name]}")
    return out


def _fetch_pubmed_meta(dst: Path, batch: int = 200) -> None:
    """PMID -> title, DOI from NCBI E-utilities (public; login node only): lets PubMedQA items link to their paper."""
    out = dst / "pmid_meta.json"
    if out.exists():
        return
    pmids = [str(x) for x in pd.read_parquet(dst / "train-00000-of-00001.parquet")["pubid"]]
    meta = {}
    for i in range(0, len(pmids), batch):
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
               + ",".join(pmids[i:i + batch]))
        with urllib.request.urlopen(url, timeout=60) as r:
            res = json.load(r)["result"]
        for u in res.get("uids", []):
            ids = {a["idtype"]: a["value"] for a in res[u].get("articleids", [])}
            meta[u] = {"title": res[u].get("title"), "doi": ids.get("doi"), "pmc": ids.get("pmc")}
        time.sleep(0.4)                                   # NCBI: <= 3 requests / s without an API key
    out.write_text(json.dumps(meta))


# ---------------------------------------------------------------- adapters: each -> list of item dicts
def _qabench(ctx: Context) -> list[dict]:
    p = ctx.cfg.stage_dir("ws2", "questions") / "questions.parquet"
    if not p.exists():
        raise FileNotFoundError("run ws2 questions first")
    q = pd.read_parquet(p)
    return [{"item_id": r.qid, "question": r.question, "answer": r.answer, "answer_type": "short", "options": [],
             "cluster": r.paper_key, "gold_paper_key": r.paper_key, "gold_refs": [], "category": r.qtype,
             "judge": "simpleqa", "gold_match": "ws2"} for r in q.itertuples()]


def _pubmedqa(root: Path) -> list[dict]:
    df = pd.read_parquet(root / "pubmedqa" / "train-00000-of-00001.parquet")
    mp = root / "pubmedqa" / "pmid_meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return [{"item_id": f"pubmedqa:{r.pubid}", "question": r.question, "answer": r.final_decision,
             "answer_type": "label", "options": ["yes", "no", "maybe"], "cluster": f"pubmedqa:{r.pubid}",
             "gold_paper_key": None, "gold_refs": [meta[str(r.pubid)]] if str(r.pubid) in meta else [],
             "category": "biomedical", "judge": "exact"} for r in df.itertuples()]


def _mc(item_id: str, question: str, correct: str, wrong: list[str], seed: int, extra: list[str] | None = None):
    opts = [correct, *wrong]
    random.Random(f"{seed}:{item_id}").shuffle(opts)
    opts += extra or []
    letters = [chr(65 + i) for i in range(len(opts))]
    text = question + "\n\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, opts, strict=True))
    return text, letters, letters[opts.index(correct)]


def _litqa2(root: Path, seed: int) -> list[dict]:
    df = pd.read_parquet(root / "litqa2" / "train-00000-of-00001.parquet")
    out = []
    for r in df.itertuples():
        text, letters, gold = _mc(str(r.id), r.question, str(r.ideal), [str(x) for x in r.distractors], seed,
                                  [INSUFFICIENT])
        out.append({"item_id": f"litqa2:{r.id}", "question": text, "answer": gold, "answer_type": "mc",
                    "options": letters, "cluster": f"litqa2:{r.id}", "gold_paper_key": None,
                    "gold_refs": [{"doi": d} for d in dois_in(r.sources)],
                    "category": "literature", "judge": "exact", "abstain_option": letters[-1]})
    return out


def _scifact(root: Path) -> list[dict]:
    cp = root / "scifact" / "corpus.jsonl"
    titles = {str(d["doc_id"]): d["title"] for d in map(json.loads, cp.read_text().splitlines())} if cp.exists() else {}
    out = []
    for line in (root / "scifact" / "claims_dev.jsonl").read_text().splitlines():
        c = json.loads(line)
        labels = {s["label"] for ev in (c.get("evidence") or {}).values() for s in ev}
        if len(labels) != 1:
            continue                                   # NEI or mixed: no single truth value
        truth = "true" if labels == {"SUPPORT"} else "false"
        out.append({"item_id": f"scifact:{c['id']}", "question": f"Is this scientific claim true or false? "
                    f"Claim: {c['claim']}", "answer": truth, "answer_type": "label", "options": ["true", "false"],
                    "cluster": f"scifact:{c['id']}", "gold_paper_key": None,
                    "gold_refs": [{"title": titles[d]} for d in (c.get("evidence") or {}) if d in titles],
                    "category": "claim", "judge": "exact"})
    return out


def _hle(root: Path, name: str) -> list[dict]:
    df = pd.read_parquet(root / name / "test-00000-of-00001.parquet")
    if "image" in df:
        df = df[df["image"].fillna("") == ""]                     # text-only
    return [{"item_id": f"{name}:{r.id}", "question": r.question, "answer": str(r.answer),
             "answer_type": "short", "options": [], "cluster": f"{name}:{r.id}", "gold_paper_key": None,
             "category": getattr(r, "category", None) or getattr(r, "raw_subject", "hle"), "judge": "hle"}
            for r in df.itertuples()]


def _gpqa(root: Path, seed: int) -> list[dict]:
    df = pd.read_csv(root / "gpqa" / "gpqa_diamond.csv")
    out = []
    for i, r in df.iterrows():
        rid = str(r.get("Record ID", i))
        text, letters, gold = _mc(rid, r["Question"], str(r["Correct Answer"]),
                                  [str(r[f"Incorrect Answer {k}"]) for k in (1, 2, 3)], seed)
        out.append({"item_id": f"gpqa:{rid}", "question": text, "answer": gold, "answer_type": "mc",
                    "options": letters, "cluster": f"gpqa:{rid}", "gold_paper_key": None,
                    "category": r.get("High-level domain", "gpqa"), "judge": "exact"})
    return out


@stage("ws3", "benchmarks", needs=(), config_keys=("ws3.benchmarks",),
       help="benchmark items (QA bench open-corpus, PubMedQA, LitQA2, SciFact; HLE / GPQA when available)")
def run(ctx: Context) -> dict:
    cfg = ctx.section("ws3.benchmarks")
    root = bench_dir(ctx.cfg)
    seed = int(cfg.get("seed", 0))
    n = cfg.get("max_items")
    if ctx.cfg.profile == "debug":
        n = int(ctx.cfg.profile_settings.get("ws3_items", 100))
    loaders = {"qabench-open": lambda: _qabench(ctx), "pubmedqa": lambda: _pubmedqa(root),
               "litqa2": lambda: _litqa2(root, seed), "scifact": lambda: _scifact(root),
               "hle": lambda: _hle(root, "hle"), "hle-gold": lambda: _hle(root, "hle-gold"),
               "gpqa": lambda: _gpqa(root, seed)}
    items, status = [], []
    for name in cfg.get("names", list(loaders)):
        try:
            its = loaders[name]()
        except FileNotFoundError as e:                  # missing files only; a loader bug must fail the stage
            gated = (SOURCES.get(name) or {}).get("gated")
            status.append({"bench": name, "items": 0, "status": "not run: gated, needs HF_TOKEN" if gated
                           else "not run: not downloaded (alexandria fetch-benchmarks)", "detail": str(e)[:200]})
            continue
        df = pd.DataFrame(its).sample(frac=1.0, random_state=seed)
        if n and len(df) > int(n):                               # spread over clusters (papers) first
            df = df.assign(_r=df.groupby("cluster").cumcount()).sort_values(["_r"], kind="stable").head(int(n))
            df = df.drop(columns="_r")
        df["bench"] = name
        items.append(df)
        status.append({"bench": name, "items": len(df), "status": "ok"})
    all_items = pd.concat(items, ignore_index=True) if items else pd.DataFrame()
    for c in ("abstain_option", "gold_refs", "gold_match"):
        if c not in all_items:
            all_items[c] = None
    all_items["gold_refs"] = [r if isinstance(r, list) else [] for r in all_items["gold_refs"]]
    qa = all_items["bench"] == "qabench-open"
    if qa.any():                                             # QA bench: the WS2 paper is the gold paper
        wp = pd.read_parquet(ctx.cfg.stage_dir("ws2", "papers") / "papers.parquet").set_index("paper_key")
        all_items.loc[qa, "gold_file"] = all_items.loc[qa, "gold_paper_key"].map(wp["file"])
        all_items.loc[qa, "gold_row"] = all_items.loc[qa, "gold_paper_key"].map(wp["row"])
    if cfg.get("link_gold", True) and all_items["gold_refs"].map(len).any():
        mw = int(cfg.get("min_title_words", 5))
        all_items = link(all_items, id_lookup(ctx.cfg, mw), mw)
    all_items["gold_row"] = pd.array(all_items.get("gold_row"), dtype="Int64")
    for s in status:
        b = all_items[all_items["bench"] == s["bench"]]
        s["gold_refs"] = int(b["gold_refs"].map(len).gt(0).sum()) if len(b) else 0
        s["gold_linked"] = int(b["gold_paper_key"].notna().sum()) if len(b) else 0
    all_items.to_parquet(ctx.out / "items.parquet", index=False)
    ctx.publish("WS3 benchmarks", [
        {"type": "table", "title": "Benchmarks", "columns": [
            {"key": "bench", "label": "benchmark"}, {"key": "items", "label": "items", "fmt": "int"},
            {"key": "gold_refs", "label": "items naming a paper", "fmt": "int"},
            {"key": "gold_linked", "label": "gold paper in dataset", "fmt": "int"},
            {"key": "status", "label": "status"}], "rows": status,
         "hint": "gold paper in dataset = linked by DOI or exact title to a row of the release; only these items can "
                 "be helped by finding their paper (all items stay in the benchmark)"}],
        summary=f"{len(all_items)} items in {sum(s['items'] > 0 for s in status)} benchmarks")
    return {"items": len(all_items), "benchmarks": {s["bench"]: s["items"] for s in status}}
