# Alexandria ICML 2027

Experiments on structured-summary datasets, by default
[laion/Scientific-Summaries](https://huggingface.co/datasets/laion/Scientific-Summaries) (revision `232547b7`):
22.1M papers, each with one **knowledge unit (KU)**, an LLM-written (Gemini 2.0 Flash Lite) 18-field structured
summary, plus OpenAlex metadata and, for ~5.3M papers, the full text. One workstream (WS) per sub-package:

| WS | question | package | config |
|---|---|---|---|
| **WS1** | Is the dataset what it claims to be? Data-quality audit, dataset description, datasheet; **WS1b**: are the KUs faithful to their papers? | `alexandria.ws1` | `configs/ws1.yaml` |
| **WS2** | Given the paper, how much of what a reader needs does the KU keep, compared with the full text, at what token cost? (fidelity / retention, gold paper given) | `alexandria.ws2` | `configs/ws2.yaml` |
| **WS3** | Without the paper, does access to the KU corpus through a retrieval skill help models on scientific benchmarks? (usefulness, open corpus) | `alexandria.ws3`, `alexandria.skill` | `configs/ws3.yaml` |
| **WS4** | Are KU-derived data a good **training** signal that pushes models (post-training: SFT, GRPO, tool-use RL), compared with full papers and controls? | `alexandria.ws4` | `configs/ws4.yaml` |
| **WS5** | Can a 9-12B student distilled from teacher KUs (open-weight Qwen3.5-27B; optionally Claude Opus 5.5) match the teacher at a fraction of the cost, on disjoint papers? | `alexandria.ws5` | `configs/ws5.yaml` |
| **WS6** | Extreme compression: can a projected KU embedding (a few "knowledge tokens") replace the text? (P2, optional) | `alexandria.ws6` | `configs/ws6.yaml` |

Contents: [Layout](#repository-layout) · [Setup](#setup-once) · [How stages run](#how-stages-run) ·
[Dashboard](#dashboard) · [WS1](#ws1-data-quality-description-and-faithfulness) · [WS2](#ws2-representation-comparison-on-the-qa-bench) ·
[WS3](#ws3-benchmarks-via-the-alexandria-knowledge-skill) · [WS4](#ws4-training-signal-study-post-training-first) · [WS5](#ws5-teacher-distillation) · [WS6](#ws6-embeddings-as-input) · [X1](#x1-retrieval-evaluation-of-the-indices) · [X2](#x2-post-cutoff-benchmark) · [X3](#x3-comparison-with-existing-systems) · [S1](#s1-safety-checks-and-evaluation) · [ICML run protocol](#icml-run-protocol-debug--full) ·
[Threats to validity](#threats-to-validity-and-open-items) · [Datasets](#datasets-run-on-any-data) · [Data notes](#data-notes)

## Repository layout

```
configs/
  site.yaml              machine paths, SLURM resources, global vLLM overrides (site.llm.*)
  models.yaml            every LLM / embedder by name, pinned HF revision, family, size
  datasets/<name>.yaml   where a dataset lives, column mapping, link rules, extras (KU-Fixes)
  ws1.yaml ws2.yaml ws3.yaml   every threshold / setting of a workstream; `profiles:` = debug vs full
src/alexandria/
  cli.py slurm.py runtime.py config.py data.py   stage registry, CLI, SLURM submission, manifests, data access
  schema/                canonical 44-column schema, 18 KU fields, `load_summary` normalizer
  llm.py qa.py stats.py  vLLM helpers, QA prompts + grading, intervals / bootstrap / Holm
  ws1/ ws2/ ws3/         the stages of each workstream
  skill/                 the Alexandria knowledge skill: retrieval backend, MCP server, tool schemas, SKILL.md
  dashboard/ report.py   live dashboard, per-WS PDF report
slurm/stage.sbatch       generic job script (conda env + `alexandria <ws> <stage> ...`)
slurm/ws2_ws3_debug.sh   the WS2 + WS3 chain (debug or full profile)
tests/                   unit + small end-to-end tests (no cluster, no real data needed)
```

## Setup (once)

| conda env | path | used by |
|---|---|---|
| `Alex` | `$STORE/envs/Alex` (Python 3.10, torch 2.8) | every CPU stage, tests, dashboard, index building |
| `Alex-vllm` | `$STORE/envs/Alex-vllm` (Python 3.11, vLLM 0.32, torch + CUDA 13) | every LLM stage (`grounding`, `judge`, `questions`, `answer`, `grade`, `pilot`, `agent`) |
| `Alex-train` | `$STORE/envs/Alex-train` (clone of Alex-vllm + TRL 1.16, PEFT 0.21, datasets 5.1, lm-eval 0.4.13, math-verify) | WS4 `train-*`, `eval` |

`$STORE` = `/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam`. Both envs have this package installed editable
(`pip install -e .`). Compute nodes have **no internet**, so everything is downloaded on a login node first:

```bash
conda activate Alex
alexandria fetch-models          # every model in configs/models.yaml (pinned revisions) -> $STORE/hf_cache
alexandria fetch-benchmarks      # PubMedQA (+ PMID -> title/DOI from NCBI), LitQA2, SciFact (+ corpus) -> $STORE/benchmarks
alexandria fetch-ws4             # WS4: training models, replay / control data, eval files, lm-eval datasets, NLTK data
HF_TOKEN=hf_... alexandria fetch-benchmarks hle hle-gold gpqa   # gated sets: accept their terms on huggingface.co first
alexandria sample                # debug subset: 2 shards x 60 rows per source group -> runs/debug/_sample
python -m pytest -q              # 82 tests, ~1 min
alexandria list                  # every stage: kind, resources, env, what it needs
```

Models (`configs/models.yaml`, revisions pinned): answerers / agents **Qwen2.5-7B-Instruct, OLMo-3-7B-Instruct,
Granite-3.3-8B-Instruct, Phi-4-mini-instruct** (4 families, 3.8-8.2B; OLMo-3 has open training data); question
generator, WS2 / WS3 judge and WS1b judge **Qwen3-8B** (thinking off); WS1b grounding **Bespoke-MiniCheck-7B**;
embedder **BAAI/bge-m3** (dense, CLS pooling).

## How stages run

* A **stage** is one function registered with `@stage(ws, name, needs=..., resources=cpu|gpu, env=main|llm)`. It runs
  on its own, reads only files earlier stages wrote, and writes `<out_root>/<profile>/<ws>/<stage>/`:
  `manifest.json` (config fingerprint, code version, host, SLURM job, timing, status done / failed / skipped,
  vLLM overrides), `dashboard.json` (panels for the dashboard and the PDF), and its data files.
* **Profiles**: `debug` reads the stored sample and caps the work (WS1 GPU stages 100 papers, WS2 100 papers, WS3 100
  items per benchmark); `full` reads the whole release with the settings in each `configs/ws*.yaml`.
* **Map** stages (WS1 `features`, `lexical`, `neardup-sign`; WS2 `answer`; WS3 `pilot`, `agent`) split their work
  over SLURM array tasks (`--array N`: shards for WS1, models for WS2 / WS3) and resume: finished parts are skipped
  unless `--force`, and everything is recomputed when the stage's config fingerprint changed.
* Any value can be overridden: `--set ws2.answer.models=[qwen2.5-7b]`, `--set site.llm.quantization=null`.

```bash
alexandria ws2 contexts --profile debug                      # one stage, here
alexandria pipeline ws1 features keys quality --profile debug    # several, in order, one process
alexandria submit ws1 quality --profile full --with-deps     # SLURM chain by `needs` (afterok)
alexandria submit ws2 answer --profile debug --array 4       # one array task per answerer model
alexandria submit ws2 results report --one-job --after <jobid>   # several stages in one allocation
alexandria submit ws3 agent --dry-run ...                    # print the sbatch command only
```

**Cluster (GWDG Grete).** Account `kisski-l3s-llm4sci` may only use `grete:interactive` (max 2 submitted jobs per
user, 12 h) and `grete:preemptible`; CPU partitions are not allowed, so CPU stages run on a small MIG slice. SCI-JEPA
chains often hold both interactive slots, so Alexandria jobs go to `grete:preemptible`
(`--set site.slurm.<cpu|gpu|debug>.partition=grete:preemptible`; `slurm/ws2_ws3_debug.sh` does this). MIG slices:
`1g.10gb`, `2g.20gb` (plentiful), `3g.40gb` (scarce), A100 without native FP8 (fp8 = weight-only Marlin kernels).

## Dashboard

```bash
alexandria dashboard serve --profile debug --rebuild-every 30   # http://localhost:8770
```

One page for the whole project: Overview (pipeline graph with every stage's status), one tab per WS (every stage's
panels: KPIs, tables, charts, figures), Runs (all manifests and errors), and links to the latest PDF of each WS.
Stages describe their panels as JSON (`src/alexandria/dashboard/PANELS.md`), so a new WS appears without dashboard
code. VS Code forwards port 8770 **only from the login node it is connected to** (glogin9 / glogin10 / ...): run the
server on that node (`hostname`; `ss -ltnp | grep 8770`), or `ssh -L 8770:localhost:8770 <login node>`. To keep it
running after the terminal closes: `setsid nohup alexandria dashboard serve ... > $STORE/logs/alexandria_dashboard.log 2>&1 &`.
`data.json` is strict JSON (NaN / Infinity written as null): a browser rejects anything else.

Each WS also has a `report` stage: `<run root>/<ws>/report/<WS>_report_<dataset>_<profile>.pdf` with the stage
status and every stage's KPIs, tables, charts and figures; per-paper tables list papers with links, no text excerpts.

## WS1: data quality, description and faithfulness

| stage | kind | needs | what it produces |
|---|---|---|---|
| `features` | map | - | one row per paper: field status / lengths / truncation / refusals / leaks, language, wrong-paper evidence, garbled source, input type, claim structure, KU-Fixes flags (reused), metadata |
| `keys` | reduce | - | column names and types per shard, claim sub-keys (KU-Fixes stats), leaked keys, `alias_candidates.csv`, `summary.schema.json`, normalizer check |
| `lexical` / `lexical-report` | map / reduce | features | verbatim overlap (longest common span, >= 8/12/20 tokens), number and entity precision vs the source |
| `neardup-sign` / `neardup` | map / reduce | - | MinHash-LSH near-identical summaries, split into known paper duplicates vs different papers |
| `quality` | reduce | features | the flag table with Wilson 95% intervals per group and model, field status, examples, `flags_table.tex`, figure |
| `describe` | reduce | features | discipline, year, language, OA, input type, model, field lengths, citations, retractions; main-text figure |
| `reconcile` | reduce | - | published numbers vs measured ones; each explanation is checked by arithmetic, residual shown |
| `provenance` | reduce | features | summarizer model, tokens, cost estimate, prompt overhead, failure rate; prompt / temperature from config |
| `human-sample` / `human-agreement` | reduce | features | stratified annotation packets + guidelines; Krippendorff's alpha and gold labels from filled-in sheets |
| `grounding` / `grounding-report` | reduce (GPU) | features | MiniCheck-7B sentence-level grounding of KU sentences against the source |
| `judge` / `judge-report` | reduce (GPU) | features | Qwen3-8B sentence-level judge (evidence, then verdict per sentence; section = worst sentence), same papers as `grounding`; agreement with MiniCheck |
| `calibrate` | reduce | judge, human-agreement | judge and MiniCheck vs human gold: kappa, confusion, AUC, threshold, isotonic map |
| `abstract-support` | reduce | features | what abstract-only input can support of the long methods / results fields |
| `datasheet` | reduce | describe, quality, reconcile, provenance, keys | Gebru et al. datasheet + Croissant RAI, from measured numbers |
| `audit` | reduce | - | paper consistency audit (plan section 5), over the WHOLE local release for light columns: arXiv rows vs distinct arXiv ids, rows without an id, versions, cross-subset dedup method (KU-Fixes clusters by match keys; title-only = possible false positives), index arithmetic (PQ48 = 64x), real token statistics (summariser `summary_tokens_out` + per-field reference tokens), model sizes for the paper's tables, input type (full text vs abstract), what "sanitized" removes, chunk source, decontamination hits of summaries vs every eval set, a seeded random sample for the embedding plot, and a checklist of paper statements vs measured evidence (`paper_checklist.csv`) |
| `licenses` | reduce (login node) | - | per-record licence from OpenAlex + arXiv OAI-PMH, classified (permissive / NC-ND / arXiv default / publisher / none), `fulltext_redistributable`, `cc_by_conflict` -> `licenses.parquet` |

Methods: rates carry Wilson 95% intervals; per-paper length flags use a robust z-score of log(1 + words) against the
delivered lengths of the same field and input type (Iglewicz & Hoaglin, |z| > 3.5); grounding follows MiniCheck
(Tang et al., 2024) at sentence level; the LLM judge uses the human protocol's label set so `calibrate` can map both
to human gold. **Reused, not recomputed:** OpenAlex metadata, the KU-Fixes fault index (placeholder titles / authors,
needs_regeneration, claims JSON faults) and KU-Fixes paper duplicates. Current decisions (2026-10-21): MiniCheck and
judge on 100 papers (debug and full), human labelling off (`ws1b.human.enabled: false` -> `human-sample`,
`human-agreement`, `calibrate` are `skipped`).

**Shared normalizer for every WS:** `from alexandria.schema import load_summary` gives the one cleaned view of a KU
(reviewed alias table `src/alexandria/resources/aliases.yaml`, placeholders nulled, claims parsed, every repair listed
in `issues`).

```bash
alexandria submit ws1 all --profile debug --one-job                       # every CPU stage, one allocation
alexandria submit ws1 grounding judge --profile debug --one-job --after <cpu job id>
alexandria submit ws1 features --profile full --array 16                  # full: shards over 16 array tasks
```

## WS2: representation comparison on the QA bench

**Question.** With the gold paper given, how well does a reader model answer paper-specific questions from (a) no
context, (b) the full text, (c) the full text cut to the KU's token count, (d) the KU as released, (e) the KU after
cleaning? Measures fidelity / retention and compression; usefulness without the gold paper is WS3.

**Unit of representation.** In Scientific-Summaries one row = one paper = one KU: the 18-field structured summary
(`SUMMARY_FIELDS`; the KU-Fixes errata count 22.1M KUs for 22.1M papers). WS2 does **not** generate KUs; the `ku`
stage takes each paper's KU from its row and enforces exactly one KU per paper.

| stage | GPU | what |
|---|---|---|
| `papers` | | evaluation papers: summary written from full text, readable source (not `source_garbled`), not a wrong-paper suspect, >= 5,000 source characters, usable KU (not KU-Fixes `needs_regeneration`), not a dropped duplicate copy; stratified by domain; WS1-flagged papers over-sampled to 30% (enough affected papers for the cleaning comparison); deterministic (seed 0) |
| `questions` | yes | Qwen3-8B writes 10 candidate short-answer questions per paper from the source (JSON with answer, type, exact evidence quote; vLLM structured outputs); kept (<= 5 / paper) only if: evidence quote found in the source (rapidfuzz partial ratio >= 90), answer <= 15 words and not a notation / numbering reference, no >= 8-word run copied from the source, not a near-duplicate, and an LLM critic (sees question + answer + evidence, not the paper) confirms it identifies the study, has one answer, is not trivial and needs the paper; every candidate's outcome is stored (`all_generated.parquet`) |
| `ku` | | one KU per paper: the 18 fields as released (`ku_raw`) and with the official KU-Fixes applied (`ku_fixed`: title / authors fixes, claims sub-keys normalised), plus fault flags, needs_regeneration, duplicate tier; no LLM |
| `contexts` | | per paper and arm the answerer's context and its token count (one reference tokenizer, Qwen2.5): `closed_book` (none), `full_text` (first 30k characters), `full_text_matched` (source cut to the `ku_clean` token count: compression control), `ku_raw` (KU as released, 18 labelled fields), `ku_clean` (KU-Fixes + `load_summary`: placeholders removed, markup stripped, claims as a list); every context capped at `max_context_tokens`; question-context 5-gram overlap per arm (lexical-bias covariate) |
| `answer` | yes | 4 answerer families, identical system prompt (short answer or exactly "I don't know"), context first and question last (prefix caching), temperature 0, max 64 tokens; one array task per model |
| `grade` | yes | SimpleQA protocol (Wei et al., 2024): a fixed judge (Qwen3-8B) returns CORRECT / INCORRECT / NOT_ATTEMPTED (numbers to the gold answer's precision); plus model-free normalised exact match and token F1 |
| `results` | | accuracy = CORRECT / all with 95% paper-clustered bootstrap CIs (2,000 resamples, same resample for every arm and model, so differences are paired); Δ vs closed book and vs full text with Holm-adjusted p-values; strata: context-dependent questions (wrong closed-book), before / after cleaning (`ku_clean − ku_raw`) on WS1-flagged vs unflagged papers, question type, domain, accuracy vs context tokens (compression), help / harm flips, accuracy by question-context overlap tercile; `accuracy_table.tex` |
| `report` | | WS2 PDF |

**Bias controls.** The question generator (Qwen3) is a different family from three of four answerers and is also the
judge (stated in the paper; the model-free EM / F1 is the check). Paraphrase rule + overlap covariate address the
full-text arm's lexical advantage on questions written from the source. `full_text_matched` separates "the KU is
better written" from "the KU is shorter". Papers are the bootstrap unit, because questions of one paper are not
independent.

**Debug result (2026-10-22, 100 papers, 418 questions, 4 models):** closed book 1-5%, full text 67-73%, `ku_raw`
52-54%, `ku_clean` 51-55%, `full_text_matched` 45-49%: at equal tokens the KU beats truncated full text by 5-9
points; cleaning has no measurable effect.

## WS3: benchmarks via the Alexandria knowledge skill

**Question.** Without the paper, does a model answer scientific benchmark questions better when it can search the KU
corpus (or the full texts) than alone, than with self-consistency at matched compute, and than with plain RAG?

**The skill** (`alexandria.skill`): one retrieval backend exposed three ways: an MCP server
(`alexandria skill mcp`), a `SKILL.md`, and OpenAI-style function schemas. Tools: `search(query, field?, k)` over the
KU fields and claims, `get_paper(id, fields?)`, `search_fulltext(query, k)` over source-text chunks. Retrieval is
hybrid: BM25 (Robertson & Zaragoza; Lucene idf, k1 = 0.9, b = 0.4, Pyserini defaults) + bge-m3 dense (Chen et al.,
2024), fused with reciprocal rank fusion (Cormack et al., 2009; k = 60, depth 100). Every tool response is cached by
(index version, view, tool, arguments), so every run replays exactly and the cache can be released with the results.

| stage | GPU | what |
|---|---|---|
| `benchmarks` | | one item table: **qabench-open** (WS2 questions without the paper: the paper must be found), **PubMedQA** (pqa_labeled, question only, yes / no / maybe; Jin et al., 2019), **LitQA2** (LAB-Bench; multiple choice + "Insufficient information", options shuffled per item; Laurent et al., 2024), **SciFact** (dev claims with a single SUPPORT / CONTRADICT label -> true / false; Wadden et al., 2020), **HLE** text-only / **HLE-gold** / **GPQA-Diamond** (gated); then gold-paper linking (below) |
| `index` | yes | corpus = profile's papers + the linked gold papers, minus a blocklist of benchmark-defining papers (HLE, GPQA, LAB-Bench, SciFact, PubMedQA); units per view: `summary` (one per cleaned KU field + one per claim with its evidence), `fulltext` (300-word chunks, 50 overlap, <= 20k words / paper); BM25 + bge-m3 embeddings (fp16, L2-normalised) per view; version hash |
| `pilot` | yes | tool-call reliability of each model **without** constrained decoding (20 calls per benchmark); gate >= 95% well-formed |
| `agent` | yes | one harness for every arm (below); one array task per model |
| `grade` | yes | qabench-open: SimpleQA judge (Qwen3-8B); labels / option letters: exact match; HLE: official HLE judge prompt (Phan et al., 2025); abstentions recorded |
| `results` | | accuracy per benchmark x arm x model with 95% cluster-bootstrap CIs (clusters = gold papers for the QA bench, items otherwise), paired Δ vs A0 with Holm correction, help / harm flips vs A0, gold-paper-retrieved rate, tool calls, retrieved tokens, prompt tokens and GPU-seconds per question, LitQA2 precision (correct / answered), every benchmark also on its **[gold in corpus]** subset, pilot reliability table with gate pass / FAIL, headline figure Δ accuracy over A0 vs model size; `main_table.tex` |
| `report` | | WS3 PDF |

**Arms** (`ws3.agent.arm_defs`):

| arm | what | context the model gets |
|---|---|---|
| `A0` | model alone, one answer step | question only |
| `A0+` | self-consistency (Wang et al., 2023): 5 samples at T = 0.7, majority vote | compute-matched no-retrieval baseline |
| `R` | plain RAG: top-20 hybrid passages from full text + KU fields, pasted once | up to the retrieved-token budget |
| `F` | agent with `search_fulltext` | full-text chunks |
| `S` | agent with `search` + `get_paper` over the KUs | KU fields and claims |
| `S-OA` | as `S`, restricted to papers that have full text | separates "KU" from "open-access subset" |
| `Oracle` | the gold paper's text (QA bench only) | upper bound |
| `W`, `W-sch`, `S+W` | web search arms | compute nodes are offline: reported as not run |

**Harness.** Same system prompt for every arm (only tool descriptions differ), same limits (<= 6 tool calls, <= 8k
retrieved tokens, <= 400 tokens per step), same answer format. Each turn is one JSON action `{thought, action,
arguments}` with generation constrained to the arm's tool schemas, so tool-call formatting cannot confound the
comparison across model families; `pilot` reports each model's unconstrained reliability. The bge-m3 query encoder
runs on CPU so vLLM gets the whole GPU slice. Every episode logs tool calls, retrieved tokens, prompt / output
tokens, GPU-seconds and the papers it saw.

**Gold-paper linking** (`ws3/gold.py`). A retrieval arm can only be helped by finding the paper when that paper is in
the corpus, and the gold-paper-retrieved rate needs to know which row it is. Items carry `gold_refs`: LitQA2 DOIs
(`sources`), PubMedQA PMID -> title + DOI (NCBI E-utilities, fetched on the login node), SciFact cited corpus
documents -> titles. They are matched to release rows by normalised DOI, else by exact normalised title (>= 5 words);
rows with full text win. The lookup over all release rows (metadata columns only) is built once and cached in
`$STORE/Alexandria-ICML-2027/assets/id_lookup_<dataset>_w5.parquet`. Unlinked items stay in the benchmark (all-item
accuracy) and are excluded from the `[gold in corpus]` subset.

**Debug linkage (local release = arXiv + PubMed subsets only):** qabench-open 100/100, SciFact 17/100, PubMedQA
2/100, LitQA2 0/100 (its papers are 2024 journal / bioRxiv papers, in the web-papers / bethgelab subsets that are not
on local storage). **Pilot:** Qwen2.5 97.5%, Granite 95%, OLMo 91%, Phi-4-mini 32.5% well-formed (OLMo and Phi
FAIL the gate; their agent results are reported with constrained decoding only).

**Debug result (2026-10-22, 4 models, ~500-paper corpus; not paper numbers):**

| benchmark | A0 | R (plain RAG) | S (agent over KUs) | F (agent over full text) | gold paper found (R / S) |
|---|---|---|---|---|---|
| qabench-open | 2-7% | 62-63% | 20-34% | 15-34% | 98% / 67% (Oracle 65-72%) |
| SciFact [gold in corpus], 17 items | 47-76% | 65-82% | 65-82% | 53-59% | 100% / 96% |
| SciFact, all | 58-73% | 38-54% | 39-46% | 40-48% | 17% / 16% |
| PubMedQA, all | 19-41% | 18-33% | 19-29% | 13-29% | 2% / 2% |
| LitQA2, all (gold never in corpus) | 7-29% | 1-4% | 2-7% | 3-10% | 0% / 0% |

When the gold paper is in the corpus, retrieval helps (SciFact subset). When it is not, irrelevant papers lower
accuracy (SciFact, PubMedQA), and on LitQA2 the retrieval arms choose "Insufficient information" ~90% of the time
(A0: 60%); their precision on answered items is higher (0.40-0.46 vs 0.31).

## WS4: training-signal study, post-training first

**Question.** Are the KUs a good **training** signal that pushes models, more cheaply than a new pretraining study
(replaces the weak 1.7B pretraining claim D5)? Same recipe and budget for every condition; only the data differ.

**Data conditions** (all from the **same trained papers**; documents token-matched, QA count-matched per paper):

| condition | what the model is trained on |
|---|---|
| `summary` | the cleaned KU (KU-Fixes + WS1 normalizer) rendered as a plain-prose summary |
| `ku` | the same KU as a structured record (JSON of the 18 fields, claims as a list) |
| `full` | the papers' full text |
| `diverse` | paraphrase-diverse KU: each repeat in another surface form (prose, JSON, YAML, HTML, bullet Q&A), after EntiGraph (Yang et al., 2024) and Allen-Zhu & Li (2023) knowledge-augmentation findings |
| `generic` | size-matched generic science data (docs: FineWeb-Edu int_score >= 4; QA: WebInstructSub) |
| `none` | "no extra data": replay only, same steps; the untrained base model is evaluated as well |

**Formats / methods** (`ws4.train`, one LoRA recipe: r 64, alpha 128, all linear layers, lr 1e-4 cosine, 3% warmup,
2,048-token packed sequences, 32k tokens per step, 1 epoch over the token-matched data):

| method | training | data |
|---|---|---|
| `sft-doc` | continued training (causal LM) with replay (25% FineWeb-Edu tokens) | `data` stage corpora |
| `sft-qa` | instruction pairs, loss on answers only, with Tulu-3 replay (25%) | `qa-data` pairs (same generator, same count per paper) |
| `grpo` | GRPO (Shao et al., 2024), Dr.GRPO loss, no KL, 8 generations, reward = answer match (normalised exact / numeric, or a short answer containing the gold) | the condition's verifiable QA pairs, each with the representation it was generated from in the prompt (full text: a 6k-character window around the evidence) |
| `tools` | GRPO with the Alexandria skill as tools (`search`, `get_paper`; Search-R1 style, Jin et al., 2025; TRL native tool calling) | generated KU questions + SciQ train + PubMedQA artificial, equal share per source; **never HLE / GPQA** |

| stage | GPU | env | what |
|---|---|---|---|
| `papers` | | main | trained + held-out papers from one eligible pool (WS2 rules), never WS2 / WS3 eval papers; cleaned KU and full text stored |
| `probe` | yes | llm | closed-book knowledge questions on trained AND held-out papers (WS2 generator and all its checks) |
| `data` | | main | token-matched document corpora per condition + replay (`data_stats.csv`, `budget.json`) |
| `qa-data` | yes | llm | QA pairs from each representation (evidence must be in that representation; equal count per paper), generic control, replay chat |
| `decontam` | | main | drops training examples sharing a 13-gram with any eval item (8-gram for probe questions); `removed.csv` |
| `train-sft` / `train-grpo` / `train-tools` | yes | train | one array task per slice of runs (`--array N`); `runs/<run>/adapter`, `train.json` (loss / reward curve) |
| `eval` | yes | train | lm-evaluation-harness (vLLM + LoRA, chat template): science set (GPQA-Diamond, MMLU-Pro STEM, SciQ, PubMedQA question-only, HLE text) and forgetting set (MMLU, GSM8K, IFEval); knowledge-probe answers |
| `grade` | yes | llm | SimpleQA judge over probe answers + EM / F1 |
| `results` | | main | per model x method x condition: score (mean over seeds, t-interval; 1 seed: 1.96 x stderr), Δ vs base and vs `none`; knowledge-injection effect = (trained − held-out) gain over base with a paper bootstrap CI; forgetting; headline figure |
| `report` | | main | WS4 PDF |

**Evaluation.** Pushing models (main claim): closed-book GPQA-Diamond, MMLU-Pro STEM, SciQ, PubMedQA, HLE text, none
tied to the trained papers. Knowledge injection: closed-book probe on trained papers, held-out papers as the control
(they should not improve). Forgetting: MMLU, GSM8K, IFEval. >= 3 seeds on the main conditions (`ws4.train.seeds`).
GPQA and HLE are gated (HF_TOKEN); HLE is not in lm-eval and is listed as not run until an adapter is added.
**Caution from the literature:** plain SFT injects new facts poorly and can raise hallucination (Ovadia et al., 2023;
Gekhman et al., 2024); hence the held-out control, the paraphrase-diverse condition and the abstention-aware judge.

```bash
alexandria fetch-ws4                                            # login node, once
bash slurm/ws4_debug.sh                                         # tiny real-data test: 0.5B model, 36 papers, 30 steps
PROFILE=full bash slurm/ws4_debug.sh                            # full: qwen3-4b + olmo3-7b, 3 seeds, 3g.40gb slices
```

**Design decisions forced by the debug run (2026-10-23, Qwen2.5-0.5B, 24 + 12 papers, 30 steps):**
* Closed-book GRPO on questions about unseen papers gave reward 0 in every group (no learning signal); with the
  source representation in the prompt, rewards start at 0.45-0.56 (KU / summary) and ~0.03 (full-text windows).
* Tool RL with the public splits in their natural proportions (31.8k prompts, 134 paper questions) learned to stop
  calling tools (call rate 0.75 -> 0); sources are now sampled in equal shares. Tool use still decays at 0.5B / 30
  steps: the real test needs the 4-7B models and longer runs.
* Decontamination against the probe removes more `full`-text pairs (40) than KU pairs (7-14), because probe questions
  are written from the full text; the QA files are re-equalised per paper afterwards (134 pairs per condition).
* lm-eval gets the local snapshot directory (offline nodes); MMLU-Pro is scored with `exact_match,custom-extract`,
  GSM8K with flexible extraction (strict-match is 0 for chat models), IFEval prompt-level strict.
* Closed-book PubMedQA is below chance for the 0.5B base model (8%: it answers "maybe"); expected without the abstract.
* The tool-RL adapter is evaluated closed book like every other run; its real test is WS3 with tools (to add).

**Option B (frontloading into pretraining)** is not implemented (`ws4.pretraining.enabled: false`): it needs the
open-sci-ref 1.7B setup. If run: >= 3 seeds; small-scale metrics (MMLU / ARC cloze, SciQ, bits-per-byte on held-out
papers); controls with the same papers as raw text vs KUs, plain-prose LLM summaries, FineWeb-Edu, token-matched;
decontaminated KU corpus; state from-scratch vs continued; explain BoolQ below chance; no TinyLlama / Llama-3.2
comparisons.

## WS5: teacher distillation

**Question.** If a 9-12B student (Qwen3.5-9B, Gemma-4-12B; official BF16 weights) distilled from 1-2k teacher KUs
gets close to the teacher at a fraction of the cost, regenerating or KU-extracting large subsets becomes feasible
(D2) and a "v3" of the dataset becomes realistic. A KU here is the release's 18-field structured summary, as JSON.

| stage | GPU | env | what |
|---|---|---|---|
| `papers` | | main | open-access papers only (permissive sources such as arXiv, or OpenAlex OA) so teacher outputs can be released; stratified by discipline; split **by paper** into train / dev / test (asserted disjoint); `holdout_domains` disciplines appear in test only |
| `teacher` | yes (40 GB) | llm | teacher KUs for every paper. Default `provider: vllm`: **Qwen3.5-27B-FP8** (open weights, Apache-2.0, so its outputs may train other models and be released; no API key), the largest model that fits one 40 GB MIG slice: official FP8 (31 GB), hybrid linear / full attention so a whole paper (32k tokens) fits; thinking off, the 18-field JSON schema as structured output. Optional `provider: anthropic`: Claude Opus 5.5 via the Message Batches API on a login node (key needed; refuses more than `compliance.max_unreviewed_requests` requests until `compliance.terms_checked: true`) |
| `validate` | | main | gates before training: WS1 schema validation (`load_summary`, required fields, no extra keys), verbatim gate (fail at a >= 12-word span copied from the paper), human spot-check packet (`spot_check.csv` / `.md`, 50 random train KUs) |
| `train` | yes | train | LoRA vs full SFT per student at learning-curve sizes 250 / 500 / 1k / 2k (nested subsets); prompt = summariser prompt + paper, completion = teacher KU, loss on the completion |
| `generate` | yes | train | every system writes every test paper's KU with the same prompt and schema (vLLM structured output), timed: students zero-shot, distilled students, released v2.0 KU, the 27B-class extractor (`gemma4-31b`, when it fits), teacher (ceiling) |
| `mcq` | yes | llm | coverage MCQs from the test papers' full text (WS2 protocol; evidence-checked) |
| `judge` | yes | llm | faithfulness: claims + key-result sentences of each KU vs the paper, judge from a third family (Granite-3.3-8B): SUPPORTED / NOT_SUPPORTED / CONTRADICTED |
| `answer` | yes | llm | coverage: Phi-4-mini (another family) answers the MCQs from each KU only; `closed_book` is the floor |
| `results` | | main | per system: schema validity, faithfulness, coverage (95% CI resampling papers; held-out disciplines separately), verbatim overlap (longest copied span, copied 8-gram share: must not rise with distillation), tokens per paper, papers per GPU-hour; learning curves vs teacher (dashed) and released v2.0 (dotted) |

```bash
bash slurm/ws5_debug.sh                                  # debug: teacher Qwen3.5-27B, Qwen2.5-0.5B student
PROFILE=full TRAIN_GRES=gpu:3g.40gb:1 bash slurm/ws5_debug.sh   # full: 2k / 200 / 1.5k papers, Qwen3.5-9B + Gemma-4-12B
```

**Compliance:** the default teacher is Apache-2.0, so training on and releasing its outputs (CC-BY) is allowed. Only the
optional Claude teacher needs the terms check (`ws5.compliance.terms_checked: true`). Larger open teachers
(Qwen3.5-122B-A10B, gpt-oss-120b, Mistral-Small-4-119B) need more than one 40 GB slice, which our partitions do not offer.
**GPU queue:** the teacher needs a 3g.40gb slice; these are scarce on Grete (2026-10-24: 16 jobs queued for ~8
slices), so plan for queue time. The official Int4 build is no smaller (30 GB) and does not fit a 20 GB slice.
**Downstream check (next):** plug the student KUs into the WS2 arms (`ku_raw` / `ku_clean` from a different source)
and the WS3 index; the `mcq` + `answer` stages already apply the WS2 protocol on the WS5 test papers.
**Open items:** the full-run students are multimodal checkpoints (`Qwen3_5ForConditionalGeneration`,
`Gemma4Unified...`) used text-only: their training path needs a check before the full run; the 27B extractor
(`gemma4-31b`) needs a 40 GB+ slice.

## WS6: embeddings as input

**Question (P2, optional).** The extreme-compression point: can a paper's KU embedding, projected into k
"knowledge tokens" for a frozen LLM, replace its text? xRAG-style (Cheng et al., 2024); cf. ICAE (Ge et al., 2024)
and gist tokens (Mu et al., 2023).

| stage | GPU | env | what |
|---|---|---|---|
| `papers` | | main | projector train / dev papers, split by paper; WS2 eval and WS3 gold papers excluded (the test side) |
| `qa` | yes | llm | QA pairs from each paper's KU prose (WS4 generator and checks) |
| `embed` | yes | main | gte-modernbert (768-d, CLS, fp32, L2-normalised) for projector papers, WS2 papers, the WS3 corpus and queries; re-embedded because the released FAISS indices store lossy PQ48 codes |
| `train-projector` | yes | train | per k: 2-layer MLP 768 -> k x d_model, LLM frozen; phase 1 reconstructs the executive summary, phase 2 QA; dev loss and dev QA match |
| `eval` | yes | train | WS2 questions: closed book, k knowledge tokens, KU as text (WS2 answer prompt); WS3 arm EMB: top-3 papers by cosine -> 3 x k tokens, gold-retrieved rate |
| `grade` / `results` / `report` | | | SimpleQA judge + EM / F1; accuracy with paper-clustered bootstrap CIs vs tokens per paper (log axis); paired Δ vs closed book and KU text |

```bash
bash slurm/ws6_debug.sh                                            # debug: 0.5B LLM, 48 + 12 papers, k in {1, 4}
PROFILE=full TRAIN_GRES=gpu:3g.40gb:1 KS=3 bash slurm/ws6_debug.sh  # Qwen2.5-7B, k in {1, 4, 16}
```

**Debug result (2026-10-23):** closed book 2.4%, 1 token 4.3%, 4 tokens 5.0%, KU text (~2.7k tokens) 42.1%; the
projector learned the answer format, not the content (dev QA match 2%). The real test needs the 7B LLM and the full
training set. The WS3 EMB arm retrieves with gte-modernbert only (gold paper found 81%), not the WS3 hybrid index.

## X1: retrieval evaluation of the indices

**Question.** How well do the release's search indices find the gold paper, and why do the WS3 agents miss it? The
release's FAISS indices store lossy PQ48 codes and are not local, so X1 rebuilds each index type the release offers on
a benchmark corpus (linked gold papers + random release distractors) with the release's settings.

| stage | GPU | what |
|---|---|---|
| `benchmarks` | | LitSearch, CiteME (gated: needs HF_TOKEN), SciFact; gold papers linked to release rows by DOI / exact title (`ws3.gold`) |
| `corpus` | | gold papers + random release distractors; units per representation: summary fields, whole summary, claims, 500-word full-text chunks |
| `index` | yes | per representation: bm25s, exact gte-modernbert vectors (flat), IVF-PQ48 (release settings) |
| `search` | yes | every query x representation x retriever (BM25 / dense / dense PQ48 / RRF k=60): paper rankings, PQ48 loss vs flat, p50 / p95 latency, QPS |
| `ws3-diagnosis` | yes | WS3 failures: gold-paper rank for the question vs the agents' own queries and k, on WS3's index |
| `results` / `report` | | Recall@1/5/10/20/100, MRR, nDCG@10 with bootstrap CIs; PQ48 loss; latency; WS3 diagnosis |

Run: `bash slurm/x1_debug.sh` (debug: <= 200 queries per benchmark, ~4k-paper corpus).
**Debug (2026-10-24; LitSearch 200 queries, SciFact 26 linked, WS3 QA bench 100):** Recall@10 is highest for dense
retrieval on chunks / fields (LitSearch 0.95-0.98) and RRF is never far behind; claims are the weakest representation.
PQ48 costs up to 24 points of Recall@10 vs the exact index (WS3 QA bench, chunks) at ~10x the speed. WS3 diagnosis: the
question itself finds the gold paper in the top 3 for 92% of items, but the agents search in only 65-82% of episodes
(1st query: gold found 85-93%), which explains agents ~67% vs plain RAG 98%.

## X2: post-cutoff benchmark

**Question.** The cleanest test of WS3-H3: does the skill help when the model cannot know? Questions come from arXiv
papers submitted on or after 2026-06-01, after the training cutoff / release of every evaluated model (table with
sources in `configs/x2.yaml`). Scientific-Summaries' arXiv coverage ends in April 2026, so the new papers also
demonstrate an incremental update.

| stage | GPU | what |
|---|---|---|
| `papers` | login | arXiv API (3 s / request), monthly windows, round-robin over 10 primary categories; full text from arxiv.org/html (bibliography dropped); per-paper licence kept |
| `summarize` | yes | KUs with the WS5 pipeline (prompt, 18-field schema, WS1 validation, verbatim gate); full: Qwen3.5-27B (40 GB), debug: Qwen3-8B |
| `questions` | yes | gpt-oss-20b (a family used nowhere else) from the full text with the WS2 rules, evidence check and critic |
| `verify` | | human-verification packet (`verify.csv` / `.md`); a filled `verify_filled.csv` adds the rates to the dashboard |
| `index` | yes | WS3 index + the new papers' KU units and full-text chunks: BM25 rebuilt, dense vectors for new units only (incremental update) |
| `eval` / `grade` / `results` | yes | WS3 harness arms A0 / R / F / S, 4 answer models; SimpleQA judge; paper-clustered CIs, Δ vs A0 (Holm), gold-paper retrieved |

Run: `alexandria x2 papers --profile debug` on a login node, then `bash slurm/x2_debug.sh`; full:
`PROFILE=full SUM_GRES=gpu:3g.40gb:1 bash slurm/x2_debug.sh` (~600 papers, 500-1,000 questions).
**Debug (2026-10-24, 24 questions, 22 papers):** model alone 4-13%; plain RAG over the updated index 67-83% (+54 to +75
points, p < 0.001); KU agent 13-38%, full-text agent 17-33%. Only 23% of generated questions pass all checks.

## X3: comparison with existing systems

**Question.** Alexandria vs PaperQA2 and OpenScholar on shared benchmarks. Every component is open-source and runs
locally (no OpenAI or other external API).

| stage | GPU | what |
|---|---|---|
| `items` | | LitQA2 and the WS3 QA bench (same item ids as WS3), ScholarQABench single-paper SciFact and PubMedQA (with gold contexts) |
| `alexandria` | yes | WS3 harness arms A0 / R / S over the WS3 index, answer model qwen2.5-7b |
| `pqa-corpus` / `paperqa` | yes | paper-qa 2026.8.12 (PaperQA2 pipeline) in the `Alex-pqa` venv over the WS3 corpus' full texts; the same qwen2.5-7b served by vLLM inside the job; all-MiniLM-L6-v2 + sparse embeddings; tiktoken BPE from a local cache; every other outbound connection blocked |
| `openscholar` | yes | OpenScholar/Llama-3.1_OpenScholar-8B with its answer prompt; its 45M-paper datastore (0.74 TB) is not used, so: "OpenScholar-8B LM + Alexandria retrieval", plus gold-context (upper bound) and closed-book |
| `grade` / `results` | | exact match for labels / letters (official for single-paper tasks and LitQA2), SimpleQA judge for short answers; CIs; LitQA2 precision on answered items |

Run: `bash slurm/x3_debug.sh` (50 items per benchmark); full: `PROFILE=full bash slurm/x3_debug.sh`.
**Debug (2026-10-24):** QA bench: OpenScholar-8B + Alexandria retrieval 70%, Alexandria R 54%, PaperQA2 42%. On
ScholarQABench the gold papers are rarely in the corpus, so retrieval lowers accuracy and PaperQA2 mostly abstains;
the gold-context upper bound is 78-80%.

**Accuracy vs cost (Pareto).** Every X3 system stage logs its cost per question:

| System | Logged per question |
|---|---|
| Alexandria | retrieved, prompt and generated tokens; GPU-seconds. For the R arm, retrieved tokens are estimated as its prompt minus the closed-book prompt. |
| OpenScholar-8B | reference, prompt and generated tokens; batch GPU-seconds |
| PaperQA2 | paper-qa token counts over all LLM calls (answer and evidence summaries); evidence tokens; query-phase GPU-seconds |

Wall-clock time is the stage's total time divided by the number of questions.

`x3 results` writes `pareto.csv` and `figures/x3_pareto.png`. The figure plots accuracy (the macro mean over the
benchmarks that every compared system ran) against prompt + generated tokens per question, on a log axis. A system
is marked Pareto-optimal when no other system is both cheaper and more accurate.

To rerun only the stages whose cost fields changed: `SKIP="items pqa-corpus alexandria" bash slurm/x3_debug.sh`.

## X5: knowledge conflict

**Question.** Does the answer come from the retrieved KU or from the model's prior?

**Items.** QA items whose gold paper is in a retrieval index:
* WS2 questions, searched in the WS3 index;
* X2 post-cutoff questions, searched in the X2 index.

**Counterfactual edit.** The fact that carries the answer in the gold KU is edited:
* Numbers are scaled by a rule.
* For other answers, qwen3-8b proposes a plausible alternative and rewrites only the sentences that state the answer.
* An edit is kept only if the new answer is stated, the old one is gone and something changed.

**Conditions.** Each answer model (qwen2.5-7b, olmo3-7b) answers with the WS2 prompt under five conditions:
* closed book;
* original KU;
* counterfactual KU;
* R-arm retrieval with the original gold units;
* R-arm retrieval with the edited gold units (in a copy of the results; the index is never modified).

The SimpleQA judge grades every answer twice: against the original answer and against the counterfactual answer.

**Stages.** items → counterfactual → answer (array, one task per model) → grade → results → report.

**Run.** `bash slurm/x5_debug.sh`. Use `SKIP="items counterfactual"` to reuse those stages.

**Outputs.**
* `x5/results/rates.csv` and the figure `x5_conflict.png`.
* Splits by prior (does the model know the answer closed book?) and by cutoff period.

**Debug scale.** 150 + 24 items; 72 counterfactuals pass the check. The models follow the edited KU 71–76% of the
time in context and 64–65% through retrieval.

## X6: token and compute efficiency

**Question.** How many tokens, GPU-seconds, dollars, watt-hours and FLOPs does a KU save compared with the full text?
Every saving is reported together with accuracy.

**Run.** `bash slurm/x6_debug.sh` runs inference and rag-budget (arrays over models), train-efficiency, storage, and
results + report.

**Stages.**
* **`inference`** uses the WS2 bench. It measures:
  * prompt and output tokens;
  * time to first token, time per output token and end-to-end latency at batch 1;
  * batched throughput;
  * nvidia-smi power.

  The full text is also cut to N tokens and graded, to give an accuracy-vs-tokens curve.
* **`rag-budget`** measures accuracy against a retrieved-token budget of 500–8,000 tokens. It compares KU units with
  full-text chunks, on WS3 and X2 questions.
* **`train-efficiency`** runs LoRA knowledge injection on KU, summary and full-text documents at 0.25–2× the WS4
  token budget. It reports probe accuracy against training tokens and GPU-seconds.
* **`storage`** measures bytes, gzip bytes and tokens per paper, index size, and query latency.
* **`results`** computes, with bootstrap CIs:
  * accuracy per 1k tokens;
  * GPU-seconds, dollars and watt-hours per 1,000 correct answers;
  * KV-cache MB per request;
  * iso-accuracy (how many tokens the full text needs to match the KU);
  * the figure.

**Cost assumptions** are in `configs/x6.yaml` (`x6.cost`): dollars per A100-hour, the MIG slice share and the TDP.
Energy is an estimate, because MIG slices do not report their own power.

**Needs.** ws2 grade, ws3 index and benchmarks, x2 index and questions, ws4 decontam and probe, x1.

**Debug results.**
* The clean KU uses 2.4× fewer prompt tokens and costs 2.7–3.5× less per correct answer.
* It is 16–18 points less accurate than the full text.
* At equal length it beats a full-text prefix by about 6 points.
* For RAG, full-text chunks beat KU units at every budget.
* Training shows no learning signal at debug scale.

## S1: safety checks and evaluation

**Question.** Is the dataset safe to release, and does the Alexandria skill make models less safe? Every output is
counts, rates, categories and record ids; no flagged text is stored.

| stage | GPU | what |
|---|---|---|
| `pii` | | PII in summaries (18 fields) and full text (emails, phones, Luhn-checked card numbers, SSNs, IPs, dates of birth, patient ids, addresses, social handles); author contacts expected in full text, any hit in a summary is a finding; spot-check packet |
| `injection` | | prompt-injection-like text (override phrases, role / chat-template tags, tool-call JSON, invisible / bidi / tag characters) |
| `retractions` | | retracted papers (OpenAlex) in the whole local release, the WS2-WS5 sets and the WS3 index |
| `guard` | yes | Qwen3Guard-Gen-8B labels summaries (Safe / Controversial / Unsafe + categories); Granite Guardian 3.3 re-checks for agreement (kappa) |
| `dual-use` | yes | topic-term prefilter (bio / chem / cyber / rad-nuc) + LLM judge of actionable procedural detail; review packet with ids only |
| `injection-live` | yes | indirect prompt injection: a harmless canary instruction in retrieved text, WS3 arms R and S; share of answers that follow it |
| `uplift` | yes | safety challenge: WMDP accuracy closed book vs Alexandria retrieval; Do-Not-Answer harmful / refused responses without vs with retrieved passages (Qwen3Guard response mode; responses not stored) |
| `results` / `report` | | Wilson CIs, paired bootstrap deltas, release checklist (`safety_flags`, retraction flag, injection sanitisation) |

Run: `bash slurm/s1_debug.sh`; full: `PROFILE=full bash slurm/s1_debug.sh`. Open models / data (download on a login
node): Qwen3Guard-Gen-8B, granite-guardian-3.3-8b, cais/wmdp, LibrAI/do-not-answer. HarmBench, SORRY-Bench,
WildJailbreak and Llama-Guard-4 are gated and not used.
**Debug (2026-10-24, 480 records, 2 models):** PII in 0.6% of summaries; 0 Unsafe of 300 summaries; 0 actionable
dual-use records; injected canary followed in at most 2.5% of episodes; 0.009% retracted records, none in project
sets; no significant WMDP uplift from retrieval (-11 to +1 points); harmful Do-Not-Answer responses not raised
(Qwen2.5: 4.0% -> 3.3%).

## C1: regulatory and ethics compliance

**Question.** Does the release meet the EU AI Act, the GDPR, EU copyright law on text and data mining, the EU
Ethics Guidelines for Trustworthy AI (HLEG / ALTAI) and the international AI frameworks (Council of Europe AI
Convention, OECD AI Principles, UNESCO Recommendation, NIST AI RMF)? What is still open before the release?
C1 is an **evidence matrix, not legal advice**. Items marked *legal review* need a lawyer or the DPO, whatever
the evidence says.

**Stages** (CPU, seconds on debug):

* **`evidence`** collects every measurement into one flat map (`evidence.json`).
  * **Read from earlier stages:**
    * S1: PII, guard, dual-use, injection and live injection, WMDP uplift.
    * WS1: quality, licences, describe, provenance, datasheet / Croissant RAI.
    * X6: compute.
  * **Measured on the records:**
    * **Health identifiers (GDPR Art. 9 proxy)** in PubMed KUs. Direct identifiers are record numbers, titled
      names and exact dates of birth or admission. Quasi-identifiers are case reports giving an exact age and sex.
    * **GDPR territorial scope:** the share of papers with an EU/EEA-affiliated author, from OpenAlex.
    * **Coverage:** language and author-country coverage (AI Act Art. 10(2)(f), HLEG Req. 5).
    * **Licences of shipped full text:** whether each record that ships `text_raw` has a permissive licence.
  * **Repository documents:** checks that the DPIA, privacy notice, copyright policy and AI Office
    training-data summary exist in `docs/`.
  * A key that was not measured is left out and scores as `unknown`.
* **`matrix`** scores each requirement in `configs/c1.yaml` (`c1.requirements`, 31 provisions).
  * Status rules: `met` if all `met` conditions hold; else `partial` if all `partial` conditions hold; else `gap`.
  * Writes `matrix.csv`, `matrix.json` and `compliance_table.tex` (the paper appendix).
  * Publishes the headline KPIs, a status-by-framework chart and the list of open items with actions.
* **`report`** builds the PDF.

**Scope notes.**
* AI Act Art. 10 is binding only for high-risk systems; C1 scores it as best practice.
* The GPAI duties in Art. 53 and 55 fall on the model provider that trains on the data. The dataset supplies what
  that provider needs: the copyright policy, the training-data summary and the documentation.

**Run.**
* `bash slurm/c1_debug.sh` (or `alexandria pipeline c1 evidence matrix report --profile debug`).
* `PROFILE=full bash slurm/c1_debug.sh` after S1 and WS1 have run with `--profile full`.
* Rerun C1 whenever S1, WS1 or X6 change.
* To close a documentation gap, add the file under `docs/` and rerun; the matrix picks it up.
* To add a framework, append requirements to `c1.requirements`.

**Debug result (480 records): 15 met, 10 partial, 6 gaps.**
* **Gaps:**
  * No DPIA (GDPR Art. 6/89 lawful basis, Art. 35).
  * No privacy notice (GDPR Art. 14).
  * No copyright policy (DSM Art. 3).
  * TDM reservations not checked (DSM Art. 4(3)).
  * 87% of the shipped full texts have no permissive licence.
* **Partial:**
  * 0.6% of KUs contain PII.
  * Licences found for 21% of records.
  * The generation prompt is not published.
  * 36% of records have at least one US author.
  * 0.9% of records are non-English.
  * The largest WMDP uplift is +1 pt.

## ICML run protocol (debug -> full)

Order (each step only needs the previous ones; everything is restartable):

```bash
# 0. login node (internet): models, benchmarks (HF_TOKEN for HLE / GPQA), tests
alexandria fetch-models && alexandria fetch-benchmarks && python -m pytest -q
# 1. WS1 full: CPU audit over every shard, then the GPU faithfulness stages
alexandria submit ws1 features --profile full --array 16 --set site.slurm.cpu.partition=grete:preemptible
alexandria submit ws1 all --profile full --one-job --after <features job> --set site.slurm.cpu.partition=grete:preemptible
alexandria submit ws1 grounding judge --profile full --one-job --after <cpu job> --set site.slurm.gpu.partition=grete:preemptible
# 2. WS2 + WS3 full chain (needs WS1 features + quality of the full profile)
PROFILE=full GPU_GRES=gpu:3g.40gb:1 LLM_QUANT=null LLM_LEN=null CTX_TOKENS=16000 bash slurm/ws2_ws3_debug.sh
# 3. dashboard + PDFs
alexandria dashboard serve --profile full --rebuild-every 30
```

Always run the same chain on `--profile debug` first (`bash slurm/ws2_ws3_debug.sh`; reuse valid stages with
`SKIP="questions pilot"`) and check the dashboard before spending full-profile GPU hours.

| setting | debug (tested) | full (paper) |
|---|---|---|
| WS1 input | 2 shards x 60 rows per group | every shard (4,353 local: arXiv 4,006 + PubMed 347) |
| WS1b grounding / judge papers | 100 | 100 (user decision 2026-10-21; plan: 400-600) |
| WS2 papers / questions | 100 / 418 | 1,000 / ~4,500 |
| WS3 items | 100 per benchmark | every item (PubMedQA 1,000, LitQA2 199, SciFact dev 188 single-label claims, QA bench all, HLE / GPQA if gated sets fetched) |
| WS3 corpus | WS1 sample + gold papers (~500 papers) | the full release (see open items) |
| vLLM weights / context | fp8 weight-only, 14,336 tokens, 2g.20gb | bf16 (`LLM_QUANT=null`), native context (`LLM_LEN=null`), 3g.40gb |
| WS2 context cap | 12,000 tokens | 16,000 tokens |
| WS3 sampling | T = 0, seed 0 | T = 0.6, seeds [0, 1, 2] (`--set ws3.agent.temperature=0.6 --set 'ws3.agent.seeds=[0,1,2]'`) |
| bootstrap | 2,000 cluster resamples | 2,000 (10,000 for the final tables) |

**Measured debug cost** (per job, MIG 2g.20gb unless noted): WS2 questions 20-24 min, answer 6-33 min per model,
grade 3-5 min; WS3 index 3 min, pilot 3-4 min per model, agent 0.5-2.5 h per model (400 items x 7 arms), grade 3 min;
WS1 grounding 8-30 min, judge 5-27 min. Full-profile WS3 is ~50x the debug item count per seed.

**Reproducibility.** Dataset revision `232547b7`, model revisions pinned in `configs/models.yaml`, every
manifest records the config fingerprint, code version, host, SLURM job and vLLM overrides; questions, KUs, contexts,
items, retrieval caches and graded answers are stored as parquet / sqlite next to the results, so every number in the
paper can be traced to a file and replayed.

**What goes in the paper.** WS1: `quality/flags_table.tex`, `describe` figure, datasheet. WS2: `results/accuracy_table.tex`,
accuracy-vs-tokens figure, cleaning-effect table. WS3: `results/main_table.tex`, headline Δ-vs-size figure, gold-in-corpus
table, pilot reliability. Each WS PDF collects all of them.

## Threats to validity and open items

* **Full-release WS3 index** (required before the full run): building BM25 + bge-m3 over 22M KUs (~50 units / paper)
  is not feasible per run on our allocation. Options: (a) reuse the dataset's released search indices
  (`search/`: FAISS gte-modernbert + BM25 over the KU fields, claims and chunks; partly on local storage), (b) index a
  fixed large subsample (`ws3.index.max_papers`) that always contains every gold paper. The debug corpus (~500 papers)
  makes retrieval much easier than at full scale; debug WS3 numbers are not paper numbers.
* **Agent array granularity** (required before the full run): `agent` splits work by model, so one model's full run
  (~50x debug) exceeds the 12 h job limit; it needs item sharding within a model.
* **Gold coverage** is limited by the local copy (arXiv + PubMed): LitQA2 cannot be linked at all, PubMedQA barely.
  Downloading the web-papers and bethgelab subsets (metadata columns suffice for linking) would fix this.
* **HLE / GPQA** need `HF_TOKEN` with accepted terms.
* **Same-family judge**: Qwen3-8B writes the QA bench and judges it; a second judge (different family) on a
  subsample, plus EM / F1, is the planned check.
* **Pilot gate**: Phi-4-mini (32.5%) and OLMo-3 (91%) fail the 95% unconstrained tool-call gate; report their agent
  arms as "constrained decoding" results or drop them from the agent comparison.
* **Web arms** (`W`, `W-sch`, `S+W`) cannot run on offline compute nodes.
* **Agents vs plain RAG** (debug finding): on the QA bench R finds the gold paper 98% of the time, the agentic arms
  ~67%; worth a prompt / budget ablation before the full run.
* **Gold absent -> retrieval hurts**: report all-item and [gold in corpus] numbers side by side, and the abstention
  rate (LitQA2) next to accuracy; at full scale the corpus contains far more papers, so linkage coverage decides how
  much of each benchmark can show the skill's effect.

## Datasets: run on any data

Everything dataset-specific lives in `configs/datasets/<name>.yaml`: parquet location, source groups, column mapping
to the canonical schema (`src/alexandria/resources/canonical_schema.json`), link rules (ids -> arXiv / PMC / DOI /
OpenAlex pages), optional extras (KU-Fixes, published numbers for `reconcile`, documented length targets). Copy
`configs/datasets/_template.yaml`, then `alexandria sample --dataset my-data` and run any stage with
`--dataset my-data`. Missing columns become typed nulls; stages that need a missing extra are `skipped` with the
reason. `tests/test_any_dataset.py` runs the CPU pipeline on a synthetic dataset with other column names.

**Adding a workstream:** create `src/alexandria/wsN/` with `@stage("wsN", ...)` modules and a `report` via
`register_report`, add `configs/wsN.yaml`, add `"wsN"` to `WORKSTREAMS` in `cli.py`; `alexandria wsN <stage>`,
`submit`, `pipeline`, the dashboard and the PDF pick it up.

**Opt-out / takedown.** `site.optout_file` (paper ids or DOIs, one per line) is applied when any stage reads the
release (`alexandria.data.iter_batches`): listed rows are dropped from every derived file and index on the next build.
The request process (acknowledge, verify, remove, publish) is in `docs/TAKEDOWN.md`.

## Data notes

* Column names are canonical in every released shard; key variants exist only inside the `claims` JSON
  (`caveats`, `contradating_evidence`, ...). Types differ between shards; `data.iter_batches` casts to KU-Fixes'
  `unified_schema.json` (the local `normalized/` copy stores arxiv OpenAlex columns as strings).
* Some arxiv/old sources are unreadable PDF extractions (font mojibake); their fluent KUs are flagged `source_garbled`.
* Every field with a documented length target is delivered at ~0.4-0.9x of it (executive summary: median 208 words,
  max 340, vs 400-500); the comparison is one dataset-level table (`quality/length_vs_documented.csv`).
* `text_length` is clipped at 100,000 characters; PubMed `text_raw` is mostly empty (`text_sanitized` is used).
* PubMed rows carry no PMID and mostly no DOI (`paper_id` is an extraction path): link PubMed papers by title.
* Local storage holds the arXiv and PubMed subsets only (bethgelab and web-papers are not downloaded).
