"""WS1 on a dataset that is not Scientific-Summaries: other column names, two groups, no KU-Fixes, no published
numbers. Runs the CPU pipeline end to end on a few synthetic records and checks the outputs and the PDF."""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alexandria.config import CONFIG_DIR, available_datasets, load_config
from alexandria.data import iter_batches, release_shards

LANG_MODEL = Path("/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/Alexandria-ICML-2027/assets/lid.176.bin")

DATASET = """
dataset:
  title: Demo summaries
  description: synthetic records with their own column names
  columns: {paper_id: id, executive_summary: summary, key_results: results, text_sanitized: body, oa_doi: doi}
  links:
    - {name: DOI, column: oa_doi, pattern: '^10\\..+', url: "https://doi.org/{0}"}
site:
  data_root: DATA_ROOT
  ku_fixes: null
data:
  sources:
    - {group: alpha, glob: "alpha/*.parquet"}
    - {group: beta,  glob: "beta/*.parquet"}
ws1:
  length_spec_words: {}
  reconcile: {claims: []}
"""


def _records(n: int, seed: int) -> pa.Table:
    words = "the method improves accuracy on the benchmark while reducing compute and memory".split()
    rows = []
    for i in range(n):
        body = " ".join(words[(i + j) % len(words)] for j in range(400))
        rows.append({"id": f"demo-{seed}-{i}", "summary": " ".join(words * 15) + f" Paper {i}.",
                     "results": "Accuracy improves by 3 points. " * (2 if i == 0 else 20),
                     "body": body, "doi": f"10.1234/demo.{seed}.{i}", "extra": i})
    return pa.Table.from_pylist(rows)


@pytest.fixture()
def demo(tmp_path):
    if not LANG_MODEL.exists():
        pytest.skip("fastText language model not available on this machine")
    data = tmp_path / "data"
    for g, seed in (("alpha", 1), ("beta", 2)):
        (data / g).mkdir(parents=True)
        pq.write_table(_records(40, seed), data / g / "part-0.parquet")
    cdir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, cdir)
    (cdir / "datasets" / "demo.yaml").write_text(DATASET.replace("DATA_ROOT", str(data)))
    return cdir, tmp_path


def test_columns_are_mapped(demo):
    cdir, tmp = demo
    assert "demo" in available_datasets(cdir) and "_template" not in available_datasets(cdir)
    cfg = load_config("ws1", "debug", [f"site.out_root={tmp / 'runs'}"], config_dir=cdir, dataset="demo")
    assert cfg.run_root() == tmp / "runs" / "demo" / "debug"
    shards = release_shards(cfg)
    assert {s.group for s in shards} == {"alpha", "beta"}
    tbl = next(iter_batches(cfg, shards[0], ["paper_id", "executive_summary", "claims", "extra"]))
    assert tbl["paper_id"][0].as_py() == "demo-1-0"                      # id -> paper_id
    assert tbl["executive_summary"][0].as_py().startswith("the method")  # summary -> executive_summary
    assert tbl["claims"].null_count == tbl.num_rows                       # absent canonical column -> nulls
    assert tbl["extra"][0].as_py() == 0                                   # non-canonical column kept as is


def test_ws1_pipeline_on_other_dataset(demo):
    from alexandria.cli import main
    cdir, tmp = demo
    import alexandria.config as config_mod
    old = config_mod.CONFIG_DIR
    config_mod.CONFIG_DIR = cdir
    try:
        common = ["--dataset", "demo", "--set", f"site.out_root={tmp / 'runs'}"]
        assert main(["sample", *common]) == 0
        assert main(["pipeline", "ws1", "features", "keys", "neardup-sign", "neardup", "quality", "describe",
                     "reconcile", "report", "--profile", "debug", *common]) == 0
    finally:
        config_mod.CONFIG_DIR = old
    run = tmp / "runs" / "demo" / "debug" / "ws1"
    status = {p.parent.name: json.loads(p.read_text())["status"] for p in run.glob("*/manifest.json")}
    assert status["reconcile"] == "skipped"                              # no published numbers for this dataset
    assert all(v == "done" for k, v in status.items() if k != "reconcile"), status
    flagged = (run / "quality" / "flagged_papers.csv").read_text()
    assert "https://doi.org/10.1234/demo" in flagged                      # links from the dataset's rule
    panels = json.loads((run / "quality" / "dashboard.json").read_text())["panels"]
    assert not any(p["type"] == "examples" for p in panels)
    pdfs = list((run / "report").glob("*.pdf"))
    assert len(pdfs) == 1 and pdfs[0].name == "WS1_report_demo_debug.pdf" and pdfs[0].stat().st_size > 10_000
