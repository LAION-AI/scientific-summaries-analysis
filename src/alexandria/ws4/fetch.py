"""`alexandria fetch-ws4` (login node; compute nodes are offline): everything WS4 reads from the Hub.

  * the training models (debug and full lists) and the generator / judge;
  * ws4_resources: replay (FineWeb-Edu, Tulu-3), generic controls (FineWeb-Edu, WebInstructSub), RL training splits;
  * the eval files used for decontamination (gated GPQA only with HF_TOKEN);
  * the lm-eval task datasets (run with the Alex-train python, which has lm-eval) and NLTK data for IFEval.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..runtime import log


def fetch(cfg) -> dict[str, str]:
    from huggingface_hub import hf_hub_download, snapshot_download

    from .data import EVAL_FILES
    from .evaluate import LM_TASKS
    os.environ["HF_HOME"] = str(cfg.site.hf_home)
    hub, token, status = str(cfg.site.hf_home / "hub"), os.environ.get("HF_TOKEN"), {}
    ws4 = cfg.raw.get("ws4", {})
    names = set(ws4.get("train", {}).get("models", [])) | set(cfg.profile_settings.get("ws4_models", []))
    names |= {ws4.get("qa_data", {}).get("generator", "qwen3-8b"), ws4.get("eval", {}).get("judge", "qwen3-8b")}
    for n in sorted(names):
        m = cfg.raw["models"][n]
        snapshot_download(m["hf"], revision=m.get("revision"), cache_dir=hub,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "tokenizer*", "*.tiktoken", "*.jinja"])
        status[f"model:{n}"] = "ok"
        log(f"fetch-ws4: model {n} ok")
    for name, r in cfg.raw.get("ws4_resources", {}).items():
        for f in r["files"]:
            hf_hub_download(r["repo"], f, repo_type="dataset", cache_dir=hub)
        status[f"resource:{name}"] = "ok"
        log(f"fetch-ws4: resource {name} ok")
    for name, (repo, f, _) in EVAL_FILES.items():
        try:
            hf_hub_download(repo, f, repo_type="dataset", cache_dir=hub, token=token)
            status[f"eval:{name}"] = "ok"
        except Exception as e:                                         # noqa: BLE001  (gated: reported)
            status[f"eval:{name}"] = f"not available ({type(e).__name__}; gated sets need HF_TOKEN)"
        log(f"fetch-ws4: eval file {name}: {status[f'eval:{name}']}")
    tasks = [t for s in list(ws4.get("eval", {}).get("science", [])) + list(ws4.get("eval", {}).get("general", []))
             for t in (LM_TASKS.get(s) or []) if t != "pubmedqa_closedbook"]
    nltk_dir = Path(cfg.site.store) / "nltk_data"
    code = ("import sys, nltk, lm_eval.tasks as T\n"
            f"nltk.download('punkt_tab', download_dir={str(nltk_dir)!r}, quiet=True)\n"
            "tm = T.TaskManager()\n"
            "for t in sys.argv[1:]:\n"
            "    try:\n"
            "        T.get_task_dict([t], tm); print('ok', t, flush=True)\n"
            "    except Exception as e:\n"
            "        print('FAIL', t, type(e).__name__, str(e)[:120], flush=True)\n")
    py = cfg.site.envs.get("train")
    r = subprocess.run([str(py / "bin" / "python"), "-c", code, *tasks], capture_output=True, text=True,
                       env={**os.environ, "HF_HOME": str(cfg.site.hf_home), "HF_TOKEN": token or ""})
    for line in r.stdout.splitlines():
        if line.startswith(("ok", "FAIL")):
            parts = line.split(" ", 2)
            status[f"lm_eval:{parts[1]}"] = "ok" if parts[0] == "ok" else f"failed: {parts[2] if len(parts) > 2 else ''}"
    if r.returncode:
        status["lm_eval"] = f"error: {r.stderr[-300:]}"
    log(f"fetch-ws4: lm-eval datasets: {sum(v == 'ok' for k, v in status.items() if k.startswith('lm_eval:'))}/{len(tasks)} ok")
    return status
