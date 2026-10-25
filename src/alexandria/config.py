"""Configuration: YAML files + command-line overrides, validated with pydantic.

    cfg = load_config("ws1", profile="debug", overrides=["ws1.checks.min_title_similarity=70"])
    cfg.site.out_root, cfg.raw["ws1"]["checks"], cfg.stage_dir("ws1", "features")

`configs/site.yaml` (machine paths, SLURM resources) is merged with `configs/<ws>.yaml` and then with the dataset
file `configs/datasets/<dataset>.yaml` (where the data is, how its columns map to the canonical schema, optional
extras such as KU-Fixes, link rules). The dataset is chosen with `dataset=` (CLI `--dataset`), else
$ALEXANDRIA_DATASET, else site.default_dataset. Strings may reference other values as ${dotted.key}; keys inside
site.yaml may also use ${key} for their siblings (e.g. ${store}).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("ALEXANDRIA_CONFIG_DIR", REPO_ROOT / "configs"))
PROFILES = ("debug", "full")
_REF = re.compile(r"\$\{([^}]+)\}")


class SlurmResources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partition: str
    gres: str | None = None
    cpus: int = 8
    mem: str = "32G"
    time: str = "01:00:00"
    debug_time: str = "01:00:00"     # time limit under the debug profile


class SlurmConfig(BaseModel):
    account: str
    cpu: SlurmResources
    gpu: SlurmResources
    debug: SlurmResources


class SiteConfig(BaseModel):
    store: Path
    data_root: Path
    ku_fixes: Path | None = None     # optional extra (Scientific-Summaries only): fault index, duplicates, key stats
    out_root: Path
    assets: Path
    hf_home: Path
    logs: Path
    envs: dict[str, Path]
    slurm: SlurmConfig
    default_dataset: str = "scientific-summaries"
    llm: dict[str, Any] = {}          # global LLM overrides: quantization (e.g. fp8), max_model_len cap
    optout_file: Path | None = None   # opt-out / takedown list (paper ids or DOIs, one per line): rows are dropped on read


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: str
    glob: str


class LinkRule(BaseModel):
    """Turn a record column into a link: `pattern` (regex) must match the value; `url` uses {0} for the whole match
    and {1}, {2}, ... for groups. Example: {name: arXiv, column: paper_id, pattern: '^\\d{4}\\.\\d{4,5}', url: 'https://arxiv.org/abs/{0}'}"""
    model_config = ConfigDict(extra="forbid")
    name: str
    column: str
    pattern: str = ".+"
    url: str


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    title: str = ""
    description: str = ""
    homepage: str | None = None
    out_subdir: str | None = None    # results under <out_root>/<out_subdir>/<profile>/; None = the dataset name
    columns: dict[str, str] = {}     # canonical column -> column name in this dataset (only where they differ)
    links: list[LinkRule] = []


class DataConfig(BaseModel):
    sources: list[Source]
    revision: str | None = None


class Config(BaseModel):
    """Validated top level. Workstream sections stay as plain dicts in `raw`; each stage validates its own part."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ws: str
    profile: str
    site: SiteConfig
    data: DataConfig
    profile_settings: dict[str, Any]
    raw: dict[str, Any]
    dataset: DatasetInfo

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name) or {}

    def run_root(self, profile: str | None = None) -> Path:
        """<out_root>/<dataset out_subdir>/<profile>: everything one dataset + profile produces."""
        sub = self.dataset.name if self.dataset.out_subdir is None else self.dataset.out_subdir
        return self.site.out_root / sub / (profile or self.profile)

    def stage_dir(self, ws: str, stage: str) -> Path:
        return self.run_root() / ws / stage

    def fingerprint(self, *keys: str) -> str:
        """Short hash of the named config sections (what a stage's results depend on)."""
        blob = json.dumps({k: _get(self.raw, k) for k in keys}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _get(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _resolve(node: Any, root: dict, local: dict, depth: int = 0) -> Any:
    if depth > 20:
        raise ValueError("config interpolation is nested too deeply (a cycle?)")
    if isinstance(node, dict):
        return {k: _resolve(v, root, local, depth) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, root, local, depth) for v in node]
    if not isinstance(node, str) or "${" not in node:
        return node

    def lookup(m: re.Match) -> str:
        key = m.group(1)
        val = _get(local, key, None)
        if val is None:
            val = _get(root, key, None)
        if val is None:
            raise KeyError(f"config reference ${{{key}}} is not defined")
        return str(_resolve(val, root, local, depth + 1))

    whole = _REF.fullmatch(node)
    if whole:                                     # keep non-string types for a reference that is the whole value
        key = whole.group(1)
        val = _get(local, key, None)
        val = _get(root, key, None) if val is None else val
        if val is not None and not isinstance(val, str):
            return val
    return _REF.sub(lookup, node)


def _merge(base: dict, over: dict) -> dict:
    """Deep merge: dicts merge key by key, anything else in `over` replaces the value in `base`."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def available_datasets(config_dir: Path | None = None) -> list[str]:
    d = Path(config_dir or CONFIG_DIR) / "datasets"
    return sorted(p.stem for p in d.glob("*.yaml") if not p.stem.startswith("_"))


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must look like key.path=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def load_config(ws: str, profile: str = "debug", overrides: list[str] | None = None,
                config_dir: Path | None = None, dataset: str | None = None) -> Config:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    cdir = Path(config_dir or CONFIG_DIR)
    site = yaml.safe_load((cdir / "site.yaml").read_text())
    ws_file = cdir / f"{ws}.yaml"
    body = yaml.safe_load(ws_file.read_text()) if ws_file.exists() else {}
    raw = {"site": site, **copy.deepcopy(body)}
    if (cdir / "models.yaml").exists():                       # shared model registry (every workstream)
        raw = _merge(raw, yaml.safe_load((cdir / "models.yaml").read_text()) or {})
    name = dataset or os.environ.get("ALEXANDRIA_DATASET") or site.get("default_dataset", "scientific-summaries")
    ds_file = cdir / "datasets" / f"{name}.yaml"
    if not ds_file.exists():
        raise FileNotFoundError(f"unknown dataset {name!r}: no {ds_file} (available: {available_datasets(cdir)})")
    raw = _merge(raw, yaml.safe_load(ds_file.read_text()) or {})
    raw.setdefault("dataset", {})["name"] = name
    for item in overrides or []:
        key, value = parse_override(item)
        _set(raw, key, value)
    raw["site"] = _resolve(raw["site"], raw, raw["site"])
    raw = _resolve(raw, raw, raw)
    return Config(ws=ws, profile=profile, site=SiteConfig(**raw["site"]), data=DataConfig(**raw["data"]),
                  profile_settings=(raw.get("profiles") or {}).get(profile) or {}, raw=raw,
                  dataset=DatasetInfo(**raw["dataset"]))
