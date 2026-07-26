"""Load declarative YAML configs (datasets, features, models, experiments).

This is the "config over code" surface: a new dataset/model/experiment is a YAML file,
not a code change. Loaded configs are plain dicts here; typed wrappers live next to the
components that consume them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must be a mapping, got {type(data).__name__}")
    return data


def load_experiment(path: str | Path, configs_root: str | Path = "configs") -> dict[str, Any]:
    """Load an experiment config, resolving referenced dataset/feature/model configs.

    An experiment YAML references components by name; this composes them into a single
    resolved dict so a run is fully self-describing and reproducible (NFR-3).
    """
    root = Path(configs_root)
    exp = load_yaml(path)
    resolved: dict[str, Any] = {"name": exp["name"], "description": exp.get("description", "")}

    resolved["datasets"] = [load_yaml(root / "datasets" / f"{d}.yaml") for d in exp["datasets"]]
    resolved["models"] = [load_yaml(root / "models" / f"{m}.yaml") for m in exp["models"]]
    if "cross_corpus" in exp:
        cc = exp["cross_corpus"]
        resolved["cross_corpus"] = {
            "train": cc["train"],
            "test": cc["test"],
        }
    resolved["eval"] = exp.get("eval", {"n_folds": 5, "seed": 42})
    return resolved
