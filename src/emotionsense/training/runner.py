"""Benchmark orchestration: resolved experiment config -> leaderboard rows.

Ties the pieces together: build harmonized pairs per corpus, run speaker-independent CV
for every model, and (if configured) a cross-corpus generalization test. Pure orchestration
— all the rigor lives in the harness/evaluate modules.
"""

from __future__ import annotations

from pathlib import Path

from emotionsense.common.schemas import FeatureSpec
from emotionsense.datasets.build import build_pairs
from emotionsense.ml.models.base import ModelConfig
from emotionsense.training.benchmark import (
    LeaderboardRow,
    run_cross_corpus,
    run_cv,
)


def model_cfg_from_dict(d: dict) -> ModelConfig:
    """Build a ModelConfig from a resolved model YAML dict."""
    return ModelConfig(
        family=d["family"],
        name=d["name"],
        feature=FeatureSpec(**d["feature"]),
        hyperparams=d.get("hyperparams", {}) or {},
        class_weight_balanced=d.get("class_weight_balanced", True),
    )


def run_experiment(resolved: dict, cache_dir: Path | None = None) -> list[LeaderboardRow]:
    """Execute a resolved experiment config and return leaderboard rows."""
    n_folds = resolved["eval"].get("n_folds", 5)
    seed = resolved["eval"].get("seed", 42)

    # Primary corpus = first configured dataset for in-corpus CV.
    dataset_cfgs = {d["name"]: d for d in resolved["datasets"]}
    primary_cfg = resolved["datasets"][0]
    primary_pairs = build_pairs(primary_cfg)

    rows: list[LeaderboardRow] = []
    for m in resolved["models"]:
        cfg = model_cfg_from_dict(m)
        rows.append(
            run_cv(
                cfg,
                primary_pairs,
                dataset_name=primary_cfg["name"],
                n_folds=n_folds,
                seed=seed,
                cache_dir=cache_dir,
            )
        )

    # Optional cross-corpus generalization test (R-ML4).
    if "cross_corpus" in resolved:
        cc = resolved["cross_corpus"]
        train_cfg = dataset_cfgs.get(cc["train"]) or _load_extra(cc["train"])
        test_cfg = dataset_cfgs.get(cc["test"]) or _load_extra(cc["test"])
        train_pairs = build_pairs(train_cfg)
        test_pairs = build_pairs(test_cfg)
        for m in resolved["models"]:
            cfg = model_cfg_from_dict(m)
            if cfg.family in {"majority", "random"}:
                continue  # baselines uninformative cross-corpus
            rows.append(
                run_cross_corpus(
                    cfg,
                    train_pairs,
                    test_pairs,
                    train_name=cc["train"],
                    test_name=cc["test"],
                    cache_dir=cache_dir,
                )
            )
    return rows


def _load_extra(dataset_name: str, configs_root: str = "configs") -> dict:
    from emotionsense.common.yaml_config import load_yaml

    return load_yaml(Path(configs_root) / "datasets" / f"{dataset_name}.yaml")
