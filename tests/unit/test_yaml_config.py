"""Config loading tests: single YAML + experiment resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotionsense.common.yaml_config import load_experiment, load_yaml


@pytest.mark.unit
def test_load_yaml_valid(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nvalue: 3\n")
    data = load_yaml(p)
    assert data == {"name": "x", "value": 3}


@pytest.mark.unit
def test_load_yaml_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


@pytest.mark.unit
def test_load_yaml_non_mapping_raises(tmp_path: Path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_yaml(p)


@pytest.mark.unit
def test_load_experiment_resolves_references():
    resolved = load_experiment("configs/experiments/baseline_suite.yaml")
    assert resolved["name"] == "baseline-suite"
    assert len(resolved["models"]) == 5
    assert len(resolved["datasets"]) == 1
    assert resolved["cross_corpus"] == {"train": "ravdess", "test": "crema_d"}
    assert resolved["eval"]["n_folds"] == 5
    # every resolved model carries its feature block
    assert all("feature" in m for m in resolved["models"])
