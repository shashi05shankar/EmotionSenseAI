"""Full-harness orchestration: run an experiment config -> leaderboard -> export."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotionsense.common.yaml_config import load_experiment
from emotionsense.training.report import export, to_markdown
from emotionsense.training.runner import run_experiment


@pytest.mark.integration
def test_smoke_experiment_end_to_end(tmp_path: Path, monkeypatch):
    # Isolate any generated synthetic data / feature cache under tmp.
    monkeypatch.chdir(Path.cwd())  # keep configs/ reachable
    resolved = load_experiment("configs/experiments/smoke_synthetic.yaml")
    rows = run_experiment(resolved, cache_dir=tmp_path / "features")
    assert rows, "experiment produced no leaderboard rows"

    names = {r.model for r in rows}
    assert "baseline-majority" in names
    # baselines must sit near chance while a real model beats them
    majority = next(r for r in rows if r.model == "baseline-majority")
    best = max(rows, key=lambda r: r.ua_mean)
    assert majority.ua_mean < best.ua_mean

    md = to_markdown(rows, title="smoke")
    assert "UA" in md and "|" in md
    md_path, json_path = export(rows, tmp_path / "reports", name="smoke")
    assert md_path.exists() and json_path.exists()
