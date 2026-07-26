"""CLI: run the unified benchmark harness and export a leaderboard.

Usage:
    python scripts/run_benchmark.py --experiment configs/experiments/smoke_synthetic.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotionsense.common.logging import configure_logging, get_logger
from emotionsense.common.yaml_config import load_experiment
from emotionsense.training.report import export, to_markdown
from emotionsense.training.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EmotionSense benchmark harness")
    parser.add_argument(
        "--experiment",
        default="configs/experiments/smoke_synthetic.yaml",
        help="Path to an experiment YAML",
    )
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--out-dir", default="experiments/reports")
    args = parser.parse_args()

    configure_logging()
    log = get_logger("benchmark")

    resolved = load_experiment(args.experiment)
    log.info("benchmark.start", experiment=resolved["name"], models=len(resolved["models"]))
    rows = run_experiment(resolved, cache_dir=Path(args.cache_dir))
    md_path, json_path = export(rows, Path(args.out_dir), name=resolved["name"])
    log.info("benchmark.done", rows=len(rows), markdown=str(md_path), json=str(json_path))
    print("\n" + to_markdown(rows, title=f"Leaderboard — {resolved['name']}"))


if __name__ == "__main__":
    main()
