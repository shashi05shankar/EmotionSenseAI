"""Build speaker-independent k-fold split manifests for a dataset.

Usage:
    python scripts/build_splits.py --dataset synthetic --folds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotionsense.common.logging import configure_logging, get_logger
from emotionsense.common.yaml_config import load_yaml
from emotionsense.datasets.build import build_pairs
from emotionsense.datasets.manifest import write_fold
from emotionsense.datasets.split import (
    assert_speaker_independent,
    make_speaker_independent_folds,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--configs", default="configs")
    args = ap.parse_args()

    configure_logging()
    log = get_logger("splits")

    dataset_dict = load_yaml(Path(args.configs) / "datasets" / f"{args.dataset}.yaml")
    pairs = build_pairs(dataset_dict)
    folds = make_speaker_independent_folds(pairs, n_folds=args.folds, seed=args.seed)
    for fold in folds:
        assert_speaker_independent(fold)  # guardrail before persisting
        path = write_fold(fold, Path(args.out), args.dataset, args.seed)
        log.info(
            "splits.fold", fold=fold.index, path=str(path), test_speakers=len(fold.speakers("test"))
        )
    print(f"Wrote {len(folds)} speaker-independent folds to {args.out}")


if __name__ == "__main__":
    main()
