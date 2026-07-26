"""Train one model on a corpus and register it for serving.

Trains on all harmonized data (speaker-independent evaluation is the benchmark's job; for
the *served* model we fit on everything available), saves a checksummed artifact, and
registers it as the production default so the API/UI can serve it immediately.

Usage:
    python scripts/train_and_register.py --model svm --dataset synthetic --promote
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotionsense.common.logging import configure_logging, get_logger
from emotionsense.common.yaml_config import load_yaml
from emotionsense.datasets.build import build_pairs
from emotionsense.ml.models.registry import build_model
from emotionsense.registry.artifact import save_model_checksummed
from emotionsense.registry.local_registry import LocalRegistry, ModelRecord
from emotionsense.training.featureset import build_matrix
from emotionsense.training.runner import model_cfg_from_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svm")
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--registry", default="experiments/artifacts/registry")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    configure_logging()
    log = get_logger("train")

    model_dict = load_yaml(Path(args.configs) / "models" / f"{args.model}.yaml")
    dataset_dict = load_yaml(Path(args.configs) / "datasets" / f"{args.dataset}.yaml")
    cfg = model_cfg_from_dict(model_dict)

    pairs = build_pairs(dataset_dict)
    x, y = build_matrix(pairs, cfg.feature, cache_dir=Path("data/features"))
    model = build_model(cfg)
    model.fit(x, y)
    log.info("train.fit", model=cfg.name, samples=len(y), classes=len(model.classes))

    registry = LocalRegistry(Path(args.registry))
    name = f"{cfg.name}-{args.dataset}"
    artifact_path = Path(args.registry) / "artifacts" / f"{name}-{args.version}.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    sha = save_model_checksummed(model, artifact_path)

    record = ModelRecord(
        name=name,
        version=args.version,
        family=cfg.family,
        artifact_path=str(artifact_path),
        sha256=sha,
        feature_spec=cfg.feature.model_dump(),
        label_classes=model.classes,
        status="staging",
    )
    registry.register(record)
    if args.promote:
        registry.promote(record.ref, make_default=True)
    log.info("train.registered", ref=record.ref, sha256=sha[:12], promoted=args.promote)
    print(f"Registered {record.ref} (default={args.promote})")


if __name__ == "__main__":
    main()
