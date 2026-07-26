"""Artifact (de)serialization with a checksum-verified load path (design review R-SEC1).

Security posture: the artifact store is a TRUST BOUNDARY. Deep models are saved as
**safetensors** (never pickle), and every artifact is loaded only after its SHA-256
matches the checksum recorded in the registry. A tampered artifact fails the checksum and
is refused rather than executed. Classical (sklearn) models use joblib but are still
checksum-gated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from emotionsense.common.errors import ModelUnavailableError
from emotionsense.ml.models.base import EmotionModel, ModelConfig
from emotionsense.ml.models.registry import build_model


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model_checksummed(model: EmotionModel, path: Path) -> str:
    """Save a model and return the SHA-256 of its primary artifact."""
    path = Path(path)
    model.save(path)
    return sha256_file(path)


def load_model_verified(family: str, path: Path, expected_sha256: str) -> EmotionModel:
    """Load a model only if the artifact checksum matches (else refuse — R-SEC1)."""
    path = Path(path)
    if not path.exists():
        raise ModelUnavailableError(f"Artifact missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ModelUnavailableError(
            "Artifact checksum mismatch — refusing to load (possible tampering)",
            details={"expected": expected_sha256, "actual": actual},
        )
    from emotionsense.ml.models.baselines import MajorityBaseline, RandomBaseline
    from emotionsense.ml.models.classical import ClassicalModel

    if family == "majority":
        return MajorityBaseline.load(path)
    if family == "random":
        return RandomBaseline.load(path)
    if family in {
        "svm",
        "logreg",
        "random_forest",
        "xgboost",
        "distilhubert",
        "hubert",
        "wav2vec2",
        "emotion2vec",
    }:
        return ClassicalModel.load(path)
    if family in {"cnn", "lstm", "bilstm"}:
        from emotionsense.ml.models.deep import DeepSequenceModel

        return DeepSequenceModel.load(path)
    # Fallback via factory (should not normally hit).
    _ = build_model(ModelConfig(family=family, name=family, feature=None))  # type: ignore[arg-type]
    raise ModelUnavailableError(f"No loader for family: {family}")
