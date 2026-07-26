"""Model interface + shared config.

Every model — baseline, classical, deep, transformer-head — implements this interface so
the training harness and inference service treat them uniformly. ``predict_proba`` returns
a full probability distribution over ``classes`` (ordered), which the API surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from emotionsense.common.schemas import FeatureSpec


@dataclass(slots=True)
class ModelConfig:
    """Declarative model config (mirrors configs/models/*.yaml)."""

    family: str  # 'svm' | 'random_forest' | 'xgboost' | 'bilstm' | 'distilhubert' | ...
    name: str
    feature: FeatureSpec
    hyperparams: dict[str, Any] = field(default_factory=dict)
    class_weight_balanced: bool = True  # R-ML3: handle imbalance by default


@runtime_checkable
class EmotionModel(Protocol):
    """Uniform model contract used by training + serving."""

    classes: list[str]
    feature_spec: FeatureSpec

    def fit(self, x: np.ndarray, y: list[str]) -> None: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_classes) probabilities aligned to ``classes``."""
        ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> EmotionModel: ...


def probs_to_dict(probs: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Map a single probability row to a {label: prob} dict."""
    return {c: float(p) for c, p in zip(classes, probs, strict=True)}
