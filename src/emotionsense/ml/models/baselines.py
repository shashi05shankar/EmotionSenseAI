"""Trivial baselines (design review R-ML1).

Without a chance baseline, no accuracy number is interpretable: is 78% good for 7
imbalanced classes? These make every leaderboard row meaningful and are the single
cheapest, highest-signal ML-maturity marker in the project.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from emotionsense.common.schemas import FeatureSpec

_NULL_SPEC = FeatureSpec(extractor="mfcc")  # baselines ignore features


class MajorityBaseline:
    """Always predicts the most frequent training class."""

    def __init__(self, classes: list[str] | None = None) -> None:
        self.classes: list[str] = classes or []
        self.feature_spec = _NULL_SPEC
        self._majority_idx = 0

    def fit(self, x: np.ndarray, y: list[str]) -> None:
        if not self.classes:
            self.classes = sorted(set(y))
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]
        self._majority_idx = self.classes.index(majority)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0] if x.ndim > 1 else len(x)
        probs = np.zeros((n, len(self.classes)), dtype=np.float32)
        probs[:, self._majority_idx] = 1.0
        return probs

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"classes": self.classes, "majority_idx": self._majority_idx})
        )

    @classmethod
    def load(cls, path: Path) -> MajorityBaseline:
        data = json.loads(Path(path).read_text())
        obj = cls(classes=data["classes"])
        obj._majority_idx = data["majority_idx"]
        return obj


class RandomBaseline:
    """Predicts by sampling the training class prior (stratified random)."""

    def __init__(self, classes: list[str] | None = None, seed: int = 42) -> None:
        self.classes = classes or []
        self.feature_spec = _NULL_SPEC
        self._prior: np.ndarray = np.array([])
        self._seed = seed

    def fit(self, x: np.ndarray, y: list[str]) -> None:
        if not self.classes:
            self.classes = sorted(set(y))
        counts = Counter(y)
        self._prior = np.array([counts.get(c, 0) for c in self.classes], dtype=np.float64)
        self._prior = self._prior / self._prior.sum()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0] if x.ndim > 1 else len(x)
        return np.tile(self._prior.astype(np.float32), (n, 1))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"classes": self.classes, "prior": self._prior.tolist()}))

    @classmethod
    def load(cls, path: Path) -> RandomBaseline:
        data = json.loads(Path(path).read_text())
        obj = cls(classes=data["classes"])
        obj._prior = np.array(data["prior"], dtype=np.float64)
        return obj
