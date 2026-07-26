"""Classical models: SVM, Random Forest, XGBoost, Logistic Regression.

All wrap a scikit-learn-compatible estimator behind the uniform ``EmotionModel``
contract, with a ``StandardScaler`` in front (essential for SVM). Class imbalance is
handled via ``class_weight='balanced'`` where supported (R-ML3). Persisted with joblib +
a sidecar metadata JSON, and loaded through a checksum-verified path (R-SEC1) in the
registry layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emotionsense.common.schemas import FeatureSpec
from emotionsense.ml.models.base import ModelConfig


def _make_estimator(cfg: ModelConfig):
    hp = dict(cfg.hyperparams)
    cw = "balanced" if cfg.class_weight_balanced else None
    if cfg.family == "svm":
        return SVC(probability=True, class_weight=cw, **hp)
    if cfg.family == "logreg":
        return LogisticRegression(class_weight=cw, max_iter=1000, **hp)
    if cfg.family == "random_forest":
        return RandomForestClassifier(class_weight=cw, **hp)
    if cfg.family == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("xgboost requires the 'ml' extra") from exc
        # XGBoost has no class_weight; imbalance handled via sample weights at fit time.
        return XGBClassifier(**hp)
    raise ValueError(f"Unknown classical family: {cfg.family}")


class ClassicalModel:
    def __init__(self, cfg: ModelConfig, classes: list[str] | None = None) -> None:
        self.cfg = cfg
        self.feature_spec: FeatureSpec = cfg.feature
        self.classes: list[str] = classes or []
        self._pipeline: Pipeline | None = None

    def fit(self, x: np.ndarray, y: list[str]) -> None:
        if not self.classes:
            self.classes = sorted(set(y))
        self._pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", _make_estimator(self.cfg))]
        )
        fit_kwargs = {}
        if self.cfg.family == "xgboost" and self.cfg.class_weight_balanced:
            fit_kwargs["clf__sample_weight"] = _balanced_sample_weights(y)
        self._pipeline.fit(x, y, **fit_kwargs)
        # Align class order to the estimator's learned order.
        self.classes = list(self._pipeline.named_steps["clf"].classes_)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not fitted")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._pipeline.predict_proba(x).astype(np.float32)

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        meta = {
            "family": self.cfg.family,
            "name": self.cfg.name,
            "classes": self.classes,
            "feature_spec": self.feature_spec.model_dump(),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> ClassicalModel:
        import joblib

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        cfg = ModelConfig(
            family=meta["family"],
            name=meta["name"],
            feature=FeatureSpec(**meta["feature_spec"]),
        )
        obj = cls(cfg, classes=meta["classes"])
        obj._pipeline = joblib.load(path)
        return obj


def _balanced_sample_weights(y: list[str]) -> np.ndarray:
    from collections import Counter

    counts = Counter(y)
    n, k = len(y), len(counts)
    return np.array([n / (k * counts[label]) for label in y], dtype=np.float32)
