"""Model factory: family name -> constructed model instance.

Central place mapping a declarative ``ModelConfig`` to a concrete model. Adding a new
model family = one line here + a config file (config over code).
"""

from __future__ import annotations

from emotionsense.ml.models.base import EmotionModel, ModelConfig
from emotionsense.ml.models.baselines import MajorityBaseline, RandomBaseline
from emotionsense.ml.models.classical import ClassicalModel

_CLASSICAL = {"svm", "logreg", "random_forest", "xgboost"}
_DEEP = {"cnn", "lstm", "bilstm"}


def build_model(cfg: ModelConfig) -> EmotionModel:
    """Construct a model from its config."""
    if cfg.family == "majority":
        return MajorityBaseline()
    if cfg.family == "random":
        return RandomBaseline(seed=cfg.hyperparams.get("seed", 42))
    if cfg.family in _CLASSICAL:
        return ClassicalModel(cfg)
    if cfg.family in _DEEP:
        from emotionsense.ml.models.deep import DeepSequenceModel

        return DeepSequenceModel(cfg)
    if cfg.family in {"distilhubert", "hubert", "wav2vec2", "emotion2vec"}:
        # Frozen SSL embeddings + a classical head reuse ClassicalModel with an SSL spec.
        return ClassicalModel(cfg)
    raise ValueError(f"Unknown model family: {cfg.family}")
