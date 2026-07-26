"""Model definitions shared by training and inference."""

from emotionsense.ml.models.base import EmotionModel, ModelConfig
from emotionsense.ml.models.baselines import MajorityBaseline, RandomBaseline
from emotionsense.ml.models.classical import ClassicalModel
from emotionsense.ml.models.registry import build_model

__all__ = [
    "ClassicalModel",
    "EmotionModel",
    "MajorityBaseline",
    "ModelConfig",
    "RandomBaseline",
    "build_model",
]
