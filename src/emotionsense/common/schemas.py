"""Shared Pydantic DTOs used across planes (API, inference, training)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Identifies a served model version."""

    name: str
    version: str


class AudioMeta(BaseModel):
    """Metadata about a processed audio clip."""

    duration_sec: float
    sample_rate: int


class Prediction(BaseModel):
    """A single emotion prediction result."""

    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model: ModelInfo
    audio: AudioMeta
    latency_ms: int


class FeatureSpec(BaseModel):
    """Preprocessing + feature configuration that travels WITH a model (ADR-3).

    The inference path reads this from model metadata so serve-time features exactly
    match train-time features. This is the single most important anti-skew mechanism.
    """

    extractor: str  # 'mfcc' | 'mel' | 'chroma' | 'ssl'
    sample_rate: int = 16_000
    n_mfcc: int = 40
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    max_duration_sec: float = 4.0
    ssl_model: str | None = None  # e.g. 'ntu-spml/distilhubert' when extractor == 'ssl'
    aggregate: str = "mean_std"  # pooling over time frames


class EvalResult(BaseModel):
    """One evaluation record for the leaderboard."""

    model: str
    dataset: str
    eval_kind: str  # 'validation' | 'test' | 'cross_corpus'
    accuracy: float
    weighted_accuracy: float
    unweighted_accuracy: float  # headline metric (imbalance-robust, R-ML3)
    macro_f1: float
    accuracy_std: float = 0.0  # std across CV folds (R-ML2)
    ua_std: float = 0.0
    n_folds: int = 1
    per_class_f1: dict[str, float] = Field(default_factory=dict)
