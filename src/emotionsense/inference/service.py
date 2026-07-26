"""Inference service: the CPU serving pipeline.

validate -> preprocess -> features (from the MODEL's own spec, ADR-3) -> predict ->
post-process. Variable-length clips are windowed and per-window probabilities averaged
(R-A7). The feature extractor is built from the model's stored ``feature_spec``, so
serve-time features can never drift from train-time features.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from emotionsense.common.schemas import (
    AudioMeta,
    FeatureSpec,
    ModelInfo,
    Prediction,
)
from emotionsense.inference.model_cache import ModelCache
from emotionsense.inference.validators import validate_duration
from emotionsense.ml.features import build_extractor
from emotionsense.ml.models.base import probs_to_dict
from emotionsense.ml.preprocess import load_audio, preprocess, window
from emotionsense.registry.artifact import load_model_verified
from emotionsense.registry.local_registry import LocalRegistry


class InferenceService:
    def __init__(
        self,
        registry: LocalRegistry,
        cache_size: int = 3,
        max_duration_sec: float = 30.0,
    ) -> None:
        self.registry = registry
        self.cache = ModelCache(cache_size)
        self.max_duration_sec = max_duration_sec

    def _get_model(self, ref: str):
        cached = self.cache.get(ref)
        if cached is not None:
            return cached
        rec = self.registry.get(ref)
        model = load_model_verified(rec.family, Path(rec.artifact_path), rec.sha256)
        self.cache.put(ref, model)
        return model

    def predict_path(self, audio_path: str | Path, model_ref: str | None = None) -> Prediction:
        rec = self.registry.get(model_ref) if model_ref else self.registry.default()
        spec = FeatureSpec(**rec.feature_spec)
        model = self._get_model(rec.ref)

        start = time.perf_counter()
        y = load_audio(audio_path, sr=spec.sample_rate)
        duration = len(y) / spec.sample_rate
        validate_duration(duration, self.max_duration_sec)
        y = preprocess(y, sr=spec.sample_rate)

        extractor = build_extractor(spec)
        windows = window(y, spec.sample_rate, spec.max_duration_sec)
        feats = np.vstack([extractor.extract(w) for w in windows])
        proba = model.predict_proba(feats).mean(axis=0)  # aggregate over windows (R-A7)
        latency_ms = int((time.perf_counter() - start) * 1000)

        prob_dict = probs_to_dict(proba, model.classes)
        top_idx = int(np.argmax(proba))
        return Prediction(
            predicted_label=model.classes[top_idx],
            confidence=float(proba[top_idx]),
            probabilities=prob_dict,
            model=ModelInfo(name=rec.name, version=rec.version),
            audio=AudioMeta(duration_sec=round(duration, 3), sample_rate=spec.sample_rate),
            latency_ms=latency_ms,
        )
