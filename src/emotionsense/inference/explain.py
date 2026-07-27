"""Lightweight explainability: per-window emotion timeline.

The serving path already windows a clip and averages per-window probabilities. This module
exposes those *per-window* predictions so you can see **which parts of a clip drove which
emotion** — a cheap, honest form of interpretability that needs no extra model. It reuses the
same preprocessing + feature extractor as training/serving, so explanations are consistent
with predictions (ADR-3).

Caveat surfaced deliberately: softmax confidences are **uncalibrated** — read them as a
ranking, not a probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emotionsense.common.schemas import FeatureSpec
from emotionsense.ml.features import build_extractor
from emotionsense.ml.models.base import EmotionModel, probs_to_dict
from emotionsense.ml.preprocess import load_audio, preprocess, window


@dataclass(slots=True)
class WindowExplanation:
    index: int
    start_sec: float
    end_sec: float
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(slots=True)
class Explanation:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]  # aggregated (mean over windows)
    duration_sec: float
    n_windows: int
    windows: list[WindowExplanation] = field(default_factory=list)
    note: str = "Confidences are uncalibrated softmax scores — treat as ranking, not probability."


def explain_audio(audio_path: str | Path, model: EmotionModel, spec: FeatureSpec) -> Explanation:
    """Return the aggregated prediction plus a per-window breakdown for one clip."""
    y = load_audio(audio_path, sr=spec.sample_rate)
    y = preprocess(y, sr=spec.sample_rate)
    duration = len(y) / spec.sample_rate

    windows = window(y, spec.sample_rate, spec.max_duration_sec)
    extractor = build_extractor(spec)
    feats = np.vstack([extractor.extract(w) for w in windows])
    proba = model.predict_proba(feats)  # (n_windows, n_classes)
    classes = model.classes
    hop = spec.max_duration_sec

    win_expl: list[WindowExplanation] = []
    for i, p in enumerate(proba):
        top = int(np.argmax(p))
        win_expl.append(
            WindowExplanation(
                index=i,
                start_sec=round(i * hop, 3),
                end_sec=round(min((i + 1) * hop, duration), 3),
                label=classes[top],
                confidence=float(p[top]),
                probabilities=probs_to_dict(p, classes),
            )
        )

    agg = proba.mean(axis=0)
    top = int(np.argmax(agg))
    return Explanation(
        predicted_label=classes[top],
        confidence=float(agg[top]),
        probabilities=probs_to_dict(agg, classes),
        duration_sec=round(duration, 3),
        n_windows=len(windows),
        windows=win_expl,
    )


def explain_from_registry(
    audio_path: str | Path, registry, model_ref: str | None = None
) -> Explanation:
    """Resolve a registered model (default if unset), verify it, and explain a clip."""
    from emotionsense.registry.artifact import load_model_verified

    rec = registry.get(model_ref) if model_ref else registry.default()
    spec = FeatureSpec(**rec.feature_spec)
    model = load_model_verified(rec.family, Path(rec.artifact_path), rec.sha256)
    return explain_audio(audio_path, model, spec)
