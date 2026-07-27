"""Explainability: per-window timeline for a clip."""

from __future__ import annotations

import numpy as np
import pytest

from emotionsense.common.schemas import FeatureSpec
from emotionsense.inference.explain import explain_audio
from emotionsense.ml.models.base import ModelConfig
from emotionsense.ml.models.classical import ClassicalModel


@pytest.fixture
def trained_svm(synthetic_pairs):
    from emotionsense.training.featureset import build_matrix

    spec = FeatureSpec(extractor="mfcc", n_mfcc=20, max_duration_sec=0.6)
    x, y = build_matrix(synthetic_pairs, spec)
    model = ClassicalModel(ModelConfig(family="svm", name="svm", feature=spec))
    model.fit(x, y)
    return model, spec


@pytest.mark.integration
def test_explain_returns_per_window_breakdown(trained_svm, synthetic_root):
    model, spec = trained_svm
    clip = next(synthetic_root.rglob("*angry*.wav"))
    expl = explain_audio(clip, model, spec)

    assert expl.predicted_label in expl.probabilities
    assert 0.0 <= expl.confidence <= 1.0
    assert abs(sum(expl.probabilities.values()) - 1.0) < 0.05
    assert expl.n_windows >= 1
    assert len(expl.windows) == expl.n_windows
    for w in expl.windows:
        assert w.label in model.classes
        assert w.end_sec >= w.start_sec
        assert abs(sum(w.probabilities.values()) - 1.0) < 0.05


@pytest.mark.integration
def test_explain_windows_a_long_clip(trained_svm, tmp_path):
    import soundfile as sf

    model, spec = trained_svm
    sr = spec.sample_rate
    # 2.0s clip with a 0.6s window -> multiple windows
    y = (0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 2.0, sr * 2))).astype(np.float32)
    p = tmp_path / "long.wav"
    sf.write(p, y, sr)
    expl = explain_audio(p, model, spec)
    assert expl.n_windows >= 2
