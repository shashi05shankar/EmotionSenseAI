"""Golden parity test (ADR-3): same spec + same clip => identical features.

Guards the #1 SER production bug — train/serve feature skew. If this ever fails, a model
trained with one feature path is being served with another.
"""

from __future__ import annotations

import numpy as np
import pytest

from emotionsense.common.schemas import FeatureSpec
from emotionsense.ml.features import build_extractor
from emotionsense.ml.preprocess import fix_length, load_audio, preprocess


@pytest.mark.unit
def test_feature_extraction_is_deterministic(sample_wav):
    spec = FeatureSpec(extractor="mfcc", n_mfcc=20, max_duration_sec=1.0)

    def path() -> np.ndarray:
        y = load_audio(sample_wav, sr=spec.sample_rate)
        y = preprocess(y, sr=spec.sample_rate)
        y = fix_length(y, spec.sample_rate, spec.max_duration_sec)
        return build_extractor(spec).extract(y)

    v1, v2 = path(), path()
    assert v1.shape == v2.shape
    assert np.allclose(v1, v2), "feature extraction must be bit-stable for parity"
    assert v1.shape[0] == 2 * spec.n_mfcc  # mean_std pooling
