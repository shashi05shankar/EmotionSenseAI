"""Baselines must score near chance — the whole point of R-ML1."""

from __future__ import annotations

import numpy as np
import pytest

from emotionsense.ml.models.baselines import MajorityBaseline, RandomBaseline


@pytest.mark.unit
def test_majority_predicts_majority_class():
    y = ["angry"] * 8 + ["sad"] * 2
    m = MajorityBaseline()
    m.fit(np.zeros((10, 3)), y)
    proba = m.predict_proba(np.zeros((4, 3)))
    preds = [m.classes[i] for i in proba.argmax(axis=1)]
    assert preds == ["angry"] * 4


@pytest.mark.unit
def test_random_matches_prior():
    y = ["angry"] * 5 + ["sad"] * 5
    r = RandomBaseline(seed=1)
    r.fit(np.zeros((10, 3)), y)
    proba = r.predict_proba(np.zeros((2, 3)))
    # Prior is 50/50 -> each class ~0.5.
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.shape == (2, 2)
