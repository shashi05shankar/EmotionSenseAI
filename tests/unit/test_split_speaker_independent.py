"""The integrity test: no speaker may leak across train/val/test (ADR-4, R-ML2)."""

from __future__ import annotations

import pytest

from emotionsense.datasets.split import (
    assert_speaker_independent,
    make_speaker_independent_folds,
)


@pytest.mark.unit
def test_folds_are_speaker_independent(synthetic_pairs):
    folds = make_speaker_independent_folds(synthetic_pairs, n_folds=3, seed=42)
    assert len(folds) == 3
    for fold in folds:
        # Must not raise.
        assert_speaker_independent(fold)
        tr, va, te = fold.speakers("train"), fold.speakers("val"), fold.speakers("test")
        assert not (tr & te)
        assert not (tr & va)
        assert not (va & te)
        assert te, "every fold must have a non-empty test set"


@pytest.mark.unit
def test_folds_are_deterministic(synthetic_pairs):
    a = make_speaker_independent_folds(synthetic_pairs, n_folds=3, seed=7)
    b = make_speaker_independent_folds(synthetic_pairs, n_folds=3, seed=7)
    assert [f.speakers("test") for f in a] == [f.speakers("test") for f in b]


@pytest.mark.unit
def test_too_few_speakers_raises(synthetic_pairs):
    with pytest.raises(ValueError):
        make_speaker_independent_folds(synthetic_pairs, n_folds=999, seed=42)
