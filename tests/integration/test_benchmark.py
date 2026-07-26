"""Benchmark harness tests: speaker-independent CV + cross-corpus (R-ML2, R-ML4).

These exercise the project's differentiator end-to-end on the synthetic fixture. Marked
integration because they extract features from audio files.
"""

from __future__ import annotations

import pytest

from emotionsense.common.constants import CROSS_CORPUS_LABELS
from emotionsense.common.schemas import FeatureSpec
from emotionsense.ml.models.base import ModelConfig
from emotionsense.training.benchmark import (
    leaderboard_to_dicts,
    run_cross_corpus,
    run_cv,
)


def _cfg(family: str, name: str) -> ModelConfig:
    return ModelConfig(
        family=family,
        name=name,
        feature=FeatureSpec(extractor="mfcc", n_mfcc=20, max_duration_sec=0.6),
        hyperparams={} if family != "logreg" else {"C": 1.0},
    )


@pytest.mark.integration
def test_run_cv_returns_valid_row(synthetic_pairs):
    row = run_cv(_cfg("logreg", "logreg"), synthetic_pairs, "synthetic", n_folds=3, seed=42)
    assert row.eval_kind == "cv"
    assert row.n_folds == 3
    assert 0.0 <= row.ua_mean <= 1.0
    assert row.ua_std >= 0.0
    assert row.per_class_f1  # non-empty


@pytest.mark.integration
def test_run_cv_reports_extended_metrics(synthetic_pairs):
    row = run_cv(_cfg("logreg", "logreg"), synthetic_pairs, "synthetic", n_folds=3, seed=42)
    # Transformer Phase 1 metrics, reported identically for every model.
    assert 0.0 <= row.weighted_f1_mean <= 1.0
    assert row.train_time_s >= 0.0
    assert row.infer_ms_per_sample >= 0.0
    assert row.extract_time_s >= 0.0
    # CV-aggregated confusion matrix: square over the present labels.
    assert row.confusion_labels
    assert len(row.confusion) == len(row.confusion_labels)
    assert all(len(r) == len(row.confusion_labels) for r in row.confusion)
    # confusion counts sum to the number of test samples across all folds (== dataset size).
    assert sum(sum(r) for r in row.confusion) == len(synthetic_pairs)


@pytest.mark.integration
def test_majority_baseline_is_near_chance(synthetic_pairs):
    row = run_cv(_cfg("majority", "majority"), synthetic_pairs, "synthetic", n_folds=3, seed=42)
    # 7 classes -> chance UA ~ 1/7. Baseline must not look competitive.
    assert row.ua_mean < 0.3


@pytest.mark.integration
def test_cross_corpus_projects_to_intersection(synthetic_pairs):
    # Use the synthetic corpus as both train and (held-out) test to exercise the
    # cross-corpus path; assertion is on label projection + structure, not generalization.
    row = run_cross_corpus(
        _cfg("logreg", "logreg"),
        synthetic_pairs,
        synthetic_pairs,
        train_name="synthetic",
        test_name="synthetic2",
    )
    assert row.eval_kind == "cross_corpus"
    assert row.dataset == "synthetic->synthetic2"
    # only the 6-class intersection may appear (surprise must be dropped)
    assert set(row.per_class_f1) <= set(CROSS_CORPUS_LABELS)
    assert "surprise" not in row.per_class_f1


@pytest.mark.integration
def test_leaderboard_sorts_cross_corpus_last(synthetic_pairs):
    cv = run_cv(_cfg("logreg", "logreg"), synthetic_pairs, "synthetic", n_folds=3)
    cc = run_cross_corpus(
        _cfg("logreg", "logreg"), synthetic_pairs, synthetic_pairs, "synthetic", "synthetic2"
    )
    dicts = leaderboard_to_dicts([cc, cv])
    assert dicts[-1]["eval_kind"] == "cross_corpus"  # cross-corpus always ranked last
