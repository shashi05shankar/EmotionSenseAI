"""Evaluation metric tests — including the UA-is-imbalance-robust property (R-ML3)."""

from __future__ import annotations

import pytest

from emotionsense.training.evaluate import aggregate, compute_metrics


@pytest.mark.unit
def test_perfect_prediction_scores_one():
    labels = ["angry", "happy", "sad"]
    m = compute_metrics(labels, labels, labels)
    assert m.accuracy == 1.0
    assert m.unweighted_accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.confusion == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.mark.unit
def test_ua_penalizes_majority_predictor():
    # Imbalanced: 8 angry, 2 sad. A majority predictor gets accuracy 0.8 but UA 0.5.
    labels = ["angry", "sad"]
    y_true = ["angry"] * 8 + ["sad"] * 2
    y_pred = ["angry"] * 10
    m = compute_metrics(y_true, y_pred, labels)
    assert m.accuracy == pytest.approx(0.8)
    assert m.unweighted_accuracy == pytest.approx(0.5)  # mean per-class recall
    assert m.per_class_recall["sad"] == pytest.approx(0.0)


@pytest.mark.unit
def test_per_class_f1_keys_match_labels():
    labels = ["angry", "happy", "sad"]
    y_true = ["angry", "happy", "sad", "angry"]
    y_pred = ["angry", "sad", "sad", "angry"]
    m = compute_metrics(y_true, y_pred, labels)
    assert set(m.per_class_f1) == set(labels)
    assert set(m.per_class_recall) == set(labels)


@pytest.mark.unit
def test_aggregate_mean_and_std():
    labels = ["a", "b"]
    perfect = compute_metrics(labels, labels, labels)
    half = compute_metrics(["a", "b"], ["a", "a"], labels)  # ua = 0.5
    agg = aggregate([perfect, half])
    assert agg.n_folds == 2
    assert agg.ua_mean == pytest.approx((1.0 + 0.5) / 2)
    assert agg.ua_std > 0.0
    assert len(agg.per_fold_ua) == 2


@pytest.mark.unit
def test_weighted_f1_is_computed_and_prevalence_weighted():
    labels = ["angry", "sad"]
    # 8 angry, 2 sad; predict all angry -> angry F1 high, sad F1 = 0.
    y_true = ["angry"] * 8 + ["sad"] * 2
    y_pred = ["angry"] * 10
    m = compute_metrics(y_true, y_pred, labels)
    # weighted-F1 (prevalence-weighted) should exceed macro-F1 (unweighted) here.
    assert m.weighted_f1 > m.macro_f1
    assert 0.0 <= m.weighted_f1 <= 1.0


@pytest.mark.unit
def test_aggregate_includes_weighted_f1():
    labels = ["a", "b"]
    perfect = compute_metrics(labels, labels, labels)
    half = compute_metrics(["a", "b"], ["a", "a"], labels)
    agg = aggregate([perfect, half])
    assert 0.0 <= agg.weighted_f1_mean <= 1.0
    assert agg.weighted_f1_std >= 0.0
