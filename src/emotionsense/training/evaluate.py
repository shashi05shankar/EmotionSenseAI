"""Evaluation metrics with an imbalance-robust headline (design review R-ML3).

Metrics reported:
* **Unweighted Accuracy (UA)** — mean per-class recall; the HEADLINE metric because it is
  robust to class imbalance (a majority-class predictor scores near chance, not high).
* Weighted Accuracy (WA) / plain accuracy — overall correct fraction.
* macro-F1 and per-class F1.
* Confusion matrix.

These are pure functions over (y_true, y_pred[, y_proba]); the harness aggregates them
across CV folds (R-ML2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


@dataclass(slots=True)
class Metrics:
    accuracy: float
    weighted_accuracy: float
    unweighted_accuracy: float  # headline
    macro_f1: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_recall: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> Metrics:
    """Compute the full metric bundle for one evaluation."""
    acc = float(accuracy_score(y_true, y_pred))
    ua = float(balanced_accuracy_score(y_true, y_pred))  # mean per-class recall
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(
        accuracy=acc,
        weighted_accuracy=acc,  # equal to plain accuracy for single-label; kept explicit
        unweighted_accuracy=ua,
        macro_f1=macro_f1,
        per_class_f1={lab: float(v) for lab, v in zip(labels, per_f1, strict=True)},
        per_class_recall={lab: float(v) for lab, v in zip(labels, per_rec, strict=True)},
        confusion=cm.tolist(),
        labels=labels,
    )


@dataclass(slots=True)
class AggregateMetrics:
    """Mean ± std across CV folds (R-ML2)."""

    accuracy_mean: float
    accuracy_std: float
    ua_mean: float
    ua_std: float
    macro_f1_mean: float
    macro_f1_std: float
    n_folds: int
    per_fold_ua: list[float] = field(default_factory=list)


def aggregate(fold_metrics: list[Metrics]) -> AggregateMetrics:
    """Aggregate per-fold metrics into mean ± std."""
    acc = np.array([m.accuracy for m in fold_metrics])
    ua = np.array([m.unweighted_accuracy for m in fold_metrics])
    f1 = np.array([m.macro_f1 for m in fold_metrics])
    return AggregateMetrics(
        accuracy_mean=float(acc.mean()),
        accuracy_std=float(acc.std()),
        ua_mean=float(ua.mean()),
        ua_std=float(ua.std()),
        macro_f1_mean=float(f1.mean()),
        macro_f1_std=float(f1.std()),
        n_folds=len(fold_metrics),
        per_fold_ua=ua.tolist(),
    )
