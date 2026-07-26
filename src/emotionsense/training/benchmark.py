"""Unified benchmark harness — the project's differentiator (FR-E1..E4).

Runs every configured model under IDENTICAL speaker-independent k-fold splits and the same
metrics, producing one apples-to-apples leaderboard. Also runs cross-corpus evaluation
(train on corpus A, test on held-out corpus B) after projecting both to the shared 6-class
label intersection (R-ML4).

This is deliberately a first-class subsystem, not a script: honest, comparable evaluation
is the thing that distinguishes this project from every repo in the landscape survey.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from emotionsense.common.constants import CANONICAL_LABELS, CROSS_CORPUS_LABELS
from emotionsense.common.schemas import FeatureSpec
from emotionsense.datasets.base import Sample
from emotionsense.datasets.harmonize import project_to_cross_corpus
from emotionsense.datasets.split import (
    assert_speaker_independent,
    make_speaker_independent_folds,
)
from emotionsense.ml.models.base import ModelConfig
from emotionsense.ml.models.registry import build_model
from emotionsense.training.evaluate import Metrics, aggregate, compute_metrics
from emotionsense.training.featureset import build_matrix


@dataclass(slots=True)
class LeaderboardRow:
    model: str
    dataset: str
    eval_kind: str  # 'cv' | 'cross_corpus'
    ua_mean: float
    ua_std: float
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    n_folds: int
    per_class_f1: dict[str, float] = field(default_factory=dict)
    # Extended metrics (Transformer Phase 1) — reported identically for every model.
    weighted_f1_mean: float = 0.0
    extract_time_s: float = 0.0  # mean per fold: audio -> features/embeddings (0 if cached)
    train_time_s: float = 0.0  # mean per fold: classifier fit
    infer_ms_per_sample: float = 0.0  # mean per fold: classifier predict, ms / test sample
    confusion: list[list[int]] = field(default_factory=list)  # CV-aggregated over all folds
    confusion_labels: list[str] = field(default_factory=list)


def _predict_labels(model, x: np.ndarray) -> list[str]:
    proba = model.predict_proba(x)
    idx = np.argmax(proba, axis=1)
    return [model.classes[i] for i in idx]


def run_cv(
    model_cfg: ModelConfig,
    pairs: list[tuple[Sample, str]],
    dataset_name: str,
    n_folds: int = 5,
    seed: int = 42,
    cache_dir: Path | None = None,
    labels: list[str] | None = None,
) -> LeaderboardRow:
    """Speaker-independent k-fold evaluation of one model on one corpus."""
    labels = labels or list(CANONICAL_LABELS)
    folds = make_speaker_independent_folds(pairs, n_folds=n_folds, seed=seed)
    fold_metrics: list[Metrics] = []
    extract_times, train_times, infer_times = [], [], []
    all_true: list[str] = []
    all_pred: list[str] = []
    for fold in folds:
        assert_speaker_independent(fold)  # guardrail every fold
        t0 = time.perf_counter()
        x_tr, y_tr = build_matrix(fold.train, model_cfg.feature, cache_dir)
        x_te, y_te = build_matrix(fold.test, model_cfg.feature, cache_dir)
        extract_times.append(time.perf_counter() - t0)

        present = sorted(set(y_tr) | set(y_te))
        eval_labels = [c for c in labels if c in present]
        model = build_model(model_cfg)

        t0 = time.perf_counter()
        model.fit(x_tr, y_tr)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = _predict_labels(model, x_te)
        infer_times.append((time.perf_counter() - t0) / max(1, len(y_te)) * 1000.0)

        fold_metrics.append(compute_metrics(y_te, y_pred, eval_labels))
        all_true.extend(y_te)
        all_pred.extend(y_pred)

    agg = aggregate(fold_metrics)
    # Average per-class F1 across folds (union of labels).
    per_class: dict[str, list[float]] = {}
    for m in fold_metrics:
        for lab, v in m.per_class_f1.items():
            per_class.setdefault(lab, []).append(v)
    per_class_mean = {k: float(np.mean(v)) for k, v in per_class.items()}

    # CV-aggregated confusion matrix: folds partition the test sets, so concatenating all
    # fold predictions yields one prediction per sample over the whole dataset.
    cm_labels = [c for c in labels if c in set(all_true) | set(all_pred)]
    overall = compute_metrics(all_true, all_pred, cm_labels)

    return LeaderboardRow(
        model=model_cfg.name,
        dataset=dataset_name,
        eval_kind="cv",
        ua_mean=agg.ua_mean,
        ua_std=agg.ua_std,
        accuracy_mean=agg.accuracy_mean,
        accuracy_std=agg.accuracy_std,
        macro_f1_mean=agg.macro_f1_mean,
        n_folds=agg.n_folds,
        per_class_f1=per_class_mean,
        weighted_f1_mean=agg.weighted_f1_mean,
        extract_time_s=float(np.mean(extract_times)),
        train_time_s=float(np.mean(train_times)),
        infer_ms_per_sample=float(np.mean(infer_times)),
        confusion=overall.confusion,
        confusion_labels=overall.labels,
    )


def run_cross_corpus(
    model_cfg: ModelConfig,
    train_pairs: list[tuple[Sample, str]],
    test_pairs: list[tuple[Sample, str]],
    train_name: str,
    test_name: str,
    cache_dir: Path | None = None,
) -> LeaderboardRow:
    """Train on one corpus, evaluate on a DIFFERENT held-out corpus (R-ML4).

    Both corpora are projected to the 6-class intersection first so the label spaces
    match — otherwise the result is meaningless.
    """
    train_pairs = project_to_cross_corpus(train_pairs)
    test_pairs = project_to_cross_corpus(test_pairs)
    labels = [c for c in CROSS_CORPUS_LABELS if c in {lab for _, lab in test_pairs}]

    t0 = time.perf_counter()
    x_tr, y_tr = build_matrix(train_pairs, model_cfg.feature, cache_dir)
    x_te, y_te = build_matrix(test_pairs, model_cfg.feature, cache_dir)
    extract_time = time.perf_counter() - t0

    model = build_model(model_cfg)
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = _predict_labels(model, x_te)
    infer_ms = (time.perf_counter() - t0) / max(1, len(y_te)) * 1000.0

    m = compute_metrics(y_te, y_pred, labels)

    return LeaderboardRow(
        model=model_cfg.name,
        dataset=f"{train_name}->{test_name}",
        eval_kind="cross_corpus",
        ua_mean=m.unweighted_accuracy,
        ua_std=0.0,
        accuracy_mean=m.accuracy,
        accuracy_std=0.0,
        macro_f1_mean=m.macro_f1,
        n_folds=1,
        per_class_f1=m.per_class_f1,
        weighted_f1_mean=m.weighted_f1,
        extract_time_s=extract_time,
        train_time_s=train_time,
        infer_ms_per_sample=infer_ms,
        confusion=m.confusion,
        confusion_labels=m.labels,
    )


def leaderboard_to_dicts(rows: list[LeaderboardRow]) -> list[dict]:
    """Serialize + sort rows: cross-corpus last, then by UA descending."""
    ordered = sorted(rows, key=lambda r: (r.eval_kind == "cross_corpus", -r.ua_mean))
    return [asdict(r) for r in ordered]


def build_feature_spec(cfg: dict) -> FeatureSpec:
    """Helper: construct a FeatureSpec from a model config's 'feature' block."""
    return FeatureSpec(**cfg)
