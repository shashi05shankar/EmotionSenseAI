"""Manifest round-trip: write folds -> read back -> reconstruct pairs."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotionsense.datasets.manifest import read_manifest, rows_to_pairs, write_fold
from emotionsense.datasets.split import make_speaker_independent_folds


@pytest.mark.unit
def test_write_read_roundtrip(synthetic_pairs, tmp_path: Path):
    fold = make_speaker_independent_folds(synthetic_pairs, n_folds=3, seed=42)[0]
    path = write_fold(fold, tmp_path, "synthetic", seed=42)
    assert path.exists()

    rows = read_manifest(path)
    total = len(fold.train) + len(fold.val) + len(fold.test)
    assert len(rows) == total
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert all(r["seed"] == 42 for r in rows)

    train_pairs = rows_to_pairs(rows, "train")
    assert len(train_pairs) == len(fold.train)
    sample, label = train_pairs[0]
    assert sample.dataset == "synthetic"
    assert label  # canonical label preserved


@pytest.mark.unit
def test_rows_to_pairs_filters_by_split(synthetic_pairs, tmp_path: Path):
    fold = make_speaker_independent_folds(synthetic_pairs, n_folds=3, seed=1)[0]
    path = write_fold(fold, tmp_path, "synthetic", seed=1)
    rows = read_manifest(path)
    test_pairs = rows_to_pairs(rows, "test")
    assert len(test_pairs) == len(fold.test)
    # test speakers must be disjoint from train speakers (speaker-independent)
    test_speakers = {s.speaker_id for s, _ in test_pairs}
    train_speakers = {s.speaker_id for s, _ in rows_to_pairs(rows, "train")}
    assert not (test_speakers & train_speakers)
