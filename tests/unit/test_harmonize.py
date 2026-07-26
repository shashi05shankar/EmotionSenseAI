"""Label harmonization + cross-corpus intersection (R-ML4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from emotionsense.common.constants import CROSS_CORPUS_LABELS
from emotionsense.datasets.base import Sample
from emotionsense.datasets.harmonize import harmonize, project_to_cross_corpus


def _s(label: str, dataset: str) -> Sample:
    return Sample(path=Path("x.wav"), label=label, speaker_id="s1", dataset=dataset)


@pytest.mark.unit
def test_ravdess_calm_is_dropped():
    pairs = harmonize([_s("calm", "ravdess"), _s("happy", "ravdess")])
    labels = [lab for _, lab in pairs]
    assert "happy" in labels
    assert len(pairs) == 1  # calm dropped


@pytest.mark.unit
def test_cross_corpus_projection_drops_surprise():
    pairs = [(_s("surprised", "ravdess"), "surprise"), (_s("angry", "ravdess"), "angry")]
    projected = project_to_cross_corpus(pairs)
    labels = {lab for _, lab in projected}
    assert labels <= set(CROSS_CORPUS_LABELS)
    assert "surprise" not in labels
    assert "angry" in labels
