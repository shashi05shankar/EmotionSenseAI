"""Shared pytest fixtures: synthetic corpus + tiny sample clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from emotionsense.datasets.loaders import SyntheticLoader
from emotionsense.datasets.loaders.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_corpus")
    generate(root, speakers=6, clips_per_class=3, duration=0.6)
    return root


@pytest.fixture(scope="session")
def synthetic_pairs(synthetic_root: Path):
    from emotionsense.datasets.harmonize import harmonize

    samples = SyntheticLoader().load(synthetic_root)
    return harmonize(samples)


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "sample.wav"
    sf.write(p, y, sr)
    return p
