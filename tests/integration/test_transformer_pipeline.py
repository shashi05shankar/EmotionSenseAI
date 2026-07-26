"""Transformer Phase 1: Distil-HuBERT (frozen) + SVM head wiring and extraction.

The config-wiring and device-selection tests run everywhere (no torch needed). The actual
SSL extraction test requires the 'transformer' extra and is skipped when unavailable, so it
runs on Kaggle/Colab where torch + transformers are installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emotionsense.common.schemas import FeatureSpec
from emotionsense.common.yaml_config import load_yaml
from emotionsense.ml.features.ssl import SSLExtractor, _select_device
from emotionsense.ml.models.classical import ClassicalModel
from emotionsense.ml.models.registry import build_model
from emotionsense.training.runner import model_cfg_from_dict


@pytest.mark.unit
def test_distilhubert_config_is_svm_head_on_distilhubert_embeddings():
    cfg = model_cfg_from_dict(load_yaml("configs/models/distilhubert.yaml"))
    # Classifier is an SVM (trained ONLY on embeddings) via the reused classical path.
    assert cfg.family == "svm"
    assert isinstance(build_model(cfg), ClassicalModel)
    # Feature extractor is the frozen Distil-HuBERT transformer.
    assert cfg.feature.extractor == "ssl"
    assert cfg.feature.ssl_model == "ntu-spml/distilhubert"
    assert cfg.feature.aggregate == "mean_std"


@pytest.mark.unit
def test_select_device_prefers_cuda_when_available():
    class HasGpu:
        class cuda:  # noqa: N801  (mimics torch.cuda attribute access)
            @staticmethod
            def is_available() -> bool:
                return True

    class NoGpu:
        class cuda:  # noqa: N801  (mimics torch.cuda attribute access)
            @staticmethod
            def is_available() -> bool:
                return False

    assert _select_device(HasGpu) == "cuda"
    assert _select_device(NoGpu) == "cpu"


@pytest.mark.unit
def test_ssl_extractor_requires_ssl_model():
    with pytest.raises(ValueError):
        SSLExtractor(FeatureSpec(extractor="ssl", ssl_model=None))


@pytest.mark.integration
def test_ssl_extractor_pools_last_hidden_state():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    spec = FeatureSpec(extractor="ssl", ssl_model="fake/distilhubert", aggregate="mean_std")
    ext = SSLExtractor(spec)

    def fake_proc(y, sampling_rate, return_tensors):
        return {"input_values": torch.zeros(1, len(y))}

    class FakeOut:
        last_hidden_state = torch.randn(1, 5, 8)  # (batch, frames, dim=8)

    fake_model = MagicMock()
    fake_model.return_value = FakeOut()
    fake_model.parameters.return_value = []

    with (
        patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=fake_proc),
        patch("transformers.AutoModel.from_pretrained", return_value=fake_model),
    ):
        vec = ext.extract(np.zeros(16000, dtype=np.float32))

    # mean_std pooling of dim-8 hidden states -> 16-dim vector.
    assert vec.shape == (16,)
    assert vec.dtype == np.float32
