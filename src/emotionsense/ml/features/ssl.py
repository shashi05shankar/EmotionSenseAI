"""Self-supervised (SSL) embedding extractor: Distil-HuBERT / HuBERT / wav2vec2.

Frozen-embedding strategy (design review, cheap on free GPU): run a pretrained SSL model
as a **frozen** feature extractor and pool the last hidden state into a fixed vector, which
a light classical head (an SVM) then classifies. The transformer is never fine-tuned.

Runs on **GPU when available** (Kaggle/Colab): the model and inputs are moved to CUDA if
`torch.cuda.is_available()`, otherwise CPU. Torch + transformers are optional extras, so
this module imports them lazily and is excluded from the default (light) test run.
"""

from __future__ import annotations

import numpy as np

from emotionsense.common.schemas import FeatureSpec


def _select_device(torch_module) -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch_module.cuda.is_available() else "cpu"


class SSLExtractor:
    """Lazy-loaded, frozen SSL embedding extractor. Requires the ``transformer`` extra."""

    def __init__(self, spec: FeatureSpec) -> None:
        if spec.ssl_model is None:
            raise ValueError("FeatureSpec.ssl_model must be set for extractor='ssl'")
        self.spec = spec
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover - exercised only with extra installed
            raise RuntimeError(
                "SSL features require the 'transformer' extra: pip install -e '.[transformer]'"
            ) from exc
        self._device = _select_device(torch)
        self._processor = AutoFeatureExtractor.from_pretrained(self.spec.ssl_model)
        self._model = AutoModel.from_pretrained(self.spec.ssl_model)
        self._model.eval()
        self._model.to(self._device)
        # Frozen: no parameter ever requires a gradient.
        for p in self._model.parameters():
            p.requires_grad_(False)

    def extract(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - needs heavy extra
        self._ensure_loaded()  # raises a friendly error if the 'transformer' extra is absent
        import torch

        assert self._processor is not None and self._model is not None
        inputs = self._processor(y, sampling_rate=self.spec.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # (frames, dim)
        if self.spec.aggregate == "mean":
            return hidden.mean(axis=0).astype(np.float32)
        return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)]).astype(np.float32)
