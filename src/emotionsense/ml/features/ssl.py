"""Self-supervised (SSL) embedding extractor: Distil-HuBERT / HuBERT / wav2vec2.

Frozen-embedding strategy (design review, cheap on free GPU): run a pretrained SSL model
as a feature extractor and mean-pool the last hidden state into a fixed vector, which a
light classical/MLP head then classifies. Torch + transformers are optional extras, so
this module imports them lazily and is skipped in the default (light) test run.
"""

from __future__ import annotations

import numpy as np

from emotionsense.common.schemas import FeatureSpec


class SSLExtractor:
    """Lazy-loaded SSL embedding extractor. Requires the ``transformer`` extra."""

    def __init__(self, spec: FeatureSpec) -> None:
        if spec.ssl_model is None:
            raise ValueError("FeatureSpec.ssl_model must be set for extractor='ssl'")
        self.spec = spec
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover - exercised only with extra installed
            raise RuntimeError(
                "SSL features require the 'transformer' extra: pip install -e '.[transformer]'"
            ) from exc
        self._processor = AutoFeatureExtractor.from_pretrained(self.spec.ssl_model)
        self._model = AutoModel.from_pretrained(self.spec.ssl_model)
        self._model.eval()

    def extract(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - needs heavy extra
        import torch

        self._ensure_loaded()
        assert self._processor is not None and self._model is not None
        inputs = self._processor(y, sampling_rate=self.spec.sample_rate, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**inputs)
        hidden = out.last_hidden_state.squeeze(0).cpu().numpy()  # (frames, dim)
        if self.spec.aggregate == "mean":
            return hidden.mean(axis=0).astype(np.float32)
        return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)]).astype(np.float32)
