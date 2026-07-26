"""Deep sequence models (CNN / LSTM / BiLSTM) on framed spectral features.

Requires the ``deep`` extra (torch + safetensors). Weights are saved via **safetensors**
(design review R-SEC1) — never pickle — so loading a compromised artifact cannot execute
arbitrary code. This module is a Should-have in Lean Core; it imports torch lazily and is
excluded from the default light test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emotionsense.common.schemas import FeatureSpec
from emotionsense.ml.models.base import ModelConfig


class DeepSequenceModel:  # pragma: no cover - exercised only with the 'deep' extra
    """A small CNN/LSTM/BiLSTM classifier over pooled spectral feature vectors."""

    def __init__(self, cfg: ModelConfig, classes: list[str] | None = None) -> None:
        self.cfg = cfg
        self.feature_spec: FeatureSpec = cfg.feature
        self.classes: list[str] = classes or []
        self._net = None
        self._input_dim: int | None = None

    def _build_net(self, input_dim: int, n_classes: int):
        import torch.nn as nn

        family = self.cfg.family
        hidden = int(self.cfg.hyperparams.get("hidden", 128))
        if family in {"lstm", "bilstm"}:
            bidir = family == "bilstm"
            return _RNNHead(input_dim, hidden, n_classes, bidirectional=bidir)
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def fit(self, x: np.ndarray, y: list[str]) -> None:
        import torch
        from torch import nn, optim

        if not self.classes:
            self.classes = sorted(set(y))
        self._input_dim = x.shape[1]
        y_idx = torch.tensor([self.classes.index(c) for c in y])
        x_t = torch.tensor(x, dtype=torch.float32)

        # Class-weighted loss for imbalance (R-ML3).
        from collections import Counter

        counts = Counter(y)
        weights = torch.tensor(
            [len(y) / (len(self.classes) * counts.get(c, 1)) for c in self.classes],
            dtype=torch.float32,
        )
        self._net = self._build_net(self._input_dim, len(self.classes))
        opt = optim.Adam(self._net.parameters(), lr=self.cfg.hyperparams.get("lr", 1e-3))
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        epochs = int(self.cfg.hyperparams.get("epochs", 30))
        self._net.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self._net(x_t)
            loss = loss_fn(logits, y_idx)
            loss.backward()
            opt.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Model not fitted")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(x, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs.astype(np.float32)

    def save(self, path: Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.contiguous() for k, v in self._net.state_dict().items()}
        save_file(state, str(path))
        meta = {
            "family": self.cfg.family,
            "name": self.cfg.name,
            "classes": self.classes,
            "input_dim": self._input_dim,
            "hyperparams": self.cfg.hyperparams,
            "feature_spec": self.feature_spec.model_dump(),
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> DeepSequenceModel:
        from safetensors.torch import load_file

        path = Path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        cfg = ModelConfig(
            family=meta["family"],
            name=meta["name"],
            feature=FeatureSpec(**meta["feature_spec"]),
            hyperparams=meta["hyperparams"],
        )
        obj = cls(cfg, classes=meta["classes"])
        obj._input_dim = meta["input_dim"]
        obj._net = obj._build_net(obj._input_dim, len(obj.classes))
        obj._net.load_state_dict(load_file(str(path)))
        return obj


class _RNNHead:  # pragma: no cover
    """Wraps an (B, F) vector as a length-1 sequence for LSTM/BiLSTM demonstration."""

    def __new__(cls, input_dim: int, hidden: int, n_classes: int, bidirectional: bool):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=bidirectional)
                self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

            def forward(self, x):  # x: (B, F)
                out, _ = self.rnn(x.unsqueeze(1))
                return self.head(out[:, -1, :])

        return Net()
