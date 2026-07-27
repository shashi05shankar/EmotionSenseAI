"""EmotionSense AI — self-contained Streamlit demo for Hugging Face Spaces.

Runs entirely in-process (no API, DB, or object store): on first load it trains a small SVM
on the synthetic fixture, registers it, then serves upload -> prediction -> per-window
explainability. Deployable to a free HF Space with just this file + requirements.txt.

NOTE: the bundled model is trained on the SYNTHETIC fixture so the Space is zero-dependency.
For real-speech predictions, train on RAVDESS/CREMA-D (see the repo README) and drop the
resulting artifact into the registry.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

from emotionsense.common.constants import CANONICAL_LABELS
from emotionsense.common.schemas import FeatureSpec
from emotionsense.datasets.build import build_pairs
from emotionsense.inference.explain import explain_from_registry
from emotionsense.ml.models.base import ModelConfig
from emotionsense.ml.models.classical import ClassicalModel
from emotionsense.registry.artifact import save_model_checksummed
from emotionsense.registry.local_registry import LocalRegistry, ModelRecord
from emotionsense.training.featureset import build_matrix

REG_ROOT = Path("experiments/artifacts/registry")
SPEC = FeatureSpec(extractor="mfcc", n_mfcc=40, max_duration_sec=3.0, aggregate="mean_std")


@st.cache_resource(show_spinner="Preparing the demo model…")
def get_registry() -> LocalRegistry:
    reg = LocalRegistry(REG_ROOT)
    try:
        reg.default()
        return reg
    except Exception:
        pass  # nothing registered yet — train the demo model
    dataset_cfg = {
        "name": "synthetic",
        "loader": "synthetic",
        "root": "data/raw/synthetic",
        "label_map": {lab: lab for lab in CANONICAL_LABELS},
    }
    pairs = build_pairs(dataset_cfg)
    cfg = ModelConfig(
        family="svm",
        name="svm-mfcc-demo",
        feature=SPEC,
        hyperparams={"C": 10.0, "kernel": "rbf"},
    )
    x, y = build_matrix(pairs, SPEC, cache_dir=Path("data/features"))
    model = ClassicalModel(cfg)
    model.fit(x, y)
    art = REG_ROOT / "artifacts" / "svm-mfcc-demo-v1.joblib"
    art.parent.mkdir(parents=True, exist_ok=True)
    sha = save_model_checksummed(model, art)
    rec = ModelRecord(
        name="svm-mfcc-demo",
        version="v1",
        family="svm",
        artifact_path=str(art),
        sha256=sha,
        feature_spec=SPEC.model_dump(),
        label_classes=model.classes,
    )
    reg.register(rec)
    reg.promote(rec.ref, make_default=True)
    return reg


st.set_page_config(page_title="EmotionSense AI", page_icon="🎙️", layout="wide")
st.title("🎙️ EmotionSense AI — live demo")
st.caption(
    "Upload a short clip → emotion + probability distribution + a per-window timeline. "
    "**Research/education only** — not for surveillance, hiring, or clinical use."
)
st.info(
    "Demo model is trained on a **synthetic fixture** so this Space needs no data or GPU. "
    "For real-speech accuracy, train on RAVDESS/CREMA-D (see the repo README)."
)

reg = get_registry()
uploaded = st.file_uploader("Audio clip (wav/flac/mp3/ogg)", type=["wav", "flac", "mp3", "ogg"])
if st.button("Try a sample tone"):
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    tone = (0.6 * np.sin(2 * np.pi * 240 * t) + 0.2 * np.sin(2 * np.pi * 480 * t)).astype(
        np.float32
    )
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    st.session_state["audio_bytes"] = buf.getvalue()

audio_bytes = uploaded.read() if uploaded else st.session_state.get("audio_bytes")

if audio_bytes:
    st.audio(audio_bytes)
    tmp = Path("/tmp/esa_upload.wav")
    tmp.write_bytes(audio_bytes)
    expl = explain_from_registry(tmp, reg)

    left, right = st.columns([1, 1])
    with left:
        st.metric("Predicted emotion", expl.predicted_label, f"{expl.confidence * 100:.1f}% conf")
        st.caption(expl.note)
        st.bar_chart(dict(sorted(expl.probabilities.items(), key=lambda kv: -kv[1])))
    with right:
        st.subheader("Per-window timeline (explainability)")
        st.dataframe(
            [
                {
                    "window": w.index,
                    "span (s)": f"{w.start_sec:.2f}-{w.end_sec:.2f}",
                    "emotion": w.label,
                    "confidence": round(w.confidence, 3),
                }
                for w in expl.windows
            ],
            use_container_width=True,
        )
