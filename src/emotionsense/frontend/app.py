"""EmotionSense AI — Streamlit demo UI.

Run: streamlit run src/emotionsense/frontend/app.py

Pages:
* Predict — upload a clip, see the emotion + probability distribution + waveform.
* Leaderboard — the benchmark results (in-corpus CV vs cross-corpus) with std.

Ethics note is shown prominently: research/education only.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st

from emotionsense.frontend import api_client

st.set_page_config(page_title="EmotionSense AI", page_icon="🎙️", layout="wide")

st.title("🎙️ EmotionSense AI")
st.caption(
    "Production-ready Speech Emotion Recognition. "
    "**Research/education use only — not for surveillance, hiring, or clinical diagnosis.**"
)

tab_predict, tab_board = st.tabs(["🔮 Predict", "📊 Leaderboard"])


with tab_predict:
    st.subheader("Predict emotion from speech")
    col_l, col_r = st.columns([1, 1])
    with col_l:
        uploaded = st.file_uploader(
            "Upload a short clip (wav/flac/mp3/ogg)", type=["wav", "flac", "mp3", "ogg"]
        )
        use_sample = st.button("Try a sample clip")
        model_choice = None
        try:
            models = api_client.list_models()
            refs = [m["ref"] for m in models]
            if refs:
                model_choice = st.selectbox("Model", ["(default)", *refs])
                model_choice = None if model_choice == "(default)" else model_choice
        except Exception:
            st.info("Backend not reachable — start the API to enable predictions.")

    audio_bytes: bytes | None = None
    filename = "audio.wav"
    if uploaded is not None:
        audio_bytes = uploaded.read()
        filename = uploaded.name
    elif use_sample:
        sr = 16000
        t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
        y = (0.6 * np.sin(2 * np.pi * 240 * t) + 0.2 * np.sin(2 * np.pi * 480 * t)).astype(
            np.float32
        )
        buf = io.BytesIO()
        import soundfile as sf

        sf.write(buf, y, sr, format="WAV")
        audio_bytes = buf.getvalue()
        filename = "sample_angry.wav"

    if audio_bytes:
        with col_l:
            st.audio(audio_bytes)
        try:
            result = api_client.predict(audio_bytes, filename, model=model_choice)
            with col_r:
                st.metric(
                    "Predicted emotion",
                    result["predicted_label"],
                    f"{result['confidence']*100:.1f}% conf",
                )
                st.caption(
                    f"model {result['model']['name']}:{result['model']['version']} · "
                    f"{result['latency_ms']} ms · confidences are uncalibrated"
                )
                probs = dict(sorted(result["probabilities"].items(), key=lambda kv: -kv[1]))
                st.bar_chart(probs)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


with tab_board:
    st.subheader("Benchmark leaderboard")
    st.caption(
        "UA = Unweighted Accuracy (imbalance-robust headline). ± = std across speaker-"
        "independent CV folds. Cross-corpus rows measure generalization to an unseen corpus."
    )
    report = Path("experiments/reports")
    files = sorted(report.glob("*.json"))
    if not files:
        st.info("No leaderboard yet. Run: `python scripts/run_benchmark.py`")
    else:
        chosen = st.selectbox("Experiment", [f.stem for f in files])
        rows = json.loads((report / f"{chosen}.json").read_text())
        st.dataframe(
            [
                {
                    "Model": r["model"],
                    "Dataset": r["dataset"],
                    "Eval": r["eval_kind"],
                    "UA": f"{r['ua_mean']:.3f} ± {r['ua_std']:.3f}",
                    "Accuracy": f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}",
                    "macro-F1": f"{r['macro_f1_mean']:.3f}",
                    "Folds": r["n_folds"],
                }
                for r in rows
            ],
            use_container_width=True,
        )
