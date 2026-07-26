# Phase 1–2: Market Research & Comparison Sheet

**Project:** EmotionSense AI — Production-Ready Speech Emotion Recognition Platform
**Date:** 2026-07-26
**Author:** research pass (Claude Code)

> Purpose: survey the existing SER landscape (repos + literature), record what each
> project actually does, and expose where the field is thin. Feeds directly into
> `03-gap-analysis.md` and the v2.0 design.

---

## 1. Repositories surveyed

Nine reference points below: the 5 repos named in the roadmap, plus the modern
transformer/deployment references that represent where the field has moved since
2020. The older repos define the *classical* baseline; the HuggingFace/SpeechBrain
side defines the *transformer* frontier.

### Comparison sheet

| # | Repository | Dataset(s) | Models | Reported accuracy | API | UI | Docker | Last activity | Key weakness |
|---|-----------|-----------|--------|-------------------|-----|----|--------|---------------|--------------|
| 1 | [x4nth055/emotion-recognition-using-speech](https://github.com/x4nth055/emotion-recognition-using-speech) | RAVDESS, TESS, EMO-DB, custom | SVC, RF, GBoost, KNN, MLP, Bagging, LSTM | 89.6% (3-cls MLP); 93.5% (5-cls RF); 77.2% (5-cls LSTM) | ❌ | ❌ | ❌ | ~2021 (stale) | No deployment; manual ffmpeg; class imbalance across 9 labels |
| 2 | [Renovamen/Speech-Emotion-Recognition](https://github.com/Renovamen/Speech-Emotion-Recognition) | RAVDESS, SAVEE, EMO-DB, CASIA | SVM, MLP, CNN, LSTM | "~80%" (vague) | ❌ | ❌ | ❌ | low | Small datasets (500–1500); needs OpenSMILE; train-from-scratch only |
| 3 | [yeyupiaoling/SpeechEmotionRecognition-Pytorch](https://github.com/yeyupiaoling/SpeechEmotionRecognition-Pytorch) | RAVDESS + 25k custom | BiLSTM, BaseModel, **Emotion2Vec** | 85.3% (RAVDESS); 92.9% (custom, BaseModel) | ❌ (CLI `infer.py`) | ❌ | ❌ | 2024 (active) | Emotion2Vec extraction slow, single-threaded; no serving layer |
| 4 | [Demfier/multimodal-speech-emotion-recognition](https://github.com/Demfier/multimodal-speech-emotion-recognition) | IEMOCAP | LogReg, SVM, RF, XGBoost, NB, MLP, LSTM (audio+text) | 75.3% (4-cls, audio+text MDRE); 56.6% audio-only | ❌ | ❌ | ❌ | 2019 (research paper) | Research code; "needs refactoring"; hard-coded paths |
| 5 | [hkveeranki/speech-emotion-recognition](https://github.com/hkveeranki/speech-emotion-recognition) | EMO-DB | SVM, RF, NN, CNN, LSTM | not reported | ❌ | ❌ | ❌ | stale | **Python 2.x** (EOL); single dataset; no benchmarks |
| 6 | [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | IEMOCAP | wav2vec2 fine-tuned | ~78% (4-cls) | inference API (HF) | ❌ | ❌ | active | 4-class only; IEMOCAP licence-gated |
| 7 | [ehcalabres/wav2vec2-lg-xlsr-en-SER](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) | RAVDESS | wav2vec2-XLSR fine-tuned | ~82% (8-cls) | HF inference | ❌ | ❌ | active | Single-corpus; no app around it |
| 8 | [audeering/w2v2-how-to](https://github.com/audeering/w2v2-how-to) (model: wav2vec2-large-robust-…-msp-dim) | MSP-Podcast | wav2vec2 (dimensional: arousal/valence/dominance) | dimensional (CCC), not class acc | model + how-to | ❌ | ❌ | active | Dimensional not categorical; no full product |
| 9 | [firdhokk/SER-with-whisper-large-v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) | multi | Whisper-large-v3 fine-tuned | high, heavyweight | HF inference | ❌ | ❌ | active | Whisper-large is large/slow for real-time |

Legend: ❌ = absent · "active" = commits within ~18 months.

---

## 2. What the table tells us

**Pattern 1 — The classical repos (1,2,5) are educational, not products.**
They stop at "train a model, print accuracy in a notebook/CLI." No serving, no UI,
no containerization, no experiment tracking. Repo 5 is on Python 2 (dead). These
are the ones a recruiter would call "a tutorial project."

**Pattern 2 — The modern repos (3,6,7,8,9) have better *models* but still no *product*.**
Transformer embeddings (Emotion2Vec, wav2vec2, Whisper) push accuracy to the mid-80s
to low-90s, but they ship as a `pip install` + `infer.py`, or a HuggingFace weight
with an inference widget. None bundle FastAPI + Streamlit + DB + monitoring.

**Pattern 3 — Almost everyone benchmarks on ONE corpus.**
RAVDESS or IEMOCAP or EMO-DB, rarely cross-corpus. This matters: single-corpus
accuracy is inflated because the model memorizes speakers/recording conditions.
Cross-corpus generalization is the honest, and largely unaddressed, hard problem.

**Pattern 4 — Accuracy numbers are not comparable across repos.**
Different class counts (3 vs 4 vs 8), different splits (speaker-dependent vs
speaker-independent), different metrics (weighted vs unweighted accuracy). A repo
claiming 93% (5-class RF, speaker-dependent) is *not* better than one claiming 78%
(4-class wav2vec2, speaker-independent). **This is itself a gap we can exploit:
a rigorous, apples-to-apples benchmark harness.**

---

## 3. The "product" whitespace

Cross-referencing all nine: the thing that is consistently missing is not a *model*,
it's the **engineering wrapper** — a real API, a usable UI, reproducible benchmarking,
and honest cross-corpus evaluation. That is exactly the surface where EmotionSense AI
can differentiate without needing to beat SOTA on raw accuracy (which requires
big-GPU pretraining we don't have).

→ Detailed opportunity list in [`03-gap-analysis.md`](03-gap-analysis.md).

---

## Sources

- [x4nth055/emotion-recognition-using-speech](https://github.com/x4nth055/emotion-recognition-using-speech)
- [Renovamen/Speech-Emotion-Recognition](https://github.com/Renovamen/Speech-Emotion-Recognition)
- [yeyupiaoling/SpeechEmotionRecognition-Pytorch](https://github.com/yeyupiaoling/SpeechEmotionRecognition-Pytorch)
- [Demfier/multimodal-speech-emotion-recognition](https://github.com/Demfier/multimodal-speech-emotion-recognition)
- [hkveeranki/speech-emotion-recognition](https://github.com/hkveeranki/speech-emotion-recognition)
- [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
- [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- [audeering/w2v2-how-to](https://github.com/audeering/w2v2-how-to)
- [firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3)
