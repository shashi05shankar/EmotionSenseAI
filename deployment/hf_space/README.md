---
title: EmotionSense AI
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.33.0
app_file: app.py
pinned: false
license: mit
---

# EmotionSense AI — live demo

Upload a short speech clip and see the predicted emotion, the full probability
distribution, and a **per-window explainability timeline** (which parts of the clip drove
which emotion).

This Space is self-contained: it trains a small SVM on a synthetic fixture at first load, so
it needs no data or GPU. For real-speech accuracy, train on RAVDESS/CREMA-D — see the
[project repo](https://github.com/shashi05shankar/EmotionSenseAI).

> Research / education use only. Not for surveillance, hiring, or clinical use.
