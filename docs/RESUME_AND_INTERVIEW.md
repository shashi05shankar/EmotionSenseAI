# Resume Bullets & Interview Talking Points — EmotionSense AI

Grounded in the **measured** v1.0.0 results. Pick 3–4 bullets; keep the numbers exact.

---

## Resume bullets (pick the strongest 3–4)

**Lead with rigor + a quantified result:**

- Built an end-to-end **Speech Emotion Recognition** platform benchmarking 6+ models
  (baselines → SVM/RF/LogReg → **frozen Distil-HuBERT + SVM**) under a **speaker-independent
  5-fold** protocol; frozen Distil-HuBERT embeddings raised RAVDESS unweighted accuracy from
  **0.42 → 0.69 (+26 pts)** over MFCC.

- Designed a **cross-corpus evaluation** (train RAVDESS → test CREMA-D, 6-class intersection)
  that surfaced a non-obvious finding: the strongest in-corpus model **generalized worse**
  across corpora (UA 0.21 vs the MFCC baseline's 0.26) — quantifying a domain-robustness gap
  that single-corpus leaderboards hide.

- Engineered a reproducible benchmark harness reporting **UA / accuracy / macro- & weighted-F1
  / confusion matrix / train & inference time**, with **trivial baselines pinned at chance**
  (1/7, 1/6) so every score is interpretable.

- Shipped it as a product: **FastAPI** service (JWT auth, checksum-verified model registry),
  **Streamlit** UI with a **per-window explainability** timeline, **Docker Compose** stack,
  and **GitHub Actions CI** (ruff/black/mypy + tests, **83% coverage** gate).

- Enforced **train/serve feature parity** (the feature spec travels with the model, guarded by
  a golden test) and a **security-hardened** loader (safetensors + SHA-256, never pickle).

**One-line version (for a dense resume):**
> *EmotionSense AI — production SER platform: speaker-independent + cross-corpus benchmark of
> 6+ models (Distil-HuBERT lifted RAVDESS UA 0.42→0.69), served via FastAPI/Streamlit with
> explainability, Docker, and CI (83% coverage).*

---

## Interview talking points

### 30-second pitch
"Most SER repos report one inflated single-corpus number. I built a platform around honest
evaluation: speaker-independent k-fold with baselines, plus a cross-corpus test. Distil-HuBERT
embeddings got me to 69% UA on RAVDESS — but the interesting result is that it generalized
*worse* cross-corpus than plain MFCC, which is the kind of thing you only catch if you test
across corpora. It ships as an API + UI with explainability, Docker, and CI."

### "Why is speaker-independent evaluation a big deal?"
If the same speaker appears in train and test, the model memorizes the voice, not the emotion —
accuracy looks great and collapses in production. I group folds by speaker so no speaker leaks;
a unit test asserts it. My baselines land at exactly 1/7 and 1/6, which is how I know the
harness isn't leaking.

### "Why Unweighted Accuracy instead of accuracy?"
Classes are imbalanced (RAVDESS neutral has half the samples; CREMA-D is skewed). Plain accuracy
rewards predicting the majority class. UA = mean per-class recall, so a majority predictor scores
at chance, not high. I report both, plus macro- and weighted-F1.

### "Walk me through the cross-corpus finding."
Distil-HuBERT is 1536-dim after mean+std pooling — high capacity. On RAVDESS it captured emotion
*and* corpus-specific structure (studio mic, 24 actors, two fixed sentences). Trained an SVM on
that, tested on CREMA-D (91 different speakers, different sentences) → the boundaries collapsed;
it predicted `fear`/`sad` for almost everything (macro-F1 0.12). MFCC is lower-capacity and
transferred slightly better. Lesson: representation strength and transfer are different axes.
The next step would be multi-corpus training or a light domain-adaptation layer.

### "How did you keep it reproducible?"
Config-driven experiments (YAML), fixed seeds, versioned data, and CV reported as mean ± std.
I actually caught a reproducibility bug during validation: `SVC(probability=True)` does internal
Platt calibration with no `random_state`, so SVM runs drifted ±1%. Fixed it by pinning
`random_state` in the config.

### "How is it 'production', not a notebook?"
Layered FastAPI (router → service → repository), a model registry that's the serving source of
truth (separate from experiment tracking), checksum-verified artifact loading, structured logging
with correlation IDs, Docker Compose for the full stack, and CI with an 83% coverage gate. The
training plane (GPU/Kaggle) and serving plane (CPU) only communicate through the registry +
object storage — that's what lets me train on Kaggle and serve on a laptop.

### "What's the explainability?"
SER confidences are uncalibrated, so a single softmax number is misleading. Instead I split the
clip into windows, score each, and show a per-window timeline — *which parts of the audio drove
which emotion* — reusing the exact training feature path so explanations match predictions.

### "Biggest limitations / what you'd do next" (shows maturity)
- Cross-corpus performance is weak (~0.21–0.26 UA) — the honest state of SER generalization;
  I'd tackle it with multi-corpus training + domain adaptation.
- Only Distil-HuBERT so far; HuBERT-base / wav2vec2 are the next breadth step.
- The DB-backed persistence layer is designed but the runnable path uses a filesystem registry.
- Confidences are uncalibrated (temperature scaling is a quick add).

### Numbers to have memorized
| Metric | Value |
|--------|-------|
| RAVDESS UA — Distil-HuBERT+SVM | **0.689 ± 0.080** |
| RAVDESS UA — MFCC+SVM | ~0.42 |
| CREMA-D UA — MFCC+SVM (91 spk) | **0.476 ± 0.017** |
| Cross-corpus UA — best (MFCC) | 0.258 |
| Chance (7-cls / 6-cls) | 0.143 / 0.167 |
| Test coverage | 83% |
| Datasets / clips | RAVDESS 1,440 · CREMA-D 7,442 |
