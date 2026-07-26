# Phase 3: Gap Analysis

**Project:** EmotionSense AI
**Date:** 2026-07-26

> Derived from [`01-landscape-and-comparison.md`](01-landscape-and-comparison.md) and
> [`02-sota-datasets-reference.md`](02-sota-datasets-reference.md). Each gap is tagged
> with how hard it is for us to fill given **Colab/Kaggle free-GPU** constraints, and
> whether it's **Core** (must-have for a credible portfolio piece) or **Stretch**.

---

## A. Model gaps

| Gap | Seen in existing repos? | Our opportunity | Effort | Tier |
|-----|------------------------|-----------------|--------|------|
| Modern transformer backbone (Distil-HuBERT / wav2vec2 / emotion2vec) | Only in newer repos (3,6,7); classical repos lack it | Include a transformer track — the single biggest credibility signal | Med (fine-tune fits free GPU) | **Core** |
| Frozen-embedding + light-head baseline | Rare as an explicit, documented baseline | Cheap, fast, honest baseline; shows you understand the SSL-features tradeoff | Low | **Core** |
| Ensemble / model stacking | Almost none | Blend classical + transformer for a measurable bump | Med | Stretch |
| Dimensional emotion (arousal/valence/dominance) | Only audeering | Add regression head as a second output mode | Med | Stretch |

## B. Evaluation & benchmarking gaps  ← *strongest differentiator*

| Gap | Problem in the field | Our opportunity | Effort | Tier |
|-----|----------------------|-----------------|--------|------|
| **Apples-to-apples benchmark harness** | Repos report incomparable numbers (3 vs 8 classes, speaker-dependent vs -independent, WA vs UA) | A single harness that runs every model under identical splits/metrics and prints one leaderboard | Med | **Core** |
| **Speaker-independent splits** | Many repos leak speakers across train/test → inflated accuracy | Enforce speaker-independent CV; report the honest (lower) number | Low | **Core** |
| **Cross-corpus generalization** | Nearly nobody tests train-on-A → test-on-B | Train on RAVDESS+TESS, hold out CREMA-D; report the drop | Low | **Core** |
| Confusion-matrix / per-emotion analysis | Sparse | Standard per-class F1 + confusion matrices in reports | Low | **Core** |

> This cluster is where you can genuinely *beat* the existing repos — not on raw
> accuracy, but on **rigor and honesty**. It's also cheap (no big GPU needed) and it
> reads as senior-level ML maturity on a resume.

## C. Engineering gaps

| Gap | Seen in repos? | Our opportunity | Effort | Tier |
|-----|---------------|-----------------|--------|------|
| REST API (FastAPI) | None of 1–5; HF widgets only | `/predict` endpoint, model-version selectable | Low-Med | **Core** |
| Interactive UI (Streamlit) | None | Upload/record audio → emotion + probabilities + waveform | Med | **Core** |
| Experiment tracking (MLflow / W&B) | None | Track every run's config, metrics, artifacts — reproducibility | Low | **Core** |
| Dockerization | None | One-command reproducible env | Low | **Core** |
| CI/CD (GitHub Actions) | None | Lint + test on push; build image | Low | Stretch |
| Structured logging + monitoring | None | Request logs, latency metrics, prediction logging | Med | Stretch |
| Authentication / user management | None | API keys or simple auth on the API | Med | Stretch |
| Model registry / versioning | None (repos ship one model) | Named, swappable models behind the API | Low-Med | **Core** |

## D. Product / data gaps

| Gap | Our opportunity | Effort | Tier |
|-----|-----------------|--------|------|
| Multi-dataset support with **label harmonization** | Unify RAVDESS/TESS/SAVEE to one taxonomy — most repos are single-corpus | Med | **Core** |
| Real-time / mic input | Streamlit mic capture → live prediction | Med | Stretch |
| Explainability (which frames/features drove the prediction) | Saliency over the spectrogram / attention weights | High | Stretch |
| Robustness to noise | Data augmentation (noise, pitch, time-shift) + report robustness | Med | Stretch |

---

## Prioritized "Core deliverable" (what guarantees a demoable, credible project)

If you build **only** these, you already beat every repo in the survey on
completeness:

1. **Data pipeline** — download + unify RAVDESS+TESS+SAVEE, speaker-independent split, augmentation.
2. **≥6 models under one harness** — 3 classical (SVM, RF, MLP on MFCC/mel/chroma),
   1 deep (CNN or BiLSTM), 2 transformer (emotion2vec frozen head + fine-tuned Distil-HuBERT).
3. **Rigorous benchmark** — identical splits/metrics, confusion matrices, **+ cross-corpus CREMA-D test**.
4. **FastAPI `/predict`** with model-registry (swap models by name).
5. **Streamlit UI** — upload/record → emotion + probabilities + waveform/spectrogram.
6. **MLflow tracking + Docker + README/architecture diagram + live demo.**

## Stretch (add once core is green)

Ensembles · dimensional emotion · auth · monitoring dashboard · CI/CD · real-time
mic · explainability · IEMOCAP + multimodal (MELD).

---

## The one-sentence positioning (for the README / resume)

> *"Where existing SER repos stop at a notebook with an inflated single-corpus
> accuracy number, EmotionSense AI benchmarks 6+ models under identical
> speaker-independent splits, honestly reports cross-corpus generalization, and ships
> the whole thing as a versioned API + UI with experiment tracking and Docker."*

That is a defensible, non-copycat story — and every claim in it is achievable on free
GPU compute.
