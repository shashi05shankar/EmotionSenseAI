# Model Card — `<name>:<version>`

> Fill one of these per shipped model (responsible-AI hygiene). Auto-populatable from the
> registry record + MLflow run.

## Overview
- **Model family:** e.g. SVM (RBF) on MFCC mean+std / Distil-HuBERT frozen embeddings + SVM head
- **Version:** vN
- **Trained by / date:**
- **Registry ref:** `<name>:<version>`  ·  **Artifact SHA-256:** `<hash>`

## Intended use
- **In scope:** research, education, demos, benchmarking baselines.
- **Out of scope (do NOT use for):** surveillance, hiring/HR decisions, clinical or
  mental-health diagnosis, any high-stakes decision about a person.

## Training data
- **Corpus / version:** e.g. RAVDESS (speech only) + TESS, unified to the 7-class canonical taxonomy.
- **Split:** speaker-independent k-fold (seed, folds). No speaker appears in train and test.
- **Class balance:** report per-class counts; imbalance handled via class-weighted training.

## Evaluation
| Metric | In-corpus (mean ± std) | Cross-corpus (CREMA-D) |
|--------|------------------------|------------------------|
| Unweighted Accuracy (headline) | | |
| Accuracy | | |
| macro-F1 | | |

- **Confusion matrix:** link to `experiments/reports/…`.
- **Per-class F1:** …

## Limitations & caveats
- **Lexical leakage:** RAVDESS uses 2 fixed sentences; TESS single words — models may
  exploit content, not just affect. Cross-corpus numbers are the honest generalization signal.
- **Speaker/demographic coverage:** TESS is 2 female speakers; acted emotions ≠ natural emotion.
- **Confidence calibration:** softmax confidences are **uncalibrated** — treat as ranking, not probability.
- **Language:** English-first; performance on other languages is untested.

## Ethical considerations
Emotion inference from voice is scientifically contested and culturally biased. Predictions
must never be used to make consequential judgments about individuals.
