# Validation Results (real corpora)

Speaker-independent, mean ± std over 5 folds. Run on Kaggle GPU, 2026-07-27, from the frozen
`v1.0.0` code. Full machine-readable records: `rc1-validation.json`, `rc1-cremad.json`,
`rc1-transformer.json` (produced by `scripts/run_benchmark.py`).

## RAVDESS — in-corpus (7-class, 1,440 clips → 1,248 after dropping `calm`)

| Model | UA | Accuracy | macro-F1 | weighted-F1 |
|-------|----|----------|----------|-------------|
| distilhubert-svm (frozen Distil-HuBERT + SVM) | **0.689 ± 0.080** | 0.696 | 0.690 | 0.695 |
| svm-mfcc | 0.416–0.426 | 0.44 | 0.42 | 0.44 |
| rf-mfcc | 0.410 | 0.417 | 0.400 | 0.406 |
| logreg-mfcc | 0.410 | 0.409 | 0.400 | 0.402 |
| baseline-majority / random | 0.143 (= 1/7) | 0.154 | 0.038 | 0.041 |

## CREMA-D — in-corpus (6-class, 7,442 clips, 91 speakers)

| Model | UA | Accuracy | macro-F1 |
|-------|----|----------|----------|
| svm-mfcc | **0.476 ± 0.017** | 0.477 | 0.474 |
| rf-mfcc | 0.466 | 0.466 | 0.452 |
| logreg-mfcc | 0.457 | 0.457 | 0.451 |
| baseline-majority / random | 0.167 (= 1/6) | 0.171 | 0.049 |

## Cross-corpus — train RAVDESS → test CREMA-D (6-class intersection)

| Model | UA | macro-F1 |
|-------|----|----------|
| svm-mfcc | **0.258** | 0.233 |
| rf-mfcc | 0.238 | 0.169 |
| logreg-mfcc | 0.209 | 0.148 |
| distilhubert-svm | 0.212 | 0.120 |

## Headline finding

Frozen Distil-HuBERT lifts RAVDESS in-corpus UA **0.42 → 0.69 (+26 pts)** — but its
**cross-corpus** UA (0.21) *falls below* the MFCC baseline (0.26). The stronger representation
captures more corpus-specific structure (mics, actors, RAVDESS's two fixed sentences) and
transfers worse. **In-corpus SOTA ≠ cross-corpus robustness** — the gap single-corpus
leaderboards hide.

*Reproducibility note:* `svm-mfcc` varies ~±1% between runs because `SVC(probability=True)`
Platt calibration has no fixed `random_state` (fixed in `v1.0.0`: add `random_state: 42` to
the SVM configs). Baselines and the headline conclusions are stable.
