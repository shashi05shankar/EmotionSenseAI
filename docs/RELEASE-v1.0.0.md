# EmotionSense AI v1.0.0

First stable release: a production-grade Speech Emotion Recognition platform with honest,
speaker-independent + cross-corpus benchmarking, a served API + UI, explainability, and a
validated real-data result set.

## Highlights

- **Validated on real corpora** (RAVDESS 1,440 + CREMA-D 7,442, speaker-independent 5-fold):
  - RAVDESS in-corpus UA: **0.689 ± 0.080** (frozen Distil-HuBERT + SVM) vs ~0.42 (MFCC).
  - CREMA-D in-corpus UA: **0.476 ± 0.017** (91 speakers).
  - Cross-corpus RAVDESS→CREMA-D UA: 0.258 (MFCC) / 0.212 (Distil-HuBERT) — the honest
    generalization-gap finding. Full record: [`docs/results/`](results/README.md).
- **Transformer Phase 1:** frozen Distil-HuBERT embeddings + SVM head (GPU-aware, never
  fine-tuned), benchmarked under the identical protocol as the classical models.
- **Explainability:** per-window emotion timeline (`scripts/explain.py`) — which parts of a
  clip drove which emotion — with an explicit uncalibrated-confidence caveat.
- **Deployable:** self-contained Hugging Face Spaces app (`deployment/hf_space/`) +
  Docker Compose stack. See [`docs/DEPLOY.md`](DEPLOY.md).
- **Quality:** ruff/black/mypy + **40 tests, 84% coverage** (gate enforced in CI).
- **Reproducibility fix:** pinned `SVC(probability=True)` calibration via `random_state` in
  the SVM configs (removed ~±1% run-to-run drift).

## Metrics reported per model

Accuracy · Weighted-F1 · macro-F1 · Unweighted Accuracy (±std) · confusion matrix ·
extract/train/infer timings — for baselines, classical, and transformer alike.

## Security & integrity

- Checksum-verified (SHA-256) artifact loading; safetensors for deep weights (never pickle).
- JWT auth with env-provided admin credentials (no secrets in code).
- Speaker-independent split enforced at the pipeline level (guardrail-tested).

## Known limitations (tracked)

- Cross-corpus performance is weak (~0.21–0.26 UA) — the real state of SER generalization;
  next: multi-corpus training / domain adaptation.
- Only Distil-HuBERT so far (HuBERT-base / wav2vec2 are future breadth).
- DB-backed persistence is designed; the runnable path uses a filesystem registry.
- Confidences are uncalibrated (temperature scaling is a quick add).

## Provenance

Full research → design → review → implementation → audit → validation trail in
[`docs/research/`](research/) and [`docs/design/`](design/).
