# 🎙️ EmotionSense AI

**Production-Ready Speech Emotion Recognition Platform**

An end-to-end SER system that does what most SER repos don't: it benchmarks multiple
models under **honest, speaker-independent, cross-corpus** evaluation, and ships the whole
thing as a versioned API + UI with experiment tracking and containerized reproducibility.

> ⚠️ **Research / education use only.** Not for surveillance, hiring decisions, or clinical
> diagnosis. Emotion recognition from speech is error-prone and culturally biased.

---

## Why this project is different

Most SER repositories stop at a notebook reporting a single, often **inflated**
single-corpus accuracy (speaker-dependent splits leak speaker identity into the test set).
EmotionSense AI is built around the opposite discipline:

| Differentiator | What it means |
|----------------|---------------|
| **Trivial baselines** | Majority + stratified-random rows make every score interpretable (chance = 1/7 for 7 classes). |
| **Speaker-independent k-fold** | No speaker appears in train *and* test. Reported as **mean ± std** across folds, not one noisy split. |
| **Unweighted Accuracy headline** | UA (mean per-class recall) is imbalance-robust — a majority predictor scores at chance, not high. |
| **Cross-corpus generalization** | Train on RAVDESS/TESS, test on held-out **CREMA-D**, projected to the shared 6-class label intersection. |
| **Train/serve feature parity** | The feature spec travels *with* the model; a golden test guarantees serve-time features == train-time features. |
| **Checksum-verified model loading** | Deep weights via safetensors (never pickle); every artifact is SHA-256 verified before load. |

See the full design rationale in [`docs/design/`](docs/design/) and the pre-build review in
[`docs/design/09-architecture-review.md`](docs/design/09-architecture-review.md).

---

## Results (real corpora, speaker-independent)

Validated on **RAVDESS** (1,440 clips) and **CREMA-D** (7,442 clips) on a Kaggle GPU.
Full records: [`docs/results/`](docs/results/). Every number is **speaker-independent** and
reported as mean ± std across 5 folds; baselines confirm the chance floor.

**In-corpus — RAVDESS (7-class):**

| Model | UA (headline) | Accuracy | macro-F1 |
|-------|---------------|----------|----------|
| **distilhubert-svm** (frozen Distil-HuBERT + SVM) | **0.689 ± 0.080** | 0.696 | 0.690 |
| svm-mfcc | 0.42 | 0.44 | 0.42 |
| baseline (chance = 1/7) | 0.143 | — | — |

**In-corpus — CREMA-D (6-class, 91 speakers):** svm-mfcc **UA 0.476 ± 0.017** (chance 0.167).

**Cross-corpus — train RAVDESS → test CREMA-D (6-class):**

| Model | UA | macro-F1 |
|-------|----|----------|
| svm-mfcc | **0.258** | 0.233 |
| distilhubert-svm | 0.212 | 0.120 |

### The finding that matters

Frozen Distil-HuBERT embeddings lift RAVDESS in-corpus UA from **0.42 → 0.69** (+26 pts) —
but its **cross-corpus** UA *drops below* the humble MFCC baseline (0.21 vs 0.26). The
higher-capacity representation captures more corpus-specific structure (mics, actors, the
two fixed RAVDESS sentences), so it transfers *worse*. **A stronger in-corpus model can
generalize worse across corpora — a robustness gap that single-corpus leaderboards hide.**
This is exactly what the cross-corpus protocol is designed to surface.

---

## Quickstart (zero downloads, no GPU)

Everything runs on a **synthetic audio fixture** so you can exercise the entire pipeline —
features → models → benchmark → serving — with no corpora and no GPU.

```bash
# 1. Install (CPU, light)
pip install -e ".[dev]"

# 2. Run the benchmark harness on synthetic data -> leaderboard
python scripts/run_benchmark.py --experiment configs/experiments/smoke_synthetic.yaml

# 3. Train + register a servable model
python scripts/train_and_register.py --model svm --dataset synthetic --promote

# 4. Serve it
pip install -e ".[backend]"
uvicorn emotionsense.backend.main:app --reload
#   -> http://localhost:8000/docs   (OpenAPI)

# 5. The demo UI
pip install -e ".[frontend]"
streamlit run src/emotionsense/frontend/app.py
```

Or bring up the full stack (backend, frontend, postgres, minio, mlflow):

```bash
docker compose -f deployment/compose/docker-compose.yml up --build
```

### Example benchmark output (synthetic fixture)

```
| Model             | Dataset   | Eval | UA            | Accuracy      | macro-F1 |
|-------------------|-----------|------|---------------|---------------|----------|
| logreg-mfcc       | synthetic | cv   | 0.988 ± 0.008 | 0.988 ± 0.008 | 0.988    |
| svm-mfcc          | synthetic | cv   | 0.982 ± 0.015 | 0.982 ± 0.015 | 0.982    |
| baseline-majority | synthetic | cv   | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036    |
| baseline-random   | synthetic | cv   | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036    |
```

Baselines land at **0.143 = 1/7** (chance for 7 classes) — exactly what makes the classical
rows meaningful. On real corpora, swap in the dataset configs (below).

---

## Using real datasets

```bash
python scripts/download_datasets.py            # prints licensed download instructions
# extract RAVDESS/TESS/CREMA-D under data/raw/<name>/, then:
python scripts/run_benchmark.py --experiment configs/experiments/baseline_suite.yaml
```

`baseline_suite.yaml` runs speaker-independent 5-fold CV on RAVDESS **plus a cross-corpus
RAVDESS→CREMA-D generalization test**. Measured results (above) — RAVDESS 7-class UA ≈0.69
with Distil-HuBERT, ≈0.42 with MFCC; cross-corpus drops to ≈0.21–0.26 (reported honestly,
not hidden).

### Transformer models (Distil-HuBERT) on free GPU

Frozen SSL embeddings + a light SVM head (never fine-tuned) run on Colab/Kaggle — see the
ready-to-run [`notebooks/kaggle_emotionsense_validation.ipynb`](notebooks/kaggle_emotionsense_validation.ipynb).
Install the extra locally with `pip install -e ".[transformer]"`.

### Explain a prediction

```bash
python scripts/explain.py --audio path/to/clip.wav        # per-window emotion timeline
```

Splits the clip into windows, scores each independently, and shows *which parts of the audio
drove which emotion* plus the aggregated prediction. See [Explainability](#explainability).

---

## Architecture

```
Client (Streamlit UI / API consumers)
        │  REST
   FastAPI backend ── Inference service ── Model Registry
        │                   │                   │
   PostgreSQL          feature parity        artifacts
        │              (spec on model)           │
   Object storage (MinIO/S3) ◄── Training plane (Colab/Kaggle GPU)
        │                              │
   MLflow tracking ◄──────────── Benchmark harness (k-fold + cross-corpus)
```

The **training plane** (GPU) and **serving plane** (CPU) never call each other — they
communicate only through the DB, object storage, and the registry. That's what makes
"train on Kaggle, serve on a laptop" work. Full diagrams:
[`docs/design/02-TDD.md`](docs/design/02-TDD.md).

---

## Project layout

| Path | What |
|------|------|
| `src/emotionsense/common` | config, logging, errors, canonical label taxonomy |
| `src/emotionsense/datasets` | loaders, label harmonization, speaker-independent k-fold splits |
| `src/emotionsense/ml` | preprocessing, features (MFCC/mel/chroma/SSL), models (baselines/classical/deep) |
| `src/emotionsense/training` | benchmark harness, evaluation (UA/CV mean±std), reporting |
| `src/emotionsense/inference` | serving pipeline, validators, model cache |
| `src/emotionsense/registry` | model registry + checksum-verified artifact I/O |
| `src/emotionsense/backend` | FastAPI app (auth, predict, models, admin, health) |
| `src/emotionsense/frontend` | Streamlit demo |
| `configs/` | declarative YAML: datasets, features, models, experiments |
| `docs/research`, `docs/design` | the full research + design + review trail |

---

## Explainability

SER models are opaque and their confidences are **uncalibrated** — so instead of a single
score, EmotionSense exposes *where in the clip* each emotion came from. The inference path
already splits a clip into fixed windows and averages their probabilities; the explainability
layer surfaces those **per-window predictions** as a timeline:

```bash
python scripts/explain.py --audio clip.wav
```

```
Prediction: angry  (confidence 0.71)
Clip 3.20s split into 2 window(s)

  window |     span (s) |   emotion | conf
----------------------------------------------
       0 |  0.00-3.00   |     angry | 0.74
       1 |  3.00-3.20   |       sad | 0.55
```

It reuses the *same* preprocessing + feature extractor as training (ADR-3), so explanations
are consistent with predictions, and it prints an explicit "confidences are uncalibrated"
caveat. The same view powers the live demo's timeline panel
([`deployment/hf_space/app.py`](deployment/hf_space/app.py)).

---

## Deployment

- **Live demo (Hugging Face Spaces):** self-contained Streamlit app in
  [`deployment/hf_space/`](deployment/hf_space/) — no DB/GPU needed. See
  [`docs/DEPLOY.md`](docs/DEPLOY.md).
- **Full local stack:** `docker compose -f deployment/compose/docker-compose.yml up --build`
  (backend + frontend + postgres + minio + mlflow).

---

## Development

```bash
make lint        # ruff + black
make type        # mypy strict
make test        # unit + integration
make cov         # coverage
make benchmark   # run the harness
```

CI runs lint + type + tests + a synthetic smoke benchmark on every push
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## License

MIT (code). Datasets retain their own licenses — see each `configs/datasets/*.yaml`.
