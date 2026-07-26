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
RAVDESS→CREMA-D generalization test**. Realistic speaker-independent targets: RAVDESS
8-class ≥83%, cross-corpus expect a substantial drop (reported honestly, not hidden).

### Transformer models (Distil-HuBERT) on free GPU

Frozen SSL embeddings + a light head, or a Distil-HuBERT fine-tune, run on Colab/Kaggle —
see [`notebooks/`](notebooks/). Install the extra locally with `pip install -e ".[transformer]"`.

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
