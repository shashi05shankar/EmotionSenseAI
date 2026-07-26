# Folder Structure — Production Monorepo

**Product:** EmotionSense AI
**Version:** 2.0 (design)
**Date:** 2026-07-26
**Related:** [`02-TDD.md`](02-TDD.md), [`06-coding-standards.md`](06-coding-standards.md)

> Single repository, modular Python packages under `src/`, plus top-level dirs for
> configs, deployment, tests, docs, and scripts. The layout enforces the train/serve
> separation (ADR-1) and config-over-code principle from the TDD.

---

## 1. Top-Level Layout

```text
EmotionSenseAI/
├── README.md                     # overview, quickstart, demo link, model card summary
├── LICENSE                       # MIT (code) — note dataset licences separately
├── pyproject.toml                # single source of build + tool config (ruff, black, mypy, pytest)
├── uv.lock / poetry.lock         # pinned deps for reproducibility (NFR-3)
├── .env.template                 # documented env vars; real .env is gitignored
├── .gitignore
├── .pre-commit-config.yaml       # ruff, black, mypy, detect-secrets hooks
├── Makefile                      # dev entrypoints: make lint/test/up/train/benchmark
│
├── src/                          # all first-party packages (installable, importable)
│   └── emotionsense/
│       ├── __init__.py
│       ├── common/               # shared: config, logging, errors, storage, schemas
│       ├── datasets/             # ingest, harmonize, split  (FR-D*)
│       ├── ml/                   # features + model definitions (shared by train & infer)
│       ├── training/             # trainers, benchmark harness (GPU/Colab/Kaggle)
│       ├── inference/            # serving-time predict pipeline (CPU)
│       ├── registry/             # model registry access (DB + object storage)
│       ├── backend/              # FastAPI app (routers, services, repositories)
│       └── frontend/             # Streamlit app
│
├── configs/                      # declarative YAML — the "config over code" surface
│   ├── datasets/                 # one file per corpus (ravdess.yaml, crema_d.yaml, ...)
│   ├── features/                 # mfcc.yaml, mel.yaml, distilhubert_embed.yaml
│   ├── models/                   # svm.yaml, xgboost.yaml, bilstm.yaml, distilhubert.yaml
│   ├── experiments/              # composed runs (baseline_suite.yaml, cross_corpus.yaml)
│   └── app/                      # api.yaml, inference.yaml, logging.yaml (env-overridable)
│
├── notebooks/                    # Colab/Kaggle training notebooks (thin: call src/)
│   ├── 01_download_data.ipynb
│   ├── 02_train_classical.ipynb
│   ├── 03_train_transformer_kaggle.ipynb   # checkpoint/resume aware
│   └── 04_benchmark_leaderboard.ipynb
│
├── datasets/                     # LOCAL data root (gitignored) — real data never committed
│   ├── raw/                      # downloaded corpora
│   ├── normalized/               # resampled/mono/trimmed
│   ├── features/                 # cached features/embeddings
│   └── manifests/                # split manifests (also mirrored to object storage)
│
├── experiments/                  # local experiment outputs (gitignored)
│   ├── artifacts/                # serialized models before registry upload
│   ├── checkpoints/              # resume points
│   └── reports/                  # generated confusion matrices, leaderboard exports
│
├── deployment/                   # infra-as-config
│   ├── docker/                   # Dockerfiles + entrypoints (see §3)
│   ├── compose/                  # docker-compose.yml + override files
│   ├── github-actions/           # CI/CD workflow yml (symlinked to .github/workflows)
│   ├── grafana/                  # dashboards (json) + provisioning
│   └── prometheus/               # scrape config
│
├── migrations/                   # Alembic env + versioned migration scripts
│   ├── env.py
│   └── versions/
│
├── tests/                        # mirrors src/ (see §4)
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── conftest.py
│
├── scripts/                      # operational CLIs (thin wrappers over src/)
│   ├── download_datasets.py
│   ├── build_splits.py
│   ├── run_benchmark.py
│   ├── seed_db.py
│   └── promote_model.py
│
└── docs/
    ├── research/                 # phase-1 research (already written)
    ├── design/                   # this design set
    ├── api/                      # exported OpenAPI + usage guide
    ├── architecture/             # rendered diagrams
    └── model_cards/              # per-model cards (metrics, intended use, limitations)
```

---

## 2. Package Internals (`src/emotionsense/`)

### `common/` — cross-cutting foundations
```text
common/
├── config.py          # Pydantic Settings; loads YAML + env
├── logging.py         # structlog setup; correlation-id context
├── errors.py          # typed exceptions → API error envelope
├── storage.py         # S3/MinIO client wrapper (put/get/presign)
├── db.py              # SQLAlchemy engine/session factory
├── schemas.py         # shared Pydantic DTOs (Prediction, ModelInfo, ...)
└── constants.py       # canonical label taxonomy, sample rate, etc.
```

### `datasets/` — data pipeline (FR-D1..D5)
```text
datasets/
├── base.py            # DatasetLoader interface
├── loaders/           # ravdess.py, tess.py, savee.py, crema_d.py, aesdd.py, iemocap.py
├── harmonize.py       # native → canonical label mapping
├── split.py           # speaker-INDEPENDENT split builder (ADR-4)
├── augment.py         # noise / pitch / time-stretch
└── manifest.py        # read/write split manifests
```

### `ml/` — features + models (shared by training AND inference → parity, ADR-3)
```text
ml/
├── features/
│   ├── base.py        # FeatureExtractor interface
│   ├── spectral.py    # MFCC, mel, chroma (librosa)
│   └── ssl.py         # HuBERT/wav2vec2/emotion2vec embedding extractors
├── models/
│   ├── base.py        # Model interface: fit/predict/save/load/feature_spec
│   ├── classical.py   # SVM, RandomForest, XGBoost
│   ├── deep.py        # CNN, LSTM, BiLSTM (PyTorch)
│   └── transformer.py # Distil-HuBERT fine-tune + frozen-embedding heads
└── preprocess.py      # resample/mono/trim/normalize (used at train AND serve)
```

### `training/` — GPU plane (Colab/Kaggle)
```text
training/
├── trainer.py         # orchestrator: config → data → fit → eval → log → register
├── callbacks.py       # checkpointing (resume), early stopping
├── evaluate.py        # metrics: acc, WA, UA, macro-F1, confusion matrix, ROC-AUC
├── benchmark.py       # unified harness → leaderboard (FR-E1..E4)
└── mlflow_logger.py   # experiment tracking integration
```

### `inference/` — CPU serving plane
```text
inference/
├── service.py         # InferenceService: validate→preprocess→features→predict
├── client.py          # InferenceClient interface (in-proc now, HTTP-ready)
├── model_cache.py     # LRU cache of loaded models keyed by version
└── validators.py      # file type/size/duration guards
```

### `registry/`
```text
registry/
├── registry.py        # register/resolve/promote/retire model versions
└── artifact.py        # (de)serialize artifacts + metadata to object storage
```

### `backend/` — FastAPI (layered: router → service → repository)
```text
backend/
├── main.py            # app factory, middleware wiring
├── deps.py            # dependency-injection (db session, current user, settings)
├── middleware/        # auth, correlation-id, rate-limit, metrics
├── routers/           # auth.py, predict.py, datasets.py, training.py,
│                      #   experiments.py, models.py, admin.py, health.py
├── services/          # business logic per domain
├── repositories/      # SQLAlchemy data access
└── security/          # jwt.py, api_keys.py, password.py, rbac.py
```

### `frontend/` — Streamlit
```text
frontend/
├── app.py             # entry: navigation
├── pages/             # 1_Predict.py, 2_Leaderboard.py, 3_Monitoring.py, 4_Admin.py
├── components/        # audio_recorder.py, waveform.py, prob_chart.py, spectrogram.py
└── api_client.py      # typed client for the backend API
```

---

## 3. `deployment/docker/`

```text
docker/
├── backend.Dockerfile      # FastAPI + inference (CPU) image
├── frontend.Dockerfile     # Streamlit image
├── training.Dockerfile     # optional: mirrors Colab env for local small runs
├── base.Dockerfile         # shared python+deps layer (cache-friendly)
└── entrypoints/
    ├── backend-entrypoint.sh   # run migrations then uvicorn
    └── frontend-entrypoint.sh
```

`deployment/compose/docker-compose.yml` orchestrates: `backend`, `frontend`,
`postgres`, `minio`, `mlflow`, `prometheus`, `grafana` — the full local stack (FR-P6).

---

## 4. `tests/` (mirrors `src/`)

```text
tests/
├── unit/            # pure logic: harmonize, split, features, metrics, security
├── integration/     # DB repositories, storage client, registry, API routers (test DB)
├── e2e/             # docker-compose up → predict flow → leaderboard flow
├── fixtures/        # tiny sample .wav clips, seed configs
└── conftest.py      # fixtures: test db, test storage, sample audio, auth tokens
```

---

## 5. Rationale

| Choice | Why |
|--------|-----|
| `src/` layout (single installable package) | Prevents import ambiguity; one `pip install -e .` for both planes |
| `ml/` shared by training + inference | Guarantees feature/preprocess parity (ADR-3) — the top SER prod bug |
| `configs/` split by concern + composable `experiments/` | Config-over-code; a new run = a new YAML (US-5, FR-M4) |
| `datasets/`, `experiments/` gitignored | Never commit data/artifacts; keeps repo light and licence-clean |
| `notebooks/` thin, call `src/` | Colab/Kaggle notebooks don't fork logic; reproducible (NFR-3, NFR-4) |
| `scripts/` = thin CLIs | Ops entrypoints without bloating packages; testable core stays in `src/` |
| `deployment/` centralizes infra | One place for Docker/CI/monitoring; clean cloud-migration story |
| `tests/` mirrors `src/` | Discoverability; coverage maps to modules (NFR-5) |
| `docs/model_cards/` | Responsible-AI hygiene; reads as senior maturity |
