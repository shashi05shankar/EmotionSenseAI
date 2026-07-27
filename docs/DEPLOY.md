# Deployment Guide

Two supported paths: a **one-click-ish live demo** (Hugging Face Spaces) and the **full
local stack** (Docker Compose). The demo is the recommended public deployment — it's free,
always-on, and needs no database.

---

## A. Hugging Face Spaces (recommended live demo)

The Space is self-contained (`deployment/hf_space/`): it installs the package from GitHub,
trains a small demo model on first load, and serves upload → prediction → explainability.

**Steps (≈3 min):**
1. Create a new Space at <https://huggingface.co/new-space> → SDK: **Streamlit**.
2. Add three files to the Space repo (copy from `deployment/hf_space/`):
   - `README.md` (the Space card with the front-matter)
   - `requirements.txt`
   - `app.py`
3. Commit/push. The Space builds, installs `emotionsense[frontend]` from the `v1.0.0` tag,
   and launches. First load trains the demo model (~10 s), then it's interactive.

> The bundled model is trained on the synthetic fixture (zero-dependency). To demo real
> speech, train on RAVDESS/CREMA-D locally, add the artifact to the Space's registry
> directory, and the app will serve it as the production default.

**Alternative (Docker Space):** point a Docker Space at `deployment/docker/frontend.Dockerfile`.

---

## B. Full local stack (Docker Compose)

Brings up backend + frontend + postgres + minio + mlflow:

```bash
docker compose -f deployment/compose/docker-compose.yml up --build
# API   -> http://localhost:8000/docs
# UI    -> http://localhost:8501
# MLflow-> http://localhost:5000
```

Seed a model first if the registry is empty:

```bash
python scripts/seed_db.py            # trains + registers svm-mfcc on synthetic
```

---

## C. Local (no Docker) smoke run

```bash
pip install -e ".[backend,frontend]"
python scripts/train_and_register.py --model svm --dataset synthetic --promote
uvicorn emotionsense.backend.main:app          # API
streamlit run src/emotionsense/frontend/app.py # UI (separate shell)
```

---

## Notes

- **Auth:** the API's admin account comes from `ESA_ADMIN_EMAIL` + `ESA_ADMIN_PASSWORD_HASH`
  (generate a hash with `python scripts/seed_db.py --admin-password '…' --skip-model`). No
  credentials are baked into the image.
- **Secrets:** never commit `.env`; set env vars in the host / Space settings.
- **Cloud swap:** MinIO → S3, local Postgres → managed Postgres (Neon/RDS) — same interfaces,
  no app code change (see [`docs/design/02-TDD.md`](design/02-TDD.md)).
