# Coding Standards & Engineering Conventions

**Product:** EmotionSense AI
**Version:** 2.0 (design)
**Date:** 2026-07-26
**Related:** [`05-folder-structure.md`](05-folder-structure.md), [`07-implementation-roadmap.md`](07-implementation-roadmap.md)

> These standards are enforced mechanically where possible (pre-commit + CI), so review
> focuses on design, not formatting. All rules are justified — a rule nobody can defend
> gets dropped.

---

## 1. Python Style Guide

- **Version:** Python 3.11+. Use modern typing (`list[str]`, `X | None`, `match`).
- **Formatter:** **Black** (line length 100). Non-negotiable, zero-config debates.
- **Linter:** **Ruff** (superset of flake8/isort/pydocstyle/pyupgrade). Config in `pyproject.toml`.
- **Type checking:** **mypy** in strict mode on `src/` (tests may relax). Public functions
  must be fully type-hinted. *Why:* types are executable docs and catch the class of
  train/serve-mismatch bugs early.
- **Docstrings:** Google style, on every public module/class/function. First line is a
  one-sentence imperative summary.
- **Imports:** absolute within the package (`from emotionsense.ml import ...`); no
  wildcard imports; isort ordering (stdlib → third-party → first-party).
- **Data models:** Pydantic v2 for I/O boundaries (API, config); dataclasses for
  internal value objects; SQLAlchemy models only in `repositories/`.
- **No magic numbers:** canonical constants (sample rate 16000, label set) live in
  `common/constants.py`.
- **Purity:** `ml/` and `datasets/` functions are side-effect-free where feasible
  (I/O pushed to edges) → testable and reproducible.

---

## 2. Naming Conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Package / module | `snake_case`, short | `inference`, `model_cache.py` |
| Class | `PascalCase` | `InferenceService`, `DatasetLoader` |
| Function / variable | `snake_case` | `extract_features`, `sample_rate` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_SAMPLE_RATE`, `LABELS` |
| Type alias / Protocol | `PascalCase` | `FeatureSpec`, `Model` (Protocol) |
| Pydantic DTO | `PascalCase`, suffix by role | `PredictionResponse`, `TrainRunCreate` |
| SQLAlchemy model | `PascalCase` singular | `ModelVersion`, `AuditLog` |
| DB table | `snake_case` plural | `model_versions`, `audit_logs` |
| Config file | `snake_case.yaml` | `distilhubert.yaml` |
| Env var | `UPPER_SNAKE_CASE`, prefixed | `ESA_DB_URL`, `ESA_S3_BUCKET` |
| Test | `test_<unit>_<condition>_<expected>` | `test_split_is_speaker_independent` |
| Model version | `<family>-<dataset>:<semver>` | `distilhubert-ravdess:v3` |

---

## 3. Git Workflow

- **Trunk-based with short-lived branches.** `main` is always green and deployable.
- **No direct commits to `main`.** All change via PR + green CI + (self-)review.
- **Rebase, don't merge-commit** feature branches onto `main` to keep history linear.
- **One logical change per PR.** A PR maps to one roadmap task or sub-task.
- **PRs must:** pass CI (lint + type + tests), update docs/tests, keep coverage ≥ target.

### Branch Strategy

| Branch | Purpose | Naming |
|--------|---------|--------|
| `main` | Stable, deployable trunk | — |
| `feat/*` | New feature | `feat/predict-endpoint` |
| `fix/*` | Bug fix | `fix/feature-parity-mfcc` |
| `chore/*` | Tooling/infra/deps | `chore/ci-caching` |
| `docs/*` | Docs only | `docs/model-card-distilhubert` |
| `exp/*` | Throwaway experiments (never merged as-is) | `exp/wav2vec2-finetune` |

Phase branches (from the roadmap) may act as integration branches merging several
`feat/*` PRs, then merge to `main` at phase completion.

### Commit Message Format (Conventional Commits)

```text
<type>(<scope>): <imperative summary ≤ 72 chars>

<body: what & why, wrapped at 72; not "how" — the diff shows how>

<footer: refs #issue, BREAKING CHANGE: ...>
```

- **types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`, `build`.
- **scope:** package/module (`backend`, `ml`, `datasets`, `infra`).
- **Example:** `feat(inference): resolve feature spec from model metadata`
- *Why:* enables automated changelog + semantic version bumps later.

---

## 4. Logging Standards

- **Library:** `structlog` emitting **JSON** to stdout (12-factor; NFR-6).
- **Never** use bare `print()` in `src/`. Never log secrets, raw audio, or full tokens.
- **Levels:** `DEBUG` (dev detail), `INFO` (lifecycle events), `WARNING` (recoverable),
  `ERROR` (handled failure), `CRITICAL` (service-threatening).
- **Mandatory context keys** on request-path logs: `correlation_id`, `user_id`
  (or `anon`), `route`, `latency_ms`; on inference logs add `model_version`.
- **One event per log line**, event name in `event=` (e.g. `event="prediction.completed"`).
- **No f-string interpolation of user data into the message**; pass as structured
  fields so logs stay queryable and injection-safe.

---

## 5. Error Handling Standards

- **Typed exception hierarchy** in `common/errors.py` (`AppError` → `ValidationError`,
  `NotFoundError`, `AuthError`, `ModelUnavailableError`, ...). Each maps to one HTTP
  code + error envelope (see [`04-api-specification.md`](04-api-specification.md)).
- **Fail fast at boundaries:** validate inputs at the router/service edge; interior
  code assumes validated data.
- **Never swallow exceptions** (`except: pass` is banned). Catch narrow, add context,
  re-raise or convert to a typed `AppError`.
- **No secrets/internal details in client-facing messages;** full detail goes to logs
  keyed by `correlation_id`.
- **Graceful degradation (NFR-9):** a missing/unloadable model returns `503
  MODEL_UNAVAILABLE`, never a stack trace or crash.
- **Idempotency & atomicity:** model promotion is an atomic transaction (only one
  production default per name; enforced by DB partial index + service logic).
- **Retries** only for transient I/O (object storage), with bounded backoff; never
  retry validation errors.

---

## 6. Testing Standards

- **Framework:** `pytest` + `pytest-cov`; `httpx`/`TestClient` for API; `factory-boy`
  or fixtures for data.
- **Pyramid:** many **unit** (fast, pure `ml`/`datasets`/`security` logic), fewer
  **integration** (DB/storage/registry/routers against a throwaway test DB), few
  **e2e** (compose-up smoke of predict + leaderboard flows).
- **Coverage gate:** ≥ **80%** on `ml`, `inference`, `backend`, `registry` (NFR-5); CI
  fails below. Coverage is a floor, not a goal — assert behavior, not lines.
- **Determinism:** seed all RNGs; no network in unit tests (SSL model downloads mocked
  or marked `@pytest.mark.slow` and excluded from the default run).
- **Golden tests for parity:** a fixture asserts that train-time and serve-time feature
  extraction produce identical vectors for the same clip (guards ADR-3).
- **Contract tests:** API responses validated against the Pydantic response schemas.
- **Naming:** `test_<unit>_<condition>_<expected>`; one behavior per test; Arrange-Act-
  Assert structure.
- **Markers:** `@pytest.mark.unit|integration|e2e|slow`; CI runs unit+integration on
  every push, e2e on merge to `main`.

---

## 7. Configuration & Secrets

- All config via Pydantic Settings; layering **defaults → YAML → env** (env wins).
- **No secrets in the repo.** `.env` gitignored; `.env.template` documents every key.
- `detect-secrets` pre-commit hook blocks accidental credential commits.
- Experiment configs are **snapshotted** into the `experiments`/`training_runs` records
  so a run is reproducible even if the YAML later changes (NFR-3).

---

## 8. Documentation Standards

- Every package has a `README` or module docstring stating its responsibility and
  public surface.
- Public API documented via FastAPI/OpenAPI (kept in `docs/api/`).
- Each shipped model has a **model card** in `docs/model_cards/` (data, metrics,
  intended use, **limitations & ethical note** — research/education only).
- Architecture diagrams (Mermaid) kept in `docs/`; update in the same PR as the change
  they describe (docs and code move together).

---

## 9. CI Gate (summary of what must pass)

1. `ruff check` + `black --check` (style)
2. `mypy src/` (types)
3. `pytest -m "unit or integration" --cov` ≥ threshold
4. `detect-secrets` scan
5. Build backend + frontend Docker images
6. (on `main`) e2e smoke via docker compose

No PR merges with a red gate; no bypassing hooks without an explicit, justified note.
