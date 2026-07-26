# Release Candidate 1 (RC1)

**Product:** EmotionSense AI — v2.0 Lean Core
**Tag:** `v1.0.0-rc1`
**Date:** 2026-07-27
**Scope:** resolves audit findings **A1** (test rigor) and **A2** (real auth) only. No new
features, no architecture/model/methodology changes.

---

## What changed since the audit (`d626242`)

### A1 — Test coverage & harness tests
- **Coverage: 58% → 83.01%** (gate: **70%**, enforced).
- New tests for the previously-0%-covered differentiator and support code:
  - `tests/integration/test_benchmark.py` — speaker-independent CV + cross-corpus
    (`run_cv`, `run_cross_corpus`, leaderboard ordering, baseline-near-chance, 6-class
    projection). `training/benchmark.py` 0% → **98%**, `evaluate.py` → **100%**.
  - `tests/integration/test_runner_report.py` — full experiment orchestration + export.
    `runner.py` → 64%, `report.py` → **100%**.
  - `tests/unit/test_evaluate.py` — metric correctness incl. **UA-is-imbalance-robust**.
  - `tests/unit/test_manifest.py` — manifest round-trip. `manifest.py` → **100%**.
  - `tests/unit/test_yaml_config.py` — config load + experiment resolution. → **100%**.
- **CI coverage gate added**: `--cov-fail-under=70` in `.github/workflows/ci.yml` and
  `fail_under = 70` in `pyproject.toml` (will rise toward 80 as the DB path lands).
- Optional-extra modules (`ml/models/deep.py`, `ml/features/ssl.py`) excluded from
  coverage — they require torch/transformers, not present in the default test env.
- **Benchmark logic unchanged** — no methodology was modified (constraint honored).

### A2 — Real JWT authentication
- **Removed** the demo stub: hardcoded plaintext password (`"admin123"`), `_DEMO_USERS`,
  `_demo_check`, and the dead/misleading `_DEMO_USERS["hash"]` field are all gone.
- **Wired** `hash_password()` / `verify_password()` into the login flow via a new
  `backend/users.py` provider; login now verifies a **bcrypt** hash and issues a JWT.
- **No credentials in source**: the admin account is provisioned from the environment
  (`ESA_ADMIN_EMAIL` + `ESA_ADMIN_PASSWORD_HASH`). If no hash is set, **no account can
  authenticate** (secure by default).
- **Dependency fix**: replaced the unmaintained `passlib` (broken with `bcrypt ≥ 4.1`,
  raised `ValueError` on hashing) with the `bcrypt` library directly. Verified with
  `bcrypt 5.0.0`.
- **Docs updated**: `docs/design/04-api-specification.md` now marks `/auth/login` as
  implemented (bcrypt + JWT, no refresh token) and the other `/auth/*` endpoints as
  designed-not-implemented. `.env.template` documents the admin hash; `scripts/seed_db.py`
  gained `--admin-password` to generate the hash (uses `hash_password`, so it's no longer
  dead code).

---

## Verification (all run locally, this environment)

| Gate | Result |
|------|--------|
| `ruff check src tests` | ✅ All checks passed |
| `black --check src tests` | ✅ 72 files unchanged |
| `pytest -m "unit or integration" --cov-fail-under=70` | ✅ **32 passed**, **83.01%**, exit 0 |
| Smoke benchmark (synthetic) | ✅ exit 0 |
| FastAPI boot + auth wiring | ✅ app builds; bcrypt hash/verify OK; no hardcoded creds |
| Auth flow tests | ✅ login 200 (valid) / 401 (bad pw, unknown email); admin route 401 no-token, 404 with token |

**Test count:** 12 → **32** (20 added).

---

## Known limitations carried forward (not in A1/A2 scope)

- DB-backed persistence (users/predictions/audit_logs) + Alembic migrations remain
  **designed, not built** (finding F4). The runnable path uses the filesystem registry.
- API is the Lean Core subset (F5); remaining spec endpoints are deferred.
- `datasets/augment.py` remains unintegrated (F3) — left untouched per "no new features".
- Docker build (F6) still unverifiable in this environment (no Docker).

These are unchanged by RC1 and tracked for a later milestone.

---

## RC1 sign-off

RC1 resolves the two Major audit findings (A1, A2). Lint clean, coverage gate green at 83%,
32 tests pass, auth is real and credential-free. **Ready for RC review.**
