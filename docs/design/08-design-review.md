# Design Review — Complexity Audit & Simplifications

**Product:** EmotionSense AI
**Version:** 2.0 (design)
**Date:** 2026-07-26
**Reviewer role:** self-critical architecture pass over docs 01–07.

> Purpose: pressure-test the design for **unnecessary complexity** before any code is
> written. A design that looks impressive but can't be built solo on free compute is a
> worse portfolio piece than a smaller one that ships. Each finding has a verdict:
> **KEEP**, **SIMPLIFY**, **DEFER** (move to stretch), or **CUT**.

---

## 1. Executive Verdict

The design is **sound and coherent**, and its core thesis — *honest, reproducible,
multi-model benchmarking wrapped in a real product* — is the right differentiator. The
main risk is **breadth**: ~13 core phases and ~8 backing services is a lot for one
person on free compute. The recommendations below trim roughly **20–25% of build
effort** with **near-zero loss of resume/portfolio value**, mostly by deferring
enterprise-flavored pieces that don't show off ML skill.

**Net recommendation:** adopt a **"Lean Core"** (§4) as the true v2.0 target; treat
several currently-Core items as Stretch.

---

## 2. Findings

### F1 — Prometheus + Grafana is heavyweight for a solo demo. → **SIMPLIFY**
Two extra services (scrape config, dashboards, provisioning) for what a demo needs.
**Recommendation:** keep structured JSON logging (cheap, high value) and expose a
`/metrics` endpoint, but **defer the full Prometheus+Grafana stack to Stretch (S)**.
For the demo, a single **Streamlit "Monitoring" page** that queries the `predictions`
table (latency, throughput, class distribution) delivers ~80% of the visual payoff for
~20% of the ops cost. *Effort saved: ~1.5 d + ongoing compose weight.*

### F2 — MLflow **and** a custom registry **and** a `metrics` table overlaps. → **KEEP, but clarify boundary**
This looked like duplication in review. It isn't — it's deliberate (ADR-2): MLflow =
research history, `model_versions` = serving truth, `metrics` = queryable eval rows for
the leaderboard API. **Keep**, but the risk is a beginner wiring all three
inconsistently. *Mitigation:* MLflow is the **only** writer of experiment detail; the
DB stores just the summary rows the API needs. Don't mirror everything.

### F3 — API-triggered training (`/training-runs` auto-start, Celery in stretch). → **SIMPLIFY**
Real training is on Colab/Kaggle (ADR-7), so the API's training endpoints are mostly a
**results sink**, not an executor. The "local BackgroundTasks auto-start" path adds
code for a case that barely matters. **Recommendation:** make `/training-runs` and
`/models` **registration/reporting endpoints only** (POST/PATCH from notebooks via API
key). Drop in-API training execution entirely; **cut Celery (S8) unless a real need
appears.** *Effort saved: ~1 d; removes a whole failure surface.*

### F4 — Full RBAC + refresh tokens + API-key scopes on day one. → **SIMPLIFY**
Three auth mechanisms is enterprise-grade for a portfolio project. **Recommendation:**
ship **JWT (user/admin roles) + simple API keys** in Core; keep **scopes and refresh
tokens minimal or deferred.** Two roles and single-expiry keys are enough to
demonstrate the concept. *Effort saved: ~0.5 d.*

### F5 — Six datasets × nine models is a large matrix. → **DEFER breadth**
The PRD/roadmap already hedge this, but state it plainly: **Core ships 3 datasets
(RAVDESS, TESS, CREMA-D) and 6 models** (SVM, RandomForest, XGBoost, BiLSTM,
frozen-emotion2vec head, Distil-HuBERT). SAVEE/AESDD/IEMOCAP and CNN/LSTM/wav2vec2/
HuBERT are **easy add-ons** (config + loader) that demonstrate *extensibility* better
as "and it scales to N more via YAML" than as day-one grind. *Effort saved: ~3–4 d.*

### F6 — `dataset_splits` table risk of per-sample rows. → **KEEP (already mitigated)**
Flagged and solved in the schema (manifest-URI + seed, not 100k rows). No change; just
confirming the mitigation is mandatory, not optional.

### F7 — `training.Dockerfile` mirroring the Colab env. → **CUT from Core**
Maintaining a training image that mirrors Colab/Kaggle is effort for a path users won't
take (they'll use the notebooks). **Recommendation:** **cut** the local training image
from Core; notebooks + a documented `requirements` are enough. Add back only if local
training becomes common. *Effort saved: ~0.5 d + image maintenance.*

### F8 — Batch prediction + rate limiting + pagination everywhere. → **KEEP, thin**
All reasonable, but don't gold-plate. Batch endpoint: cap small (≤20) and keep simple.
Rate limiting: one middleware, in-memory token bucket is fine for a demo (no Redis).
**Keep**, implemented minimally.

### F9 — MinIO in Core vs. just using the local filesystem. → **KEEP MinIO**
Tempting to simplify to local FS, but MinIO is *cheap* (one compose service) and buys
the **exact S3 semantics** that make the "train on Kaggle, artifacts in object storage,
serve locally" story real and cloud-portable. The parity is worth one container.
**Keep.**

### F10 — Audit logs + soft deletes + `before`/`after` snapshots. → **KEEP, scope tight**
Good responsible-engineering signal and cheap. **Keep**, but only audit **lifecycle
actions** (promote/retire, role changes, deletes) — not every read. Don't build an
audit UI in Core; the admin endpoint + table suffice.

### F11 — Nine backing services in compose. → **SIMPLIFY (consequence of F1/F3)**
After F1 (defer Prometheus/Grafana) and F3 (no Celery/Redis), Core compose is:
**backend, frontend, postgres, minio, mlflow** — five services. That's a demoable,
comprehensible stack. Prometheus/Grafana/Redis join only in Stretch.

---

## 3. Things that are RIGHT (keep as-is)

- **Speaker-independent splits enforced at the pipeline** (ADR-4) — the core integrity
  move; non-negotiable.
- **Feature spec travels with the model** (ADR-3) + parity golden-test — kills the #1
  SER production bug.
- **Cross-corpus evaluation as a first-class `eval_kind`** — the strongest, cheapest
  differentiator.
- **Config-over-code** — the reason breadth is deferrable without redesign.
- **Train/serve plane separation** — makes the free-compute constraint actually work.
- **Vertical-slice-first roadmap** — de-risks shipping.

---

## 4. Recommended "Lean Core" (the real v2.0)

| Area | Lean Core | Moved to Stretch |
|------|-----------|------------------|
| Datasets | RAVDESS, TESS, CREMA-D | SAVEE, AESDD, IEMOCAP |
| Models | SVM, RandomForest, XGBoost, BiLSTM, frozen-emotion2vec, Distil-HuBERT (6) | CNN, LSTM, wav2vec2, HuBERT fine-tune, ensembles |
| Serving | FastAPI `/predict`(+batch), registry, JWT+API-key, admin promote/retire | dimensional output, real-time mic |
| UI | Predict page, Leaderboard page, simple Monitoring page (DB-backed) | explainability overlays |
| Tracking | MLflow + DB summary rows | — |
| Observability | structlog JSON + `/metrics` + Streamlit monitoring | Prometheus + Grafana |
| Infra | compose: backend, frontend, postgres, minio, mlflow | prometheus, grafana, redis, celery, k8s |
| Training exec | Colab/Kaggle notebooks report to API | in-API/Celery training |

**Result:** ~13 core phases → the same phase *sequence*, but P12 (full monitoring), the
training image, Celery, and 3 datasets/3 models slide to Stretch. Estimated Core effort
drops from ~40 → **~30 ideal days**, and every PRD §8 acceptance criterion still holds.

---

## 5. Residual Risks After Simplification

| Risk | Status | Note |
|------|--------|------|
| Scope creep | **Reduced** | Lean Core is genuinely minimal-yet-complete |
| Free-GPU limits | **Contained** | Only P5 is GPU-bound; checkpoint/resume mandatory |
| Feature parity bugs | **Contained** | ADR-3 + golden test |
| "Looks small" perception | **Low** | Rigor + cross-corpus + real API/UI reads as senior, not small |
| Over-abstracting interfaces early | **Watch** | Implement interfaces (`Model`, `DatasetLoader`) only when the 2nd implementation lands, not speculatively |

---

## 6. Action Items Feeding Implementation

1. Adopt **Lean Core** (§4) as the P0–P13 target; relabel deferred items as Stretch in
   [`07-implementation-roadmap.md`](07-implementation-roadmap.md) at kickoff.
2. Compose stack for Core = 5 services (drop Prometheus/Grafana/Redis).
3. Training endpoints = **report-only**; no in-API execution (update TDD ADR-7 note).
4. Auth = JWT(2 roles) + simple API keys; defer scopes/refresh polish.
5. Cut `training.Dockerfile` from Core.
6. Enforce: implement an interface only when a second implementation exists (avoid
   speculative abstraction).

**Bottom line:** the architecture is well-conceived and defensible; the only real
danger is doing *all* of it at once. Ship the Lean Core, then let Stretch phases be the
"and it extends cleanly to…" story — which is itself a stronger signal than a bloated
day-one build.
