# Architecture & Design Review (Pre-Implementation Gate)

**Product:** EmotionSense AI — v2.0
**Date:** 2026-07-26
**Reviewers (personas):** Staff Software Engineer · ML Architect · MLOps Engineer · Hiring Manager (FAANG/NVIDIA/OpenAI/top-startup)
**Scope reviewed:** `docs/research/01–03`, `docs/design/01–08`
**Mandate:** find risks, bloat, gaps, and improvements — *not* rewrite the docs.
**Relationship to [08-design-review.md](08-design-review.md):** doc 08 was a self-review focused on complexity trimming. This review is broader and adversarial; it **assumes doc 08's "Lean Core" is adopted** and hunts for what doc 08 *missed* — chiefly ML rigor and security gaps.

---

## 0. TL;DR

The design is **above the bar for a portfolio/production hybrid** and its central thesis
(honest, reproducible, cross-corpus benchmarking behind a real product) is genuinely
differentiating. But the current docs are **stronger on software architecture than on ML
methodology**, and that inversion is exactly backwards for the audience that matters most
(ML hiring managers). There are **6 must-fix issues**, all in ML rigor and security, none
requiring redesign. Verdict: **Approve with changes.**

---

## 1. Architecture Review

**Scalable?** Yes, appropriately. Stateless API + shared Postgres/object-store + registry
is a textbook horizontally-scalable shape. The train/serve plane split (ADR-1) is the
strongest architectural decision — it's what makes "train on Kaggle, serve on a laptop"
real rather than aspirational.

**Over-engineered?** In places, yes — already largely caught by doc 08 (Prometheus/Grafana,
Celery, RBAC depth, training image). One thing doc 08 *under-sold*: the three "platform"
services actually **compose naturally** — MLflow uses **Postgres as its backend store and
MinIO as its artifact store**. So `postgres + minio + mlflow` isn't three independent
burdens; it's one coherent tracking substrate. **Keep all three; document the synergy** so
it doesn't read as service sprawl.

**Can services merge?** Already merged well (inference in-process with backend). Further
merging would hurt clarity. No action.

**Unnecessary dependencies?** Two to challenge:
- **Postgres vs SQLite (R-A1, minor):** for a single-node demo, SQLite would remove a
  service. **Recommendation: keep Postgres** — it's what MLflow wants anyway, and "I used a
  real RDBMS with migrations" is a better signal than SQLite. Net: no change, but the
  justification should be explicit.
- **XGBoost + sklearn + torch + transformers** — all justified; no bloat.

**Simpler architecture, same resume value?** The Lean Core (doc 08 §4) is close to optimal.
The single biggest *simplification-with-no-signal-loss* not yet stated: **the auth/RBAC/
audit subsystem carries almost no ML signal.** It's good "production hygiene" but a hiring
manager at NVIDIA/OpenAI will not be moved by JWT. Keep a *thin* version (see §4/§8); don't
invest days polishing it.

**Verdict:** architecture is sound and scalable; trim per doc 08 + keep the tracking-substrate synergy.

---

## 2. Machine Learning Review  ← *weakest area; most must-fixes here*

**Model selection:** appropriate and well-motivated (classical → deep → SSL, with
Distil-HuBERT as the compute-aware centerpiece). Good.

**Remove any models?** Yes — **CNN and LSTM are low-value middle children.** On these small
corpora, a BiLSTM covers the "RNN on sequences" story and a CNN-on-spectrogram rarely beats
the SSL models while adding tuning burden. Lean Core already defers CNN/LSTM — **endorse
that; they're demonstration filler, not signal.**

**Missing baselines (MUST-FIX, R-ML1):** there is **no trivial baseline.** A credible
benchmark *must* include a **majority-class** and a **stratified-random** baseline, and
ideally **logistic regression on MFCC-means**. Without them, "SVM got 78%" is uninterpretable
(is 78% good? what's chance for 7 imbalanced classes?). This is the cheapest possible add and
the clearest sign of ML maturity. **Add a baseline row to the harness.**

**Evaluation methodology (MUST-FIX, R-ML2):** the docs specify UA/WA/macro-F1/confusion
matrix — good — but report a **single split** (`si-fold0`). For 1,440-sample RAVDESS,
single-split accuracy has a **±3–5% swing**. A benchmark claiming rigor **must report
k-fold (speaker-independent) mean ± std**, or at minimum bootstrapped confidence intervals.
Otherwise leaderboard deltas are noise. **Make CV mean±std the default; single-split is a
dev shortcut only.**

**Class imbalance (MUST-FIX, R-ML3):** unaddressed. RAVDESS "neutral" has half the samples of
other classes; CREMA-D is imbalanced. The training configs don't mention **class weights /
stratified sampling**, and the choice between optimizing WA vs UA isn't stated as a policy.
**Decide: report UA as the headline metric (imbalance-robust) and use class-weighted loss.**

**Benchmark framework realistic?** Mostly. Two realism gaps:
- **Statistical comparison** (R-ML2 above).
- **Compute honesty:** SSL feature extraction (a HuBERT forward pass) **on CPU at serve time
  can exceed the 3s budget** for longer clips. The NFR-1 transformer budget is optimistic.
  **Either cache/limit clip length, default-serve a light model, or relax the budget with
  measured numbers.** (R-ML5, see §4.)

**Cross-corpus evaluation correct? (MUST-FIX, R-ML4):** the *intent* is right and is the
project's crown jewel, but the *mechanics* are under-specified and currently **wrong-by-
omission**: RAVDESS (8 classes incl. *calm*, *surprise*), TESS (7, incl. *pleasant surprise*),
and CREMA-D (**6**, no *calm*, no *surprise*) **do not share a label set.** A naive
train-RAVDESS→test-CREMA-D will mismatch classes. Cross-corpus eval **must project both to the
shared intersection** (typically the 6: angry, disgust, fear, happy, neutral, sad) and say so.
Also flag **lexical leakage** within corpora (RAVDESS = 2 fixed sentences; TESS = "say the word
___") — models can cheat on content; note it as a limitation. **This must be designed before
coding or the headline result is invalid.**

**Missing ML considerations to add (Should-have):**
- **Variable-length audio handling** at inference (windowing + probability aggregation) — not
  specified; real clips aren't all ≤3s.
- **Confidence calibration** — softmax confidences shown in the UI are uncalibrated;
  temperature-scaling or an explicit "scores are uncalibrated" caveat.

**Verdict:** methodology is 70% there; the missing 30% (baselines, CV+CIs, imbalance policy,
cross-corpus label intersection) is exactly what separates a *student benchmark* from a
*credible* one. All cheap. **Fix before P2/P3.**

---

## 3. Dataset Strategy Review

- **Ordering:** correct (RAVDESS → TESS → CREMA-D; IEMOCAP last, licence-gated). Good.
- **Compatibility / label mapping (see R-ML4):** the canonical-taxonomy idea is right, but the
  **RAVDESS `calm` drop** and the **cross-corpus 6-class intersection** must be explicit in
  `configs/datasets/*` and the harmonizer. Also decide RAVDESS **speech-only** (exclude song).
- **Train/val/test:** structure fine; upgrade to **k-fold** (R-ML2).
- **Speaker-independent:** correctly enforced at pipeline level (ADR-4) — the standout data
  decision. Add a **statement/lexical-leakage caveat** to model cards.
- **Augmentation:** noise/pitch/time-stretch is standard and good. **Add a guardrail:**
  augment **train only, never val/test**, and make the harness assert it — a classic leak.

**Verdict:** strong skeleton; the label-intersection and CV upgrades are the difference
between "looks careful" and "is careful."

---

## 4. Backend Design Review

- **API design:** clean, versioned, correct error taxonomy, OpenAPI. Above bar. Minor: the
  `/predict` `model` selector by arbitrary version invites cache thrash — **cap concurrently
  loadable models** (the LRU already helps; make the cap explicit).
- **Database necessity:** justified (see §1). Keep.
- **Authentication:** **over-scoped for the audience.** JWT+2 roles is plenty; **drop refresh
  tokens and API-key scopes from Core** (doc 08 agreed — reaffirmed). It's plumbing, not signal.
- **Logging:** structlog JSON + correlation IDs — genuinely good, keep.
- **Error handling:** typed hierarchy + envelope — good.
- **Security (MUST-FIX, R-SEC1):** **model artifacts loaded from object storage via
  pickle/torch.load are an RCE vector.** If the registry/bucket is ever compromised, a
  malicious pickle executes on the serving host. The docs don't address deserialization trust.
  **Require `safetensors` for torch weights and a documented, checksum-verified load path for
  sklearn/joblib artifacts; treat the artifact store as a trust boundary.** This is the one
  security issue a Staff reviewer would block on. (Input validation, rate limiting, secrets
  handling are all already well-covered — good.)
- **Inference latency (R-ML5):** default serving model should be **classical or Distil-HuBERT**,
  transformer heavy models opt-in, with **measured** latency in the model card, not an assumed
  budget.

**Verdict:** solid; fix the pickle trust boundary, thin the auth.

---

## 5. Frontend Review

- **Prediction interface:** upload/record → label + probability bar + waveform + spectrogram is
  the right, demo-friendly core. Keep.
- **Dashboard/monitoring:** doc 08's "DB-backed Streamlit monitoring page" instead of Grafana in
  Core is the right call. Endorse.
- **Experiment explorer / leaderboard:** high value — **this is the page that sells the ML
  story.** Recommend it show **cross-corpus vs in-corpus side by side** and **±std** — make the
  rigor *visible*, not buried in a JSON export.
- **User workflow:** fine. One addition: a **"try a sample clip" button** so an evaluator with no
  audio file still gets the 60-second wow (Ravi persona).
- **Risk:** Streamlit + real auth/multi-user is awkward; keep the UI essentially single-user/
  demo and let the API own auth. Don't over-invest in Streamlit auth.

**Verdict:** well-judged; make the benchmark rigor *visible* in the UI — it's your best asset.

---

## 6. MLOps Review

- **Experiment tracking:** MLflow — correct, portable, offline-capable. Keep.
- **Model registry:** custom DB registry as serving truth (ADR-2) — correct separation. Keep,
  but **consider using MLflow's own Model Registry stages** to avoid reinventing promote/retire;
  if the custom table stays, keep it thin. (Judgment call, not a blocker.)
- **CI/CD:** GitHub Actions plan is right-sized. Good.
- **Docker:** compose-for-Core (5 services) is demoable. Good.
- **Deployment (Should-fix, R-OPS1):** **the weakest MLOps link.** "Render/Railway/HF Spaces"
  is hand-wavy. Free tiers **sleep/cold-start** and a multi-service compose stack won't drop onto
  HF Spaces cleanly. **Decide the real target now:** e.g. **UI on HF Spaces + API on a small
  always-on host (Fly.io free) + managed Postgres (Neon) + object store (Cloudflare R2/S3)**, or
  accept "local `docker compose up`" as the demo and record a video. A vague deploy story
  becomes a broken demo link — worse than none.
- **Monitoring:** Lean Core (logs + `/metrics` + Streamlit page) is enough; Prometheus/Grafana as
  stretch. Good.
- **Missing (Nice-to-have):** data/version lineage — checksum-based versioning is fine for Core;
  DVC is overkill. Mention determinism caveats (GPU nondeterminism) in reproducibility claims.

**Verdict:** strong except deployment — **pin a concrete, always-on deploy target.**

---

## 7. Folder Structure Review

Maintainable, conventional (`src/` layout), mirrors the architecture. **Endorse.** Nits:
- `datasets/` (local data root) vs `src/emotionsense/datasets/` (code) is a **name collision**
  that will confuse newcomers. **Rename the data root to `data/`.** (R-STRUCT1, trivial.)
- `notebooks/` is correct and important (Colab/Kaggle) — good that logic stays in `src/`.
- Interfaces (`Model`, `DatasetLoader`) should be introduced **when the 2nd implementation lands**,
  not speculatively (doc 08 said this; reaffirm — avoid premature ABCs).

**Verdict:** good; rename `datasets/`→`data/`.

---

## 8. Resume Value Review (Hiring-Manager persona)

**What impresses (strongest signal — protect these):**
1. **Speaker-independent + cross-corpus benchmarking with baselines and CIs.** This is rare in
   student projects and immediately reads as "understands generalization and evaluation." *This
   is 60% of the project's hiring value.*
2. **Train/serve parity via feature-spec-with-model + a golden parity test.** Signals real
   production ML instincts.
3. **Reproducibility** (config snapshots, seeds, versioned data) — senior signal.
4. **Distil-HuBERT choice justified by compute constraints** — shows engineering judgment, not
   just "used the biggest model."

**What reads as bloat (low ML signal):**
- Auth/RBAC/refresh-tokens/API-key-scopes, audit-log UI, Prometheus/Grafana, Celery. Generic
  web/infra plumbing. Keep *thin* versions for the "production" claim; **do not spend days here.**
  A hiring manager spends 5 minutes — none on your JWT.

**Strongest signals to ADD:**
- The **baselines + statistical rigor** (R-ML1/2) — cheapest, highest-signal add in the whole
  project.
- A **one-page "what I learned / honest results" writeup** including the cross-corpus accuracy
  *drop* framed as a finding. Owning a "bad" cross-corpus number is a *stronger* signal than a
  suspicious 95%.
- **Measured latency/throughput numbers** in the README (not just NFR targets).

**What to remove:** CNN/LSTM (filler), deep auth, monitoring stack (from Core). Already in Lean
Core — reaffirmed.

**Net:** with the ML-rigor fixes, this is a **top-decile student/junior portfolio project** and a
credible junior-MLE work sample. Without them, it's an impressive-looking but methodologically
ordinary SER repo — which is precisely the category the project set out to beat.

---

## 9. Lean Core Review (MoSCoW)

Refines doc 08 §4 with the review's findings folded in.

### Must Have (this *is* v2.0)
- Data: RAVDESS + TESS + CREMA-D; **speaker-independent k-fold**; canonical labels + **explicit
  cross-corpus 6-class intersection**; augment-train-only guardrail.
- Models: **baselines (majority/random/logreg)** + SVM + RandomForest + XGBoost + BiLSTM +
  frozen-emotion2vec head + Distil-HuBERT. (≥6 real + baselines.)
- Eval: UA headline + WA + macro-F1 + per-class + confusion matrix, **mean ± std over folds**,
  **≥1 cross-corpus result**.
- Serving: FastAPI `/predict` (+batch), registry (**safetensors/verified load**), thin JWT auth,
  admin promote/retire.
- UI: Predict page + **Leaderboard showing in-corpus vs cross-corpus with std** + sample-clip
  button.
- MLOps: MLflow + DB summary rows + Docker compose (backend, frontend, postgres, minio, mlflow) +
  GitHub Actions (lint/type/test) + structlog + `/metrics`.
- Docs: README (reproduce-from-clone) + model cards (incl. leakage + calibration caveats) +
  **one concrete always-on deploy target**.

### Should Have
- CNN/LSTM (breadth), SAVEE/AESDD (more corpora), variable-length audio handling, confidence
  calibration, Streamlit monitoring page.

### Nice to Have
- Ensembles, dimensional emotion, real-time mic, Prometheus/Grafana, explainability overlays.

### Future Work
- IEMOCAP + multimodal (Whisper text), noise-robustness suite, k8s/autoscaling, active-learning
  loop, multilingual.

---

## 10. Final Recommendations

### Critical Issues (MUST fix before implementation)
1. **R-ML1 — Add trivial baselines** (majority/random/logreg) to the harness. *Without a chance
   baseline, no accuracy number is interpretable.*
2. **R-ML2 — Statistical rigor:** speaker-independent **k-fold mean ± std** (or bootstrap CIs) as
   the default benchmark output, not a single split.
3. **R-ML3 — Class-imbalance policy:** class-weighted training + **UA as headline metric**;
   state it in configs and model cards.
4. **R-ML4 — Cross-corpus label intersection:** design the **shared 6-class projection** and
   lexical-leakage caveat now; the flagship result is invalid without it.
5. **R-SEC1 — Artifact deserialization trust:** mandate **safetensors** for torch and a
   checksum-verified load path; treat the artifact store as a trust boundary (RCE risk).
6. **R-OPS1 — Pin a concrete, always-on deployment target** (or explicitly ship "local compose +
   demo video"); no vague free-tier promise that yields a dead link.

### Important Improvements (should fix)
- Variable-length audio inference (windowing + aggregation).
- Make benchmark rigor **visible in the UI** (in-corpus vs cross-corpus, ±std).
- Confidence **calibration** or an explicit uncalibrated caveat.
- Reaffirm Lean Core: **cut CNN/LSTM and deep auth from Core.**
- Add a **sample-clip** button for zero-friction demo.
- Measured latency numbers in README; relax/justify the transformer latency budget.

### Optional Improvements
- Use MLflow's Model Registry stages instead of a custom promote/retire (judgment call).
- Rename local data root `datasets/` → `data/` to avoid the code/data name collision.
- Document the postgres+minio+mlflow synergy so it doesn't read as service sprawl.
- Determinism/GPU-nondeterminism caveat in reproducibility claims.

### Scores

| Dimension | Score | Note |
|-----------|-------|------|
| **Architecture** | **8.5 / 10** | Scalable, clean plane-separation; minor over-provisioning already being trimmed |
| **Resume Value** | **8 / 10** now → **9 / 10** with R-ML1–4 | ML rigor is the multiplier; currently under-weighted vs software plumbing |
| **Production Readiness** | **6.5 / 10** | Design-stage; blocked mainly by R-SEC1 and R-OPS1 |
| **Complexity** *(lower is better)* | **7 / 10** as written → **4 / 10** at Lean Core | Trim auth depth + monitoring stack + CNN/LSTM |

### Recommendation

> **✅ Approve with changes.**
>
> No redesign required — the architecture and data strategy are sound and the differentiator is
> real. Implementation may begin **once the 6 Critical Issues are folded into the configs/specs**
> (they are additive edits to the ML harness, security posture, and deploy plan — not structural
> changes). Adopt **Lean Core** as the v2.0 target. Sequence the critical fixes into the phases
> where they first bite (below) rather than as a separate up-front phase.

---

## 11. Prioritized Action List (fold into the roadmap, no new phase)

| # | Action | Type | Lands in phase | Effort |
|---|--------|------|----------------|--------|
| A1 | Add majority/random/logreg baselines to harness | Critical (R-ML1) | P2/P3 | 0.5 d |
| A2 | Default benchmark = speaker-independent k-fold, report mean±std | Critical (R-ML2) | P1 (splits) + P3 | 1 d |
| A3 | Class-weighted training; UA as headline metric | Critical (R-ML3) | P2 | 0.5 d |
| A4 | Cross-corpus 6-class intersection + leakage caveat, in configs | Critical (R-ML4) | P1 + P3 | 1 d |
| A5 | safetensors + checksum-verified artifact load; artifact store = trust boundary | Critical (R-SEC1) | P6 | 0.5 d |
| A6 | Pin concrete always-on deploy target (or "local + video") | Critical (R-OPS1) | P13 (decide at P0) | 0.5 d |
| A7 | Variable-length audio: windowing + prob aggregation | Important | P6 | 0.5 d |
| A8 | Leaderboard UI: in-corpus vs cross-corpus + ±std; sample-clip button | Important | P8 | 0.5 d |
| A9 | Cut CNN/LSTM + deep auth from Core; thin JWT only | Important | P0 scope decision | — |
| A10 | Confidence calibration or explicit caveat | Important | P8 | 0.5 d |
| A11 | Rename data root `datasets/`→`data/`; document tracking-substrate synergy | Optional | P0 | 0.1 d |

**Total added Critical effort: ~4 ideal days**, all additive to existing phases — offset by the
~10 days Lean Core already removes. Net effort **decreases** while methodological credibility
**increases**.

**Gate:** implementation begins after this review is approved and A1–A6 are reflected in the
relevant configs/specs.
