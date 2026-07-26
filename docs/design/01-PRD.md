# Product Requirements Document (PRD)

**Product:** EmotionSense AI — Production-Ready Speech Emotion Recognition Platform
**Version:** 2.0 (design)
**Date:** 2026-07-26
**Status:** Draft for approval
**Related:** [`../research/03-gap-analysis.md`](../research/03-gap-analysis.md), [`02-TDD.md`](02-TDD.md)

---

## 1. Problem Statement

Speech Emotion Recognition (SER) research is abundant, but usable SER *systems* are
not. Our landscape survey of nine reference projects (see research phase) found a
consistent failure pattern:

1. **No product wrapper.** Projects stop at a notebook or a `infer.py` CLI. There is
   no API, no UI, no reproducible serving path — nothing an end user or downstream
   team can consume.
2. **Dishonest / incomparable evaluation.** Accuracy is reported on a single corpus,
   often with speaker-dependent splits that leak speaker identity into the test set,
   inflating numbers. Different projects use different class counts and metrics, so
   "93%" and "78%" cannot be compared.
3. **No cross-corpus generalization.** Almost nobody measures how a model trained on
   one dataset performs on another — the single most important question for real
   deployment.
4. **No engineering maturity.** No experiment tracking, no model versioning, no
   containerization, no monitoring, no tests.

**EmotionSense AI solves the *systems* problem, not (primarily) the *accuracy*
problem.** It delivers an end-to-end platform that (a) trains and rigorously
benchmarks multiple models under identical, honest, speaker-independent protocols,
(b) reports cross-corpus generalization openly, and (c) serves the best model through
a versioned API and an interactive UI, with experiment tracking, monitoring, and
containerized reproducibility.

---

## 2. Goals & Non-Goals

### Goals
- Provide a rigorous, reproducible multi-model, multi-dataset SER benchmark.
- Serve predictions via a documented REST API with a model registry (swap models by name).
- Provide an interactive UI for upload/record → prediction + probabilities + spectrogram.
- Track every experiment (config, metrics, artifacts) reproducibly.
- Be runnable on **free-tier compute**: training on Colab/Kaggle GPU, inference/dev on a local CPU machine.
- Be a portfolio-grade artifact: clean architecture, docs, tests, live demo.

### Non-Goals (v2.0)
- Beating published SOTA accuracy (requires large-scale GPU pretraining we don't have).
- Real-time streaming/telephony integration at scale.
- Commercial multi-tenant SaaS with billing.
- Non-speech audio (music, environmental sound) emotion.
- Clinical/diagnostic use — explicitly out of scope and disclaimed (research/education only).

---

## 3. Target Users

| Segment | Need | How they use it |
|---------|------|-----------------|
| ML students / learners | Understand the full SER pipeline end-to-end | Read docs, run notebooks, inspect benchmark harness |
| ML/AI recruiters & interviewers | Evaluate the candidate's production skills | Live demo, GitHub, architecture docs |
| Researchers | A reproducible cross-corpus benchmark baseline | Run harness, extend with new models/datasets |
| Product/analytics engineers | An API to add emotion signals to their app | Call `/predict`, read OpenAPI docs |
| Data annotators / QA | Sanity-check model behavior on samples | Upload audio in the UI, review probabilities |

---

## 4. User Personas

**Persona 1 — Aisha, ML Student (primary).**
Final-year CS student building a portfolio. Wants to *learn* the full lifecycle and
show it off. Values clear docs, a working demo, and code she can explain in an
interview. Runs training on Kaggle (has no GPU locally), does dev on a laptop.
*Success = she can reproduce the benchmark and articulate every design decision.*

**Persona 2 — Ravi, Hiring Engineer (primary evaluator).**
Senior engineer reviewing candidates. Spends ~5 minutes on the demo and skims the
architecture doc. Looks for: honest evaluation, real serving, tests, containerization.
*Success = within 5 minutes he concludes "this person can ship."*

**Persona 3 — Dr. Meera, Researcher (secondary).**
Wants a fair baseline to compare her new model against. Needs speaker-independent
splits and cross-corpus numbers she can trust and extend.
*Success = she adds a model to the registry and gets a comparable leaderboard row.*

**Persona 4 — Sam, Product Engineer (secondary).**
Integrating an emotion signal into a support-analytics dashboard. Doesn't care how the
model works — needs a stable, documented API and predictable latency.
*Success = one API call returns emotion + confidence in <1s for a short clip.*

---

## 5. Functional Requirements

IDs are referenced by user stories (§7) and the roadmap ([`07-implementation-roadmap.md`](07-implementation-roadmap.md)).

### Data
- **FR-D1** Ingest and normalize multiple datasets: RAVDESS, TESS, SAVEE, CREMA-D, AESDD, IEMOCAP (licence-gated).
- **FR-D2** Harmonize heterogeneous labels to a shared taxonomy (configurable class set).
- **FR-D3** Produce **speaker-independent** train/val/test splits, persisted and versioned.
- **FR-D4** Support audio augmentation (noise, pitch shift, time stretch) toggled by config.
- **FR-D5** Extract and cache features (MFCC, mel-spectrogram, chroma; SSL embeddings) to object storage.

### Models & Training
- **FR-M1** Support classical models: SVM, Random Forest, XGBoost.
- **FR-M2** Support deep models: CNN, LSTM, BiLSTM.
- **FR-M3** Support transformer models: Distil-HuBERT (fine-tune), HuBERT / Wav2Vec2 / emotion2vec (frozen-embedding + head).
- **FR-M4** All training driven by declarative config (no code change to run a new experiment).
- **FR-M5** Training runs on Colab/Kaggle with checkpoint/resume support.
- **FR-M6** Every training run logs config, metrics, and artifacts to the experiment tracker.

### Evaluation
- **FR-E1** Unified benchmark harness runs all models under identical splits & metrics.
- **FR-E2** Report accuracy, weighted/unweighted accuracy, precision, recall, F1 (macro + per-class), confusion matrix.
- **FR-E3** Cross-corpus evaluation: train on set A, test on held-out set B (e.g., CREMA-D).
- **FR-E4** Produce a single leaderboard comparing all models fairly.

### Serving
- **FR-S1** REST API `/predict` accepts an audio file, returns top emotion + full probability distribution + latency.
- **FR-S2** Model registry: select the serving model by name/version at request or config time.
- **FR-S3** Batch prediction endpoint for multiple files.
- **FR-S4** Health/readiness endpoints.
- **FR-S5** Interactive UI: upload or record audio → prediction, probabilities bar chart, waveform + spectrogram.

### Platform
- **FR-P1** AuthN/AuthZ: API-key or JWT auth; role separation (user vs admin).
- **FR-P2** Persist users, uploads, predictions, experiments, model versions, metrics, training runs, audit logs.
- **FR-P3** Structured request/prediction logging.
- **FR-P4** Basic monitoring: latency, throughput, error rate, prediction distribution.
- **FR-P5** Admin endpoints: promote/retire model versions, inspect audit logs.
- **FR-P6** Full stack runs via `docker compose` locally.

---

## 6. Non-Functional Requirements

| ID | Category | Requirement | Rationale / target |
|----|----------|-------------|--------------------|
| NFR-1 | Performance | p95 inference latency < 1s for ≤10s clip on CPU (classical/CNN); < 3s for transformer | Interactive UX; honest about transformer cost |
| NFR-2 | Throughput | ≥ 10 req/s single API instance (classical model) | Demo-scale, not web-scale |
| NFR-3 | Reproducibility | Any benchmark number reproducible from a config + seed + data version | Core differentiator |
| NFR-4 | Portability | Training portable to Colab/Kaggle; serving runs on 8GB-RAM CPU laptop | Free-tier constraint |
| NFR-5 | Maintainability | Type-hinted, linted, ≥80% test coverage on core `ml`/`inference`/`backend` logic | Portfolio quality |
| NFR-6 | Observability | Every request has a correlation ID in structured logs | Debuggability |
| NFR-7 | Security | No secrets in repo; auth on write/admin endpoints; input validation & file-type/size limits | Baseline safety |
| NFR-8 | Portability of models | Models serialized in a documented, versioned format with metadata | Registry integrity |
| NFR-9 | Availability | Graceful degradation: if transformer model unavailable, API returns a clear error, never crashes | Robustness |
| NFR-10 | Documentation | OpenAPI for API; README + architecture + model card for the project | Usability |

---

## 7. User Stories

Format: *As a `<persona>`, I want `<capability>` so that `<benefit>`.* (FR links in brackets.)

- **US-1** As Aisha, I want to run one command to reproduce the full benchmark so that I can trust and explain the numbers. [FR-E1, FR-E4, NFR-3]
- **US-2** As Aisha, I want training to resume after a Kaggle session times out so that I don't lose progress. [FR-M5]
- **US-3** As Ravi, I want to upload a `.wav` in a UI and see the predicted emotion with confidences so that I can judge the product in 60 seconds. [FR-S1, FR-S5]
- **US-4** As Ravi, I want to see cross-corpus results so that I know the model isn't just memorizing one dataset. [FR-E3]
- **US-5** As Dr. Meera, I want to register a new model via config and get a comparable leaderboard row so that I can benchmark my method fairly. [FR-M4, FR-E4]
- **US-6** As Sam, I want a documented `/predict` endpoint returning JSON probabilities so that I can integrate it into my dashboard. [FR-S1, NFR-10]
- **US-7** As Sam, I want to pick which model version serves my request so that I can pin behavior. [FR-S2]
- **US-8** As an admin, I want to promote a model version to "production" so that the API serves the vetted model. [FR-P5]
- **US-9** As an admin, I want an audit trail of who changed what so that the system is accountable. [FR-P2, FR-P5]
- **US-10** As any user, I want my invalid/oversized upload rejected with a clear message so that I understand what went wrong. [NFR-7, NFR-9]

---

## 8. Acceptance Criteria (product-level)

The product is "done" for v2.0 when **all** of the following hold:

1. `docker compose up` brings up API + UI + DB + tracking UI locally with one command.
2. A single documented command reproduces the benchmark leaderboard from configs + versioned data.
3. The leaderboard contains **≥6 models** (≥3 classical, ≥1 deep, ≥2 transformer) evaluated under **speaker-independent** splits, plus **≥1 cross-corpus** result.
4. Speaker-independent targets met: **RAVDESS 8-class ≥83%**, and a documented cross-corpus number on CREMA-D (no minimum — honesty over score).
5. UI: uploading/recording a clip returns emotion + probability bar chart + spectrogram in the latency budget (NFR-1).
6. API: OpenAPI docs live; `/predict`, `/health`, auth, and admin promote/retire all functional and tested.
7. Every model in the leaderboard has an experiment-tracker run and a registry entry with metadata + metrics.
8. Core modules ≥80% test coverage; CI green on push.
9. README + architecture diagram + model card + live demo link present.

---

## 9. Success Metrics

### Product / portfolio metrics
- **Demo completion:** a first-time visitor can get a prediction in < 90s without reading code.
- **Reproducibility:** independent `git clone` + documented steps reproduce the leaderboard within ±1% accuracy.
- **Breadth:** ≥6 models × ≥3 datasets in the benchmark matrix; ≥1 cross-corpus pair.

### Model metrics (per the benchmark harness)
- Accuracy, Weighted Accuracy (WA), Unweighted Accuracy (UA), macro-F1, per-class F1, confusion matrix, ROC-AUC (one-vs-rest where meaningful).
- Cross-corpus accuracy drop (reported, not hidden).

### System metrics (monitoring)
- p50/p95 inference latency, throughput (req/s), error rate, memory footprint, prediction-class distribution over time.

---

## 10. Assumptions & Constraints

- **Compute:** GPU only via Colab/Kaggle (free tier, session limits ~9–12h, must checkpoint). Local machine is CPU, ≥8GB RAM.
- **Data:** RAVDESS/TESS/SAVEE/CREMA-D/AESDD are freely downloadable; IEMOCAP requires a licence request and may be deferred.
- **Language:** English-first; multilingual (EMO-DB German, AESDD Greek) is a stretch.
- **Storage:** object storage is MinIO locally (S3-compatible), swappable for real S3 in cloud deploy.
- **Ethics:** research/education use only; not for surveillance, hiring decisions, or clinical diagnosis. Stated prominently in README + UI.

---

## 11. Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| IEMOCAP licence delay | Lose a benchmark corpus | Med | Design corpus-agnostic; ship with RAVDESS+TESS+SAVEE+CREMA-D, add IEMOCAP later |
| Kaggle session timeout mid-train | Lost training time | High | Checkpoint/resume (FR-M5); prefer Distil-HuBERT / frozen heads (cheap) |
| Scope creep (15 sub-systems) | Never ships | High | Core-vs-Stretch split enforced in roadmap; ship core first |
| Cross-corpus accuracy looks "bad" | Perceived failure | Med | Frame honesty as the feature; document expected drop |
| Transformer inference too slow on CPU | Poor UX | Med | Default serving model = light (Distil-HuBERT/classical); transformer opt-in |

---

## 12. Future Roadmap (post-v2.0)

- **v2.1** Ensembles / stacking; dimensional emotion (arousal/valence/dominance) as a second output mode.
- **v2.2** Real-time microphone streaming; noise-robustness benchmark suite.
- **v2.3** Explainability (attention/saliency over spectrogram); model cards auto-generated per run.
- **v2.4** Multimodal (audio + transcript via Whisper) using MELD/IEMOCAP.
- **v2.5** Cloud deployment with autoscaling; multilingual expansion (AESDD, EMO-DB).
- **v3.0** Active-learning loop: low-confidence predictions flagged for human labeling, fed back into training.
