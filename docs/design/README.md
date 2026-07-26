# EmotionSense AI — Design Documentation (v2.0)

Design-phase deliverables. **No implementation code yet** — this set is complete enough
that coding can begin immediately after approval.

| # | Document | Covers |
|---|----------|--------|
| 01 | [Product Requirements (PRD)](01-PRD.md) | Problem, users, personas, FR/NFR, user stories, acceptance criteria, success metrics, future roadmap |
| 02 | [Technical Design (TDD)](02-TDD.md) | High/low-level architecture, data flow, sequence diagrams, tech choices, ADR trade-offs |
| 03 | [Database Schema](03-database-schema.md) | ER diagram + DDL: users, uploads, predictions, experiments, model_versions, metrics, training_runs, audit_logs |
| 04 | [REST API Specification](04-api-specification.md) | Every endpoint: URL, method, request, response, error codes |
| 05 | [Folder Structure](05-folder-structure.md) | Production monorepo layout + rationale |
| 06 | [Coding Standards](06-coding-standards.md) | Style, naming, git/branch/commit, logging, error handling, testing |
| 07 | [Implementation Roadmap](07-implementation-roadmap.md) | Phased plan: goal, deliverables, acceptance, dependencies, effort |
| 08 | [Design Review](08-design-review.md) | Complexity audit + "Lean Core" simplifications |
| 09 | [Architecture & Design Review](09-architecture-review.md) | Adversarial multi-persona pre-implementation gate: 6 critical issues, scores, MoSCoW, prioritized action list |

## Key architectural commitments

- **Train/serve separation** — GPU training on Colab/Kaggle produces artifacts; CPU serving consumes them. They couple only through the DB, object storage, and model registry.
- **Config over code** — new datasets/models/experiments are YAML, not code changes.
- **Honest evaluation is a subsystem** — speaker-independent splits + cross-corpus benchmarking are first-class, not scripts. This is the project's differentiator.
- **Feature spec travels with the model** — eliminates train/serve skew (the #1 SER production bug).

## Before implementation, read the Design Review

[Doc 08](08-design-review.md) recommends a **Lean Core** (3 datasets, 6 models, 5-service
compose stack, report-only training endpoints) that meets every acceptance criterion at
~30 vs ~40 engineering days. Adopt it at kickoff.

## Prerequisite phases (done)

Research & gap analysis: [`../research/`](../research/).
