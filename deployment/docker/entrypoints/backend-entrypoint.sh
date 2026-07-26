#!/usr/bin/env bash
set -euo pipefail

# In the DB-backed production path, run migrations before serving:
#   alembic upgrade head
# The Lean Core runnable path uses the filesystem registry and needs no migration.

exec uvicorn emotionsense.backend.main:app --host 0.0.0.0 --port 8000
