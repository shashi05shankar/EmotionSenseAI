.PHONY: help install install-all lint format type test test-all cov \
        splits benchmark up down seed clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install core + dev (CPU, light)
	pip install -e ".[dev]"

install-all:  ## Install every extra (adds torch/transformers/backend)
	pip install -e ".[dev,ml,deep,transformer,backend,frontend]"

lint:  ## Ruff + black check
	ruff check src tests
	black --check src tests

format:  ## Auto-format
	ruff check --fix src tests
	black src tests

type:  ## mypy strict on src
	mypy src

test:  ## Fast unit+integration tests
	pytest -m "unit or integration"

test-all:  ## Everything except slow/network
	pytest -m "not slow"

cov:  ## Coverage report
	pytest -m "unit or integration" --cov --cov-report=term-missing

splits:  ## Build speaker-independent k-fold splits from configs
	python scripts/build_splits.py

benchmark:  ## Run the unified benchmark harness -> leaderboard
	python scripts/run_benchmark.py --experiment configs/experiments/baseline_suite.yaml

up:  ## Start the full local stack (backend, frontend, postgres, minio, mlflow)
	docker compose -f deployment/compose/docker-compose.yml up --build

down:  ## Stop the stack
	docker compose -f deployment/compose/docker-compose.yml down

seed:  ## Seed DB with an admin user + register default model
	python scripts/seed_db.py

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
