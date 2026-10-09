.PHONY: help install install-gpu test test-fast lint format typecheck baseline \
	data-ibm stats-ibm synthetic ladder gnn rung9 ablation drift elliptic replay api frontend \
	frontend-build db-up results clean

CONFIG ?= configs/ibm_aml.yaml

help:
	@echo "install       - editable install with dev + api extras"
	@echo "install-gpu   - CUDA PyTorch, then PyG + SHAP (see README for the CUDA tag)"
	@echo "test          - full test suite"
	@echo "test-fast     - skip slow tests"
	@echo "lint          - ruff check"
	@echo "format        - ruff format"
	@echo "typecheck     - mypy"
	@echo "baseline      - tabular baseline on generated demo data (sanity check)"
	@echo "data-ibm      - build data/processed/ from the IBM AML raw CSV + patterns"
	@echo "stats-ibm     - print the dataset-card statistics for IBM AML"
	@echo "synthetic     - generate the tier-3 synthetic dataset"
	@echo "ladder        - rungs 1-5 over 5 seeds + leakage canary"
	@echo "gnn           - rungs 6-8 (homogeneous, heterogeneous, temporal GNN)"
	@echo "rung9         - fusion + ring detection + early-warning time"
	@echo "ablation      - feature-group ablation table"
	@echo "drift         - drift experiment on the synthetic stream"
	@echo "elliptic      - tier-1 ladder on Elliptic++ (data/raw/elliptic/)"
	@echo "replay        - stream the test period into the database (alerts + evidence)"
	@echo "api           - serve the analyst API on :8000 (and the built dashboard)"
	@echo "frontend      - dashboard dev server on :5173 (proxies /api)"
	@echo "db-up         - start PostgreSQL in Docker (set NEXIS_DATABASE_URL after)"
	@echo "results       - aggregate results/ into docs/results.md"

install:
	pip install -e ".[dev,api]"

install-gpu:
	pip install torch --index-url https://download.pytorch.org/whl/cu126
	pip install -e ".[dev,api,gnn]"

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src/nexis

baseline:
	python scripts/run_baseline.py --demo

data-ibm:
	python -m nexis.data.ibm_aml --config $(CONFIG)

stats-ibm:
	python scripts/dataset_stats.py --config $(CONFIG)

synthetic:
	python -m nexis.data.synthetic --config configs/synthetic.yaml --out data/synthetic

ladder:
	python scripts/run_ladder.py --config $(CONFIG) --rungs 1 2 3 4 5 --canary

gnn:
	python scripts/run_gnn.py --config $(CONFIG)

rung9:
	python scripts/run_rung9.py --config $(CONFIG) --base xgb_graph gnn_temporal

ablation:
	python scripts/run_ablation.py --config $(CONFIG)

drift:
	python scripts/run_drift.py --config configs/synthetic.yaml

elliptic:
	python scripts/run_elliptic.py --raw data/raw/elliptic

replay:
	python scripts/replay.py --config $(CONFIG)

api:
	uvicorn nexis.api.app:app --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm install && npm run build

db-up:
	docker run -d --name nexis-pg -e POSTGRES_USER=nexis -e POSTGRES_PASSWORD=nexis \
		-e POSTGRES_DB=nexis -p 5432:5432 postgres:16

results:
	python scripts/results_table.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
