.PHONY: help install test test-fast lint format typecheck baseline clean

help:
	@echo "install    - editable install with dev extras"
	@echo "test       - full test suite"
	@echo "test-fast  - skip slow tests"
	@echo "lint       - ruff check"
	@echo "format     - ruff format"
	@echo "typecheck  - mypy"
	@echo "baseline   - run the tabular baseline experiment on demo data"

install:
	pip install -e ".[dev]"

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
