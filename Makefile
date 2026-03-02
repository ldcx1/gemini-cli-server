.PHONY: install install-dev test test-unit test-e2e lint format run clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with development dependencies
	pip install -e ".[dev]"

test: ## Run all tests (excluding E2E)
	python3 -m pytest tests/ -v --ignore=tests/test_e2e.py

test-unit: ## Run unit tests only
	python3 -m pytest tests/ -v --ignore=tests/test_e2e.py -x

test-e2e: ## Run end-to-end tests (requires gemini CLI)
	python3 -m pytest tests/test_e2e.py -v -m e2e

test-cov: ## Run tests with coverage report
	python3 -m pytest tests/ -v --ignore=tests/test_e2e.py --cov=gemini_cli_server --cov-report=term-missing

lint: ## Run linting (ruff + mypy)
	ruff check gemini_cli_server/ tests/
	mypy gemini_cli_server/

format: ## Format code with black and ruff
	black gemini_cli_server/ tests/
	ruff check --fix gemini_cli_server/ tests/

run: ## Run the server
	python3 -m gemini_cli_server

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
