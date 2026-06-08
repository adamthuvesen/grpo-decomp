.DEFAULT_GOAL := check
.PHONY: install fmt lint test test-integration check

install:  ## Sync the dev environment (incl. the CPU eval/stats layer)
	uv sync --extra dev --extra eval

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run unit tests (network tests deselected)
	uv run pytest

test-integration:  ## Run network/HuggingFace integration tests
	uv run pytest -m integration

check: lint test  ## The Phase 0 gate: lint + unit tests
