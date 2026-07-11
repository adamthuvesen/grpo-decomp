.DEFAULT_GOAL := check
.PHONY: install fmt lint typecheck test test-integration check demo results aggregate test-docs

install:  ## Sync the dev environment (incl. the CPU eval/stats layer)
	uv sync --extra dev --extra eval

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Check source-package type annotations
	uv run mypy

test:  ## Run unit tests (network tests deselected)
	uv run pytest

test-integration:  ## Run network/HuggingFace integration tests
	uv run pytest -m integration

check: lint typecheck test test-docs  ## Local gate: lint + types + unit tests + docs consistency

test-docs:  ## docs <-> JSON consistency for committed headline numbers
	uv run pytest tests/test_docs_consistency.py -q

demo:  ## Score committed mini CompletionSets; no model load, GPU, or network
	@printf 'base fixture:\n'
	uv run grpo-decomp battery --completions tests/fixtures/mini/base__mini --k 1
	@printf '\ncorrect fixture:\n'
	uv run grpo-decomp battery --completions tests/fixtures/mini/correct-seed0__mini --k 1

results:  ## Regenerate the committed figures from JSON, then verify docs <-> JSON consistency
	uv run --with matplotlib python scripts/make_figures.py
	uv run pytest tests/test_docs_consistency.py -q

aggregate:  ## Re-derive the committed JSON from local completions (needs runs/ pulled; see docs/runbook)
	uv run grpo-decomp report-passk-seeds  --completions-dir runs/passk-multiseed           --task-set gsm8k-test     --out results/pass8-multiseed.json
	uv run grpo-decomp report-passk-seeds  --completions-dir runs/passk-multiseed-countdown --task-set countdown-test  --out results/countdown/pass8-multiseed.json
	uv run grpo-decomp report-passk-seeds  --completions-dir runs/passk-multiseed           --task-set gsm-symbolic    --out results/decontam/pass8-symbolic.json
	uv run grpo-decomp report-passk-seeds  --completions-dir runs/passk-multiseed           --task-set gsm8k-platinum  --out results/decontam/pass8-platinum.json
	uv run grpo-decomp report-mechanism    --completions-dir runs/passk-multiseed           --task-set gsm8k-test     --out results/mechanism.json
	uv run grpo-decomp report-mechanism    --completions-dir runs/passk-multiseed-countdown --task-set countdown-test  --out results/countdown/mechanism.json
	uv run grpo-decomp report-control-seeds --battery-dirs runs/battery runs/battery-seed1 runs/battery-seed2 runs/battery-seed3 runs/battery-seed4 runs/battery-seed5 --out results/decomposition-multiseed.json
