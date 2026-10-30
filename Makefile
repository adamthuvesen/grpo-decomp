.DEFAULT_GOAL := check
.NOTPARALLEL: check
.PHONY: help install sync-check fmt lint imports typecheck test test-integration verify-gpu gpu-plan check demo results results-check aggregate study-plan study-validate test-docs reproduce-esme

RUN := uv run --locked --offline --no-sync

help:  ## List supported developer and study commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the locked CPU development and analysis environment
	uv sync --locked --extra dev --extra eval

sync-check:  ## Verify the locked CPU environment without network access
	uv sync --locked --offline --extra dev --extra eval

fmt:  ## Auto-format and apply safe lint fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

lint:  ## Lint and format-check with the locked formatter version
	$(RUN) ruff check .
	$(RUN) ruff format --check .

imports:  ## Enforce the one-way harness and study import boundary
	$(RUN) lint-imports --no-cache --no-logo

typecheck:  ## Check source-package type annotations
	$(RUN) mypy

test:  ## Run all offline CPU tests, including the installed-wheel smoke
	$(RUN) pytest

test-integration:  ## Run network and real transformers tests, with required dependencies
	@$(RUN) python -c "import importlib.util, sys; missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]; sys.exit('Missing integration dependencies: ' + ', '.join(missing) + '. Run uv sync --locked --extra dev --extra eval --extra generate') if missing else None"
	$(RUN) pytest -m 'integration and not gpu'

gpu-plan:  ## Show the explicit local GPU smoke prerequisites without starting a job
	@printf '%s\n' 'GPU smoke: local CUDA hardware, the locked train extra, and a local model directory.' 'Run: GRPO_GPU_MODEL=/absolute/local/model make verify-gpu' 'This verifies real vLLM generation and grading. It does not verify training or launch Modal jobs.' 'Training jobs and paid hardware require the explicit scope in docs/runbook.md.'

verify-gpu:  ## Run real local vLLM generation, fail when GPU prerequisites are absent
	$(RUN) pytest tests/integration/test_gpu.py -m gpu

check: sync-check lint imports typecheck test results-check  ## Complete offline CPU gate, shared with CI

test-docs:  ## Check published documentation against committed result JSON
	$(RUN) pytest tests/test_docs_consistency.py

demo:  ## Score committed mini CompletionSets without loading a model
	$(RUN) grpo-decomp battery --completions tests/fixtures/mini/base__mini --k 1
	$(RUN) grpo-decomp battery --completions tests/fixtures/mini/correct-seed0__mini --k 1

results:  ## Regenerate figures from published JSON and verify documented numbers
	$(RUN) python scripts/make_figures.py
	$(RUN) pytest tests/test_docs_consistency.py

results-check: reproduce-esme  ## Compare committed Esme results and regenerate figures in a temporary directory
	$(RUN) python -m llm_grpo_gains.workflows figures-check

study-plan:  ## Show every study route, seed axis, output, and reproduction level
	$(RUN) grpo-study plan

study-validate:  ## Report all missing or invalid study reproduction inputs
	$(RUN) grpo-study validate

aggregate:  ## Rebuild external-completion aggregates from the study manifest
	$(RUN) grpo-study aggregate

reproduce-esme:  ## Reproduce the legacy Esme result exactly without changing published files
	$(RUN) grpo-study reproduce --workflow esme --workflow esme-corrected
