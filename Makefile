.DEFAULT_GOAL := check
.NOTPARALLEL: check slow
.PHONY: help install sync-check fix lint imports typecheck test test-integration verify-gpu check slow verify demo figures results-check aggregate study-plan study-validate reproduce-esme

RUN := uv run --locked --offline --no-sync

help:  ## List supported developer and study commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the locked CPU development and analysis environment
	uv sync --locked --extra dev --extra eval

sync-check:  ## Verify the locked CPU environment without network access
	uv sync --locked --offline --extra dev --extra eval

fix:  ## Auto-format and apply safe lint fixes
	$(RUN) ruff check --fix . .agents/skills/verify-grpo-decomp/verify
	$(RUN) ruff format . .agents/skills/verify-grpo-decomp/verify

lint:  ## Lint and format-check with the locked formatter version
	$(RUN) ruff check . .agents/skills/verify-grpo-decomp/verify
	$(RUN) ruff format --check . .agents/skills/verify-grpo-decomp/verify

imports:  ## Enforce the harness layers and the one-way harness and study boundary
	$(RUN) lint-imports --no-cache --no-logo

typecheck:  ## Check source, scripts, tests, and the verify helper (basedpyright strict)
	$(RUN) basedpyright
	$(RUN) basedpyright --project .agents/skills/verify-grpo-decomp

test:  ## Run the fast offline CPU tests
	$(RUN) pytest

test-integration:  ## Run network and real transformers tests, with required dependencies
	@$(RUN) python -c "import importlib.util, sys; missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]; sys.exit('Missing integration dependencies: ' + ', '.join(missing) + '. Run uv sync --locked --extra dev --extra eval --extra generate') if missing else None"
	$(RUN) pytest -m 'integration and not gpu'

verify-gpu:  ## Run real local vLLM generation, fail when GPU prerequisites are absent
	$(RUN) pytest tests/integration/test_gpu.py -m gpu

check: sync-check lint imports typecheck test  ## Fast offline gate, the first CI job

slow: check results-check  ## Full offline gate: fast lane, result reproduction, installed wheel
	$(RUN) pytest -m slow

verify:  ## Drive the real CLI and Modal launcher on CPU (ARGS=all|list|control|<drive>)
	@$(RUN) python .agents/skills/verify-grpo-decomp/verify $(ARGS)

demo:  ## Score committed mini CompletionSets without loading a model
	$(RUN) grpo-decomp battery --completions tests/fixtures/mini/base__mini --k 1
	$(RUN) grpo-decomp battery --completions tests/fixtures/mini/correct-seed0__mini --k 1

figures:  ## Rewrite the committed figures from published result JSON
	$(RUN) python scripts/make_figures.py

results-check: reproduce-esme  ## Read-only: reproduce Esme results and compare figures in a temp dir
	$(RUN) python -m llm_grpo_gains.workflows figures-check

study-plan:  ## Show every study route, seed axis, output, and reproduction level
	$(RUN) grpo-study plan

study-validate:  ## Report all missing or invalid study reproduction inputs
	$(RUN) grpo-study validate

aggregate:  ## Rebuild external-completion aggregates from the study manifest
	$(RUN) grpo-study aggregate

reproduce-esme:  ## Reproduce the legacy Esme result exactly without changing published files
	$(RUN) grpo-study reproduce --workflow esme --workflow esme-corrected
