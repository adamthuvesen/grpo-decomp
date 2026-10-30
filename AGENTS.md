# AGENTS.md: grpo-decomp + llm_grpo_gains

This repo has **two packages and one boundary**:

- **`grpo_decomp`** (`src/grpo_decomp/`): the harness, a controls-first method for
  decomposing a GRPO benchmark gain into real reasoning vs. contamination, formatting, and
  elicitation. Task- and model-agnostic; knows nothing about GSM8K or Countdown. Tasks plug
  in through the registries in `grpo_decomp/registries.py`.
- **`llm_grpo_gains`** (`src/llm_grpo_gains/`): the reference study that exercises the
  harness: GSM8K (primary) plus a generated Countdown positive control. It supplies datasets,
  the `correct`/`countdown` rewards, configs, results, and a `registration.py` that wires
  itself into the harness via a `grpo_decomp.plugins` entry point.

The result is only as trustworthy as the controls. "Most of this gain was elicitation"
beats a confident, uncontrolled "RL works."

Put generic measurement machinery in `grpo_decomp` and anything GSM8K/Countdown-specific in
`llm_grpo_gains`. A new task-shaped knob is a registry entry in
`llm_grpo_gains/registration.py`, not an `if task == ...` in the harness. Add extension
points, not a framework.

## Commands

```bash
make install        # CPU env: data, rewards, eval, stats, reports
make check          # fast offline gate: lock, lint, imports, types, fast tests (first CI job)
make verify         # check + result reproduction + installed-wheel smoke (CI matrix)
make demo           # score committed mini CompletionSets; no model load
make help           # focused checks and study commands

modal run --detach modal_app.py --approved --arm configs/correct.yaml  # paid A100: only after Adam approves
grpo-decomp battery --completions runs/base__dev --k 1      # grade a CompletionSet (CPU)
```

Only `generate` and training need a model or GPU. Grading and reports run on CPU.

## Enforced Rules

| Rule | Enforced by | Added |
| --- | --- | --- |
| The harness never imports the study | `pyproject.toml::[tool.importlinter]`, `tests/test_architecture.py::test_harness_imports_without_a_study_or_gpu_stack` | 2026-11-05 |
| A paid Modal launch refuses to run without `--approved` | `modal_app.py::_require_approval`, `tests/test_modal_app.py::test_every_modal_entry_point_requires_approval`, `tests/test_modal_app.py::test_unapproved_launch_refuses_with_the_exact_approved_command` | 2026-11-06 |
| No new runtime dependency without asking Adam | `tests/test_packaging.py::test_runtime_dependencies_are_the_approved_stack` | 2026-11-06 |
| TRL and vLLM stay exact `==` pins | `tests/test_packaging.py::test_trl_and_vllm_are_exact_pins`, `uv.lock` | 2026-11-06 |
| Fewer than three seeds is labeled preliminary | `src/grpo_decomp/report/status.py::MIN_HEADLINE_SEEDS`, `tests/test_passk_seeds.py::test_below_min_seeds_is_preliminary` | 2026-07-01 |
| Result schemas reject unknown fields | `tests/test_schemas.py::test_problem_rejects_unknown_fields` | 2026-07-01 |
| Artifacts record version and content hashes | `tests/test_completions.py::test_manifest_records_version_and_content_hashes` | 2026-11-05 |
| Publication is atomic and never replaces a conflicting artifact | `tests/test_completions.py::test_conflicting_existing_artifact_is_never_replaced`, `tests/test_completions.py::test_failed_staged_write_cleans_up_and_does_not_publish` | 2026-11-05 |
| Legacy CompletionSets stay readable | `tests/test_completions.py::test_legacy_artifact_load_and_retry_preserve_all_bytes` | 2026-11-05 |
| A retry is never a new training seed | `tests/test_comparison_contracts.py::test_retry_is_not_a_new_training_seed` | 2026-11-05 |
| Multiple completed attempts need explicit selection | `tests/test_provenance.py::test_multiple_successful_attempts_require_explicit_selection` | 2026-11-05 |
| Historical result JSON stays byte-identical | `tests/test_esme_analysis.py::test_committed_legacy_summary_is_byte_identical`, `Makefile::reproduce-esme` | 2026-10-24 |
| Documented numbers match committed results | `tests/test_docs_consistency.py` | 2026-07-01 |
| The installed wheel discovers the plugin | `tests/test_packaging.py::test_installed_wheel_discovers_plugin_grades_and_loads_packaged_fixture` | 2026-11-05 |
| This table cites only real enforcers | `tests/test_agents_rules.py::test_enforced_rules_cite_existing_paths_and_symbols` | 2026-11-06 |

Ruff, mypy, and the Python version are configured in `pyproject.toml`. Approved runtime
dependencies are listed in `tests/test_packaging.py`; dev-only tools go in the `dev` extra.

## Judgment calls

- **Controls before gains.** Never report a gain without its controls, a confidence
  interval, and a paired significance test.
- **The reward curve is not evidence.** Only held-out eval, completion length, and entropy
  are. Treat a rising reward with suspicion.
- **Never claim from one run.** Aggregate over seeds.
- Re-check TRL's GRPO defaults whenever you bump the TRL pin. vLLM must stay inside the
  range that TRL version supports.
- No broad RL or agent frameworks. The stack is TRL + transformers + vLLM + datasets
  (train) and `eval-audit` (stats).
- Published aggregate-only results cannot recover missing raw evidence or lineage. Record
  corrections in separate versioned results.
- If a doc disagrees with code, fix the doc in the same change.

The rationale lives in
[`docs/architecture.md`](docs/architecture.md#design-rules-that-shaped-the-code).

## Pointers

| Topic | Doc |
| --- | --- |
| Architecture, data flow, module map, Modal execution | [docs/architecture.md](docs/architecture.md) |
| Plug in your own model and task | [docs/architecture.md](docs/architecture.md#plug-in-your-own-model-and-task) |
| Which module owns a change, and its focused check | [docs/development.md](docs/development.md) |
| Training a GRPO arm on Modal, reproducing results | [docs/runbook.md](docs/runbook.md) |
