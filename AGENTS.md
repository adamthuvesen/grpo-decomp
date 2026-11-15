# AGENTS.md: grpo-decomp + llm_grpo_gains

Two packages, one-way boundary: the `grpo_decomp` harness never imports the `llm_grpo_gains` study.
Read [docs/architecture.md](docs/architecture.md) for what goes where before you add code.

## Commands

```bash
make fix            # auto-format + safe lint fixes
make install        # CPU env: data, rewards, eval, stats, reports
make check          # fast offline gate: lock, lint, imports, types, fast tests (first CI job)
make slow           # check + result reproduction + installed-wheel smoke (CI matrix)
make demo           # score committed mini CompletionSets; no model load
make help           # focused checks and study commands

modal run --detach modal_app.py --arm configs/correct.yaml  # paid A100: only after Adam approves
grpo-decomp battery --completions runs/base__dev --k 1      # grade a CompletionSet (CPU)
```

Only `generate` and training need a model or GPU. Grading and reports run on CPU.

Prove changes with `make verify` (`verify-grpo-decomp` skill at `.agents/skills/verify-grpo-decomp/`).

## Enforced Rules

Ruff, basedpyright, and the Python version are configured in `pyproject.toml`. Approved runtime
dependencies are listed in `tests/test_packaging.py`; dev-only tools go in the `dev` extra.

| Rule | Enforced by |
| --- | --- |
| The harness never imports the study | `pyproject.toml::tool.importlinter`, `tests/test_architecture.py::test_harness_imports_without_a_study_or_gpu_stack` |
| Each harness module imports only lower layers: `cli`/`jobs` > `report` > `eval`/`train` > `rewards`/`prompts` > `registries` > `stats`/`splits`/`provenance`/`grading` > `schemas` | `pyproject.toml::tool.importlinter`, `Makefile::imports` |
| A paid Modal launch refuses to run without `--approved` | `modal_app.py::_require_approval`, `tests/test_modal_app.py::test_every_modal_entry_point_requires_approval`, `tests/test_modal_app.py::test_unapproved_launch_refuses_with_the_exact_approved_command` |
| No new runtime dependency without asking Adam | `tests/test_packaging.py::test_runtime_dependencies_are_the_approved_stack` |
| TRL and vLLM stay exact `==` pins | `tests/test_packaging.py::test_trl_and_vllm_are_exact_pins`, `uv.lock` |
| Fewer than three seeds is labeled preliminary | `src/grpo_decomp/report/status.py::MIN_HEADLINE_SEEDS`, `tests/test_passk_seeds.py::test_below_min_seeds_is_preliminary` |
| Result schemas reject unknown fields | `tests/test_schemas.py::test_problem_rejects_unknown_fields` |
| Artifacts record version and content hashes | `tests/test_completions.py::test_manifest_records_version_and_content_hashes` |
| Publication is atomic and never replaces a conflicting artifact | `tests/test_completions.py::test_conflicting_existing_artifact_is_never_replaced`, `tests/test_completions.py::test_failed_staged_write_cleans_up_and_does_not_publish` |
| Legacy CompletionSets stay readable | `tests/test_completions.py::test_legacy_artifact_load_and_retry_preserve_all_bytes` |
| A retry is never a new training seed | `tests/test_comparison_contracts.py::test_retry_is_not_a_new_training_seed` |
| Multiple completed attempts need explicit selection | `tests/test_provenance.py::test_multiple_successful_attempts_require_explicit_selection` |
| Historical result JSON stays byte-identical | `tests/test_esme_analysis.py::test_committed_legacy_summary_is_byte_identical`, `Makefile::reproduce-esme` |
| Documented numbers match committed results | `tests/test_docs_consistency.py` |
| The installed wheel registers the study and grades a fixture | `tests/test_packaging.py::test_installed_wheel_registers_study_grades_and_loads_packaged_fixture` |
| `make verify` reports a pass only when its control observes FAIL | `.agents/skills/verify-grpo-decomp/drives.py::control_battery`, `Makefile::verify` |
| Process, network, paid SDK, and Modal modules are imported only by named owner files, each held to its own modules | `pyproject.toml::tool.ruff.lint.flake8-tidy-imports.banned-api`, `tests/test_banned_modules.py::test_every_file_uses_only_the_banned_modules_it_owns`, `tests/test_banned_modules.py::test_only_named_owners_hold_a_tid251_ignore` |

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
