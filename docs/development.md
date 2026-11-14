# Development and verification

Start with `make install`, then `make help`. Installation may access the network.
`make check` is the fast offline gate and the first CI job. `make slow` adds slow work:
result reproduction, figures, and the installed-wheel smoke. Both use the lockfile and run
without network access after installation. CI runs `make slow` on macOS and Linux with
Python 3.13.

## Ownership and focused changes

| Change | Authoritative module | Focused check |
| --- | --- | --- |
| Artifact format, integrity, publication | `grpo_decomp/eval/completions.py` | `uv run pytest tests/test_completions.py` |
| Generation metadata and backend lifecycle | `grpo_decomp/eval/generate.py` | `uv run pytest tests/test_generate.py` |
| Run attempts, config, validation split | `grpo_decomp/train/`, `eval/heldout.py` | `uv run pytest tests/test_launcher.py tests/test_provenance.py tests/test_heldout.py` |
| Comparison and replicate identity | `grpo_decomp/report/evidence.py` | `uv run pytest tests/test_comparison_contracts.py` |
| Report measurements and persisted invariants | `grpo_decomp/report/`, `stats/` | `uv run pytest tests/test_passk_seeds.py tests/test_control_seeds.py tests/test_seeds.py` |
| Registries and study registration | `grpo_decomp/registries.py`, `llm_grpo_gains/registration.py` | `uv run pytest tests/test_registry.py` |
| Countdown arithmetic | `llm_grpo_gains/data/countdown.py` | `uv run pytest tests/test_countdown_grammar.py tests/test_rewards.py` |
| Study inventory and reproduction | `llm_grpo_gains/workflows.py` | `make study-plan`, `make results-check` |
| Esme analysis and figures | `llm_grpo_gains/esme_analysis.py`, `figures.py` | `make reproduce-esme`, `make results-check` |
| Package and dependency boundary | `pyproject.toml`, `uv.lock` | `make imports`, `uv run pytest tests/test_packaging.py`, `make slow` |

Run `make check` after the focused checks pass. It includes lint, formatting, imports,
types, and the fast offline tests. Run `make slow` before a change to packaging or results.
Mark an offline test `slow` when it would push `make check` past about 15 seconds.
`make demo` exercises the installed CLI with committed completions and no model load.

The wheel test builds and installs into a temporary environment outside the checkout.
It uses locked cached dependencies. Missing caches fail with the failed command and output.
Run `make install` before offline verification. Do not convert missing prerequisites into skips.

## Enforced rules

[AGENTS.md](../AGENTS.md#enforced-rules) is the single list of enforced rules and their checks.
Agent review concentrates on scientific interpretation, compatibility, and operational assumptions.

## Compatibility and evidence

Legacy CompletionSets contain two files. New CompletionSets also contain a versioned manifest.
Readers preserve legacy support. Writers reject conflicting replacements and allow identical retries.
Malformed JSON errors identify the file and line where available.

Training uses `<arm>-seed<N>/attempts/<id>/`. Each attempt owns provenance, status, and checkpoints.
A run alias resolves exactly one completed attempt. Multiple completions require an explicit path.
This prevents a retry from silently changing the experimental unit. Attempts never become extra seeds.
A legacy run without status remains readable, but its execution identity stays unverified.

New reports record input hashes, method settings, grading identity, and known training lineage.
Comparisons allow unequal base and trained sample counts. They reject incompatible decoding settings.
Known training recipes must match across paired roles. The recipe digest excludes seed and reward.
Reward identity remains explicit and must match within each role.
Unknown lineage and prompt identity remain explicit limitations. Matching labels cannot prove missing history.

Historical aggregate-only results remain unchanged. Their rates can be validated, but missing raw
samples cannot be reconstructed. Student-t intervals may extend outside the probability range.
The report schema bounds rates and effects, but preserves those interval semantics.

Esme's historical result remains available through the explicit `--legacy` analysis mode.
The default analysis records three retrospective exploratory comparisons as one Holm family.
The corrected result is separate. Its intervals are marginal, not family-adjusted.
Coverage and reliability transitions remain descriptive measurements, not causal proof.

## External verification

`make test-integration` requires the `generate` extra and network access to Hugging Face.
Missing dependencies fail explicitly. It exercises dataset loaders and real transformers generation.

[runbook.md](runbook.md) gives the prerequisites for the local CUDA smoke, `make verify-gpu`.

A training release also needs a bounded real training run, final-checkpoint loading, cleanup checks,
and held-out reconstruction on the intended GPU stack. Modal needs an authorized deployment check,
including Volume persistence across container lifetimes. A local suite does not prove those behaviors.
Volume readers need a fresh mount or reload to see externally committed attempts.

Full published-result reproduction needs the original external CompletionSets under `runs/`.
`make study-validate` reports missing inputs and unavailable source diagnostics. A successful public
artifact check does not imply that those external analyses were rerun.
