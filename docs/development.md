# Development and verification

Start with `make install`, then `make help`. Installation may access the network.
`make check` uses the lockfile and runs without network access after installation.
CI calls the same command on macOS and Linux with Python 3.13 and 3.14.

## Ownership and focused changes

| Change | Authoritative module | Focused check |
| --- | --- | --- |
| Artifact format, integrity, publication | `grpo_decomp/eval/completions.py` | `uv run pytest tests/test_completions.py` |
| Generation metadata and backend lifecycle | `grpo_decomp/eval/generate.py` | `uv run pytest tests/test_generate.py` |
| Run attempts, config, validation split | `grpo_decomp/train/`, `eval/heldout.py` | `uv run pytest tests/test_launcher.py tests/test_provenance.py tests/test_heldout.py` |
| Comparison and replicate identity | `grpo_decomp/report/evidence.py` | `uv run pytest tests/test_comparison_contracts.py` |
| Report measurements and persisted invariants | `grpo_decomp/report/`, `stats/` | `uv run pytest tests/test_passk_seeds.py tests/test_control_seeds.py tests/test_seeds.py` |
| Plugin ownership and rollback | `grpo_decomp/registries.py`, `plugins.py` | `uv run pytest tests/test_registry.py tests/test_plugins.py` |
| Countdown arithmetic | `llm_grpo_gains/data/countdown.py` | `uv run pytest tests/test_countdown_grammar.py tests/test_rewards.py` |
| Study inventory and reproduction | `llm_grpo_gains/workflows.py` | `make study-plan`, `make results-check` |
| Esme analysis and figures | `llm_grpo_gains/esme_analysis.py`, `figures.py` | `make reproduce-esme`, `make results-check` |
| Package and dependency boundary | `pyproject.toml`, `uv.lock` | `make imports`, `uv run pytest tests/test_packaging.py tests/test_architecture.py` |

Run `make check` after the focused checks pass. It includes lint, formatting, imports,
types, offline tests, installed-wheel verification, public-result reproduction, and figures.
`make demo` exercises the installed CLI with committed completions and no model load.

The wheel test builds and installs into a temporary environment outside the checkout.
It uses locked cached dependencies. Missing caches fail with the failed command and output.
Run `make install` before offline verification. Do not convert missing prerequisites into skips.

## Enforcement choices

| Guarantee | Strongest mechanism | Why another check remains necessary |
| --- | --- | --- |
| One owner per artifact | Atomic publication and immutable destination | Filesystems can fail. Failure tests exercise interrupted publication and conflicts. |
| One identity per training attempt | Unique directories and terminal status records | Process failures remain possible. Lifecycle tests verify failure states and cleanup. |
| Comparable experimental units | One `ComparisonInputs` constructor | External JSON is untyped. Boundary validation rejects drift and duplicate identities. |
| Consistent result values | Constrained models and derived-value validation | Historical JSON stores redundant fields. Validators preserve its format without trusting contradictions. |
| Harness never depends on study | One-way package design | Future imports can violate the design. Import-linter and isolated imports enforce the boundary. |
| Package works after installation | Package metadata and narrow entry points | Source tests cannot prove wheel contents or discovery. The isolated wheel test does. |
| Study commands remain reproducible | One typed workflow inventory | External inputs may be absent. Inventory validation reports every missing input before execution. |
| Correct statistical implementation | Shared estimators and report methods | Types cannot prove arithmetic or inference. Synthetic paired cases and committed Esme data verify behavior. |
| Accurate interpretation | Descriptive report output | Scientific claims still require judgment. Independent review assesses inference and evidence limits. |

Documentation explains ownership and choices. It does not replace these mechanisms.
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

`make gpu-plan` describes the local CUDA test. `GRPO_GPU_MODEL=/local/model make verify-gpu`
requires installed training dependencies, CUDA, and an existing model. It performs real vLLM
generation and grading. It neither provisions hardware nor verifies training.

A training release also needs a bounded real training run, final-checkpoint loading, cleanup checks,
and held-out reconstruction on the intended GPU stack. Modal needs an authorized deployment check,
including Volume persistence across container lifetimes. A local suite does not prove those behaviors.
Volume readers need a fresh mount or reload to see externally committed attempts.

Full published-result reproduction needs the original external CompletionSets under `runs/`.
`make study-validate` reports missing inputs and unavailable source diagnostics. A successful public
artifact check does not imply that those external analyses were rerun.
