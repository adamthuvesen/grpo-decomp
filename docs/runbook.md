# Runbook

This is the public runbook for installing the project, checking the committed
results, running CPU evaluation, and launching a GRPO arm.

## Install

```bash
make install
```

This syncs the CPU environment, including dev tools and the eval/statistics
layer.

Optional extras:

```bash
uv sync --extra generate  # local transformers generation
uv sync --extra train     # GPU training stack
```

## Local Checks

```bash
make check
```

`make check` runs Ruff, unit tests, and the docs-to-JSON consistency guard for
the committed result numbers.

To rebuild public figures from committed JSON:

```bash
make results
```

## CPU Demo

```bash
make demo
```

This scores the tiny committed fixtures in `tests/fixtures/mini`. It does not
load a model or require network access.

Expected strict accuracy:

- base fixture: `0.3333333333333333`
- correct fixture: `0.5`

## Generate And Grade Locally

```bash
uv sync --extra generate
grpo-decomp generate \
  --model Qwen/Qwen2.5-Math-1.5B \
  --set dev \
  --backend transformers \
  --out runs/base__dev
grpo-decomp battery --completions runs/base__dev --k 1
```

Generation writes a `CompletionSet` containing the sampled answers, source
problems, and provenance. `battery` grades that artifact offline.

## Train On Modal

Prerequisites:

- Modal account and CLI authentication
- Weights & Biases secret named `wandb`

```bash
uv tool install modal
modal setup
export WANDB_API_KEY="..."
modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"
```

Launch one arm:

```bash
modal run --detach modal_app.py --arm configs/correct.yaml
```

Score held-out checkpoints for an arm:

```bash
modal run --detach modal_app.py --arm configs/correct.yaml --command heldout
```

The Modal app writes checkpoints, provenance, held-out curves, and generated
completion sets to the `assay-runs` Volume.

## Reproduce Published Results

The committed JSON and figures can be checked without Modal:

```bash
make install
make results
```

To re-derive JSON from completion artifacts, first pull the expected run
directories from the `assay-runs` Volume:

```bash
modal volume get assay-runs passk-multiseed           runs/passk-multiseed
modal volume get assay-runs passk-multiseed-countdown runs/passk-multiseed-countdown
modal volume get assay-runs battery                   runs/battery
for s in 1 2 3 4 5; do modal volume get assay-runs battery-seed$s runs/battery-seed$s; done
```

Then rebuild aggregate JSON:

```bash
make aggregate
uv run grpo-decomp report-seeds --task-set gsm8k-test \
  --battery-dirs runs/battery runs/battery-seed{1,2,3,4,5} \
  --out results/seed-placebo-comparison.json
uv run grpo-decomp report --completions-dir runs/battery --task-set gsm8k-test --out results/
make results
```

`make aggregate` rebuilds the pass@k, mechanism, control, Countdown, and
decontamination aggregate artifacts from local `runs/` inputs.

## Operational Constraints

- Use held-out accuracy, completion length, and entropy to assess training.
  Reward curves are diagnostics, not evidence.
- Keep `configs/*.yaml` comparable to the committed study unless intentionally
  starting a new study: `max_steps=500`, `save_steps=100`,
  `max_completion_length=1024`, `num_generations=8`.
- `vllm` is pinned to `==0.17.1`, the top of TRL 1.0.0's supported range.
- The correctness reward uses strict `\boxed{}` extraction. A reward change
  creates a new study, not a comparable continuation.
