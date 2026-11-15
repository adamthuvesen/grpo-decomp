# Runbook

This is the public runbook for installing the project, checking the committed
results, and running CPU evaluation.

The published GPU runs used Modal for training/generation and W&B for training
curves. The public reproducibility path uses the committed JSON and figures, not
the raw run directories or checkpoints.

## Install

```bash
make install
```

This syncs the CPU environment, including dev tools, eval, and statistics code.

Optional extra for local generation:

```bash
uv sync --locked --extra dev --extra eval --extra generate  # local transformers generation
```

## GPU Runs On Modal

Training and generation run on Modal (one A100-80GB per arm) and log curves to W&B.
Every launch spends money, so each Modal entry point refuses to run without `--approved`.
Add the flag only after Adam approves the run.
`modal_app.py` imports the project, so run the Modal CLI from the project environment:

```bash
uv run modal setup
uv run modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"
uv run modal run --detach modal_app.py --arm configs/correct.yaml
```

Runs write checkpoints and completions to the `assay-runs` Volume. Launch several arms
about 40 seconds apart, since Modal rate-limits app creation.

## Local Checks

```bash
make slow
```

`make slow` runs the complete locked offline gate described in [development.md](development.md).
It includes types, import boundaries, the installed wheel, unit tests, public results, and figures.

To rewrite the committed figures from result JSON:

```bash
make figures
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


## Reproduce The Esme Result

```bash
make reproduce-esme
```

This recomputes the Esme sampled decomposition from the raw completions in
`results/esme-countdown/completions/` (6 seeds × correct/random arms at training step 240,
plus the base model, on 200 held-out tasks). It writes the summary to a temporary directory and fails if it differs from
`results/esme-countdown/sampled_multiseed_summary.json`. CI runs it on every push. It needs
no GPU or network. It also checks the separate corrected exploratory result.

## Generate And Grade Locally

```bash
uv sync --locked --extra dev --extra eval --extra generate
grpo-decomp generate \
  --model Qwen/Qwen2.5-Math-1.5B \
  --set dev \
  --backend transformers \
  --out runs/base__dev
grpo-decomp battery --completions runs/base__dev --k 1
```

Generation writes a `CompletionSet` with the sampled answers, source
problems, and provenance. `battery` grades that artifact offline.

## Reproduce Published Results

The committed JSON and figures can be checked locally without rewriting them:

```bash
make install
make results-check
```

To re-derive JSON, place matching `CompletionSet` directories under `runs/` and
run the aggregate commands:

```bash
make aggregate
uv run grpo-decomp report-seeds --task-set gsm8k-test \
  --battery-dirs runs/battery runs/battery-seed{1,2,3,4,5} \
  --out results/seed-placebo-comparison.json
make figures
make check   # includes the docs <-> JSON consistency test
```

`make aggregate` rebuilds the pass@k, mechanism, control, Countdown, same-base
Countdown, off-policy placebo, and decontamination aggregate files from local `runs/`
inputs. The committed repo includes the derived JSON and figures, not the private run
storage or full training outputs.

## Reproduce The Pre-Registered Follow-Ups

The two follow-ups in `results/preregistration-2026-10.md` train six arms and generate two
eval sets. Launch the training runs about 40 seconds apart (Modal rate-limits app creation):

```bash
uv run modal run --detach modal_app.py --arm configs/offpolicy-random.yaml
uv run modal run --detach modal_app.py --arm configs/offpolicy-random-seed1.yaml
uv run modal run --detach modal_app.py --arm configs/offpolicy-random-seed2.yaml
uv run modal run --detach modal_app.py --arm configs/countdown-math-correct.yaml
uv run modal run --detach modal_app.py --arm configs/countdown-math-correct-seed1.yaml
uv run modal run --detach modal_app.py --arm configs/countdown-math-correct-seed2.yaml
```

When each run resolves to one completed attempt, generate the eval sets.
A final directory alone does not establish completion for new attempts. Seed 0 of the off-policy
battery also generates the base model:

```bash
uv run modal run --detach modal_app.py --command battery --task gsm8k-offpolicy --scope random-vs-base --seed 0
uv run modal run --detach modal_app.py --command battery --task gsm8k-offpolicy --scope random-vs-base --seed 1
uv run modal run --detach modal_app.py --command battery --task gsm8k-offpolicy --scope random-vs-base --seed 2
uv run modal run --detach modal_app.py --command passk-multiseed --task countdown-math --correct-seeds 0,1,2
```

Pull them into the layout `make aggregate` reads. Seed labels come from the directory
names (`battery` is seed 0, `battery-seedN` is seed N), so rename the off-policy dirs:

```bash
mkdir -p runs/offpolicy
uv run modal volume get assay-runs battery-offpolicy runs/offpolicy/
uv run modal volume get assay-runs battery-offpolicy-seed1 runs/offpolicy/
uv run modal volume get assay-runs battery-offpolicy-seed2 runs/offpolicy/
mv runs/offpolicy/battery-offpolicy runs/offpolicy/battery
mv runs/offpolicy/battery-offpolicy-seed1 runs/offpolicy/battery-seed1
mv runs/offpolicy/battery-offpolicy-seed2 runs/offpolicy/battery-seed2
uv run modal volume get assay-runs passk-multiseed-countdown-math runs/
```

Both result files store per-problem data, so their intervals can also be recomputed from the
committed JSON alone.

## Operational Constraints

- Use held-out accuracy, completion length, and entropy to assess training.
  Reward curves are diagnostics, not evidence.
- Keep `configs/*.yaml` comparable to the committed study unless intentionally
  starting a new study: `max_steps=500`, `save_steps=100`,
  `max_completion_length=1024`, `num_generations=8`.
- `vllm` is pinned to `==0.17.1`, the top of TRL 1.0.0's supported range.
- The correctness reward uses strict `\boxed{}` extraction. Changing the reward
  starts a new study; it is not a comparable continuation.

## Attempt selection and reproduction limits

New executions write `<arm>-seed<N>/attempts/<id>/`. Failed retries preserve earlier attempts.
Use `grpo-decomp heldout --run /path/to/attempt` to select a specific completed execution.
Generation accepts `/path/to/attempt/checkpoints/final` as `--model` and records its lineage.
When several attempts completed, automatic arm/seed selection fails with the available paths.
Choose the intended execution before generation. Do not count retries as independent seeds.

`make study-plan` lists every published route, seed axis, and required evidence.
`make study-validate` reports all missing inputs before `make aggregate` writes outputs.
`make results-check` verifies public artifacts in temporary directories without rewriting them.
External CompletionSets, GPU training, and Modal persistence require separate verification.
The local GPU smoke needs CUDA, the locked `train` extra, and a local model directory:
`GRPO_GPU_MODEL=/absolute/local/model make verify-gpu`. It checks real vLLM generation and
grading. It does not verify training or launch Modal jobs.
See [development.md](development.md#external-verification) for exact boundaries.
