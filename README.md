# llm-grpo-gains

`llm-grpo-gains` is a controlled GRPO study. It trains small Qwen models with
verifiable rewards, then asks a stricter question than "did benchmark accuracy go up?":

**How much of the gain is new capability, and how much is reliability, formatting,
contamination, or a placebo effect?**

The answer on GSM8K is deliberately unglamorous: GRPO gives a real gain, but mostly by
making the model more reliable on problems it could already solve. The same measurement
detects genuine expansion on Countdown, a generated search task where the base model has
real headroom.

## Headline Result

On GSM8K, the correctness reward beats a random-reward placebo by **+3.9 pp, 95% CI
[2.3, 5.6]** across six seeds. That is a statistically clear improvement, not just
formatting or noise. It is also not much evidence of new reasoning.

- **The base already has the coverage.** base pass@8 (94.0%) is above correct pass@1
  (76.2%).
- **Coverage barely moves.** correct pass@8 changes by Δ +0.7 pp, with propagated
  95% CI [-0.4, +1.9].
- **Per problem, 0.0% of the GSM8K gain is new capability.** The trained model makes
  reachable problems more reliable; it does not solve a new slice outside the base
  pass@8 envelope.
- **The envelope survives decontamination.** On renumbered GSM-Symbolic problems,
  base pass@8 holds at 90.8%.
- **CoT-gated pass@k is uninformative here.** Qwen has 0.0% verifiable-chain coverage
  in these completions, so the `<<a op b = c>>` proxy never fires.

![GSM8K decomposition](results/fig-gsm8k-decomposition.svg)

Countdown is the control that keeps this from being a null result dressed up as a
method. With the same protocol, the base lacks coverage and GRPO expands what it can
solve: the placebo comparison is **+46.5 pp, 95% CI [21.4, 71.6]**, and pass@8 moves
by **Δ +41.0 pp, 95% CI [38.3, 43.7]**: base 53.6% → correct 94.6%. Per problem,
**10.9% on Countdown** is genuinely new capability.

![GSM8K vs Countdown](results/fig-task-contrast.svg)

The committed findings are in [`results/FINDINGS.md`](results/FINDINGS.md), with the
Countdown panel in [`results/countdown/FINDINGS.md`](results/countdown/FINDINGS.md) and
the decontamination panel in [`results/decontam/FINDINGS.md`](results/decontam/FINDINGS.md).

## What This Repository Contains

The repo is the full measurement stack for the study:

- deterministic data loaders for GSM8K, GSM-Symbolic, GSM-Plus, GSM8K-Platinum, and
  generated Countdown;
- verifiable rewards for math correctness, Countdown correctness, and a seeded random
  placebo;
- a GRPO training launcher and Modal entrypoint for single-GPU runs;
- `CompletionSet` artifacts that separate model generation from offline analysis;
- an eval battery for strict/lenient grading, pass@k, CoT-gated pass@k, and simple
  behavior detectors;
- paired statistics: bootstrap confidence intervals, McNemar tests, seed aggregation,
  and Holm correction for controls;
- committed result JSON, figures, and documentation consistency tests.

Trained checkpoints and full completion sets are not committed. They live on the Modal
`assay-runs` volume. The checked-in JSON under [`results/`](results/) is enough to
verify the published numbers and rebuild the figures.

## How The Measurement Works

Training and analysis are deliberately separated.

1. Train one arm: base model, reward, seed, and dataset are recorded in provenance.
2. Generate completions from the base model and trained checkpoints.
3. Freeze those completions as `CompletionSet` directories.
4. Run all grading, statistics, and reporting offline on CPU.

Only generation needs a model backend. Once a `CompletionSet` exists, the analysis is
deterministic and does not need a GPU, network access, or Hugging Face.

The architecture is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
The training procedure is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Install

```bash
make install
```

This installs the CPU environment: data, rewards, eval, stats, reports, tests, and
developer tools.

Optional extras:

```bash
uv sync --extra generate  # transformers backend for local CPU/MPS generation
uv sync --extra train     # GPU training stack: TRL + vLLM + wandb
```

## Verify The Checked-In Results

```bash
make results
```

This rebuilds the figures from `results/*.json` and checks that headline README and
findings numbers trace back to committed artifacts. It does not load a model, download a
dataset, use Modal, or require a GPU.

Run the normal project gate:

```bash
make check
```

That runs Ruff, unit tests, and the docs-to-JSON consistency guard.

## Try The Eval Path Without A GPU

`make demo` scores two tiny committed `CompletionSet` fixtures:

```bash
make demo
```

Expected: base `strict_accuracy = 0.3333333333333333`; correct `strict_accuracy = 0.5`.
Both fixtures have 12 problems and 4 samples per problem. No model is loaded.

To generate a small local completion set with `transformers`:

```bash
uv sync --extra generate
grpo-decomp generate \
  --model Qwen/Qwen2.5-Math-1.5B \
  --set dev \
  --backend transformers \
  --out runs/base__dev
grpo-decomp battery --completions runs/base__dev --k 1
```

## CLI

The `grpo-decomp` CLI exposes the artifact boundary directly:

- `generate`: load a model and write a `CompletionSet`;
- `battery`: score a `CompletionSet`;
- `report`: build the single-seed decomposition table;
- `report-seeds`, `report-passk-seeds`, `report-mechanism`, `report-control-seeds`:
  aggregate the multi-seed panels;
- `heldout`: select checkpoints from held-out accuracy, not reward curves.

Example report generation from local completion directories:

```bash
grpo-decomp report --completions-dir runs/ --out results/
```

The expected layout is `<arm>__<set>`, for example `base__gsm8k-test` and
`correct__gsm-symbolic`.

## Training

Training runs on Modal with a single A100/H100-class GPU. The Modal app records
provenance, checkpoints, held-out curves, and generated completion sets on the
`assay-runs` volume.

```bash
modal run --detach modal_app.py --arm configs/correct.yaml
modal run --detach modal_app.py --arm configs/correct.yaml --command heldout
```

The full procedure, including Modal auth, the W&B secret, smoke runs, Volume pulls, and
JSON regeneration from completions, is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Reproducibility Rules

The study is built around a few constraints:

- report no headline gain without controls, confidence intervals, and paired tests;
- aggregate over seeds before making a claim;
- keep dataset revisions, model revisions, config, commit, dependency versions, and
  sampling settings in artifacts;
- treat reward curves as debugging signals, not evidence;
- make skipped records, malformed artifacts, and unparseable completions visible.

## Stack

Python 3.11+ · `uv` · Pydantic · TRL GRPO · transformers · vLLM · Modal ·
[`eval-audit`](https://github.com/adamthuvesen/eval-audit)

## License

MIT
