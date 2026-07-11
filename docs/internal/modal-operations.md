# Internal Modal Operations

These notes are for repo operators. They live outside the public runbook because
they cover launch choreography, failure modes, and debugging signals rather than
the ordinary reader path.

## Scope

Modal runs use the `assay-runs` Volume for checkpoints, provenance, held-out
curves, and generated completion sets. Training curves log to W&B through the
Modal secret named `wandb`.

The reference GSM8K and Countdown studies were run with the same launcher shape:
one GRPO arm per `configs/*.yaml`, followed by held-out scoring and completion
generation for the aggregate reports.

## Setup

```bash
uv tool install modal
modal setup
export WANDB_API_KEY="..."
modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"
```

Hugging Face auth is optional for the public Qwen and GSM8K assets. If rate
limits block runs, create a `huggingface` secret and add it to the Modal
functions in `modal_app.py`.

```bash
export HF_TOKEN="..."
modal secret create huggingface HF_TOKEN="$HF_TOKEN"
```

## Cheap Preflight

Use the short blocking run when checking infrastructure changes:

```bash
modal run modal_app.py --no-spawn --arm configs/correct.yaml --smoke-problems 8 --max-steps 5
```

Check `completions/clipped_ratio`: it must not stay pinned near `1.0` every step.
A persistent value near `1.0` means trainer EOS and vLLM stop-token handling are
out of sync, so gradients are likely masked. Occasional non-zero clipping from
long completions is expected.

## Durable Launch Pattern

The entrypoint defaults to `--spawn`, so long GPU work should be detached:

```bash
modal run --detach modal_app.py --arm configs/correct.yaml
modal run --detach modal_app.py --arm configs/correct.yaml --command heldout
modal run --detach modal_app.py --arm configs/random.yaml
modal run --detach modal_app.py --arm configs/random.yaml --command heldout
```

A plain attached long run can die with the local client even if W&B later shows
training finished. Monitor durable runs through Modal logs and the Volume:

```bash
modal app logs grpo-decomp
modal volume ls assay-runs
```

## Launch Hazards

- Do not launch several `modal run`s that need the same uncached image at once.
  Concurrent first-time image builds can race and fail with `build has not
  completed`.
- Do not burst many `modal run`s in a few seconds. App creation is rate-limited
  separately from GPU quota; stagger launches by roughly 30-45 seconds.
- Checkpoints are large: each saved checkpoint is roughly 3-9 GB including
  optimizer state.

## Generation And Aggregation Inputs

Pull committed-study completion directories before re-deriving JSON:

```bash
modal volume get assay-runs passk-multiseed           runs/passk-multiseed
modal volume get assay-runs passk-multiseed-countdown runs/passk-multiseed-countdown
modal volume get assay-runs battery                   runs/battery
for s in 1 2 3 4 5; do modal volume get assay-runs battery-seed$s runs/battery-seed$s; done
```

`passk-multiseed` carries the GSM8K test, GSM-Symbolic, and GSM8K-Platinum
pass@8 panels. `battery` holds the seed-independent base plus seed-0
correct/random arms. `battery-seed{1..5}` holds the remaining per-seed
correct/random arms for placebo and control aggregation.

## Tuning Notes

- `max_completion_length=1024`; 512 clips too many rollouts for this study.
- `vllm_gpu_memory_utilization=0.3` is the colocated default. Raise it if
  rollouts are slow; lower it before changing model weights if the job is OOMing.
- `max_steps=500` and `save_steps=100` are the committed production values.
- Comparability with the committed study is broken by changes to reward parsing or
  sampling settings.
