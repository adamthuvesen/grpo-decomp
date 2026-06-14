# Training runbook: GRPO arms on Modal

The reusable procedure for training a GRPO arm on a single A100. The GSM8K placebo comparison
(`correct` + `random`, 6 seeds each) and the Countdown positive control (3 seeds each, on the
general base) were both run exactly this way; reuse it for a new dataset arm — e.g. MATH on the
general base — by swapping the configs and repeating. Each arm pair is ~a weekend on one A100,
~$100–250; everything GPU-independent is already built and tested, so training is the only paid
step.

**Prereqs:** a Modal account and a Weights & Biases account (training curves log to W&B).

## One-time setup

```bash
uv tool install modal                 # the Modal CLI (not a project dependency)
modal setup                           # authenticate (opens a browser)
# Required: training curves log to W&B. Read the key from your password manager,
# shell env, or CI secret store; do not commit it.
export WANDB_API_KEY="..."
modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"
```

Hugging Face auth is **optional**: Qwen2.5-Math and GSM8K are public, so runs work
anonymously (just rate-limited). Modal has no "optional secret", so it's left out by default.
To authenticate (if you hit HF rate limits): create the secret and add
`modal.Secret.from_name("huggingface")` to the functions in `modal_app.py`.

```bash
export HF_TOKEN="..."
modal secret create huggingface HF_TOKEN="$HF_TOKEN"
```

The `assay-runs` Volume (checkpoints + provenance) is created automatically on first run.

## Day 1: smoke, then the `correct` arm

```bash
# 1. Cheap dry run (8 problems, 5 steps) to absorb the infra tax (image build, vLLM colocate).
modal run modal_app.py --arm configs/correct.yaml --smoke-problems 8 --max-steps 5
#    EOS-sync check: completions/clipped_ratio must NOT be pinned ~1.0 every step (persistent ~1.0
#    ⇒ trainer EOS and vLLM stop token disagree ⇒ gradient silently masked; stop and fix).
#    Occasional non-zero is just long completions hitting max_completion_length (masked safely).

# 2. The full correct arm. Long run -> --detach --spawn so it survives a local disconnect
#    (a plain `modal run` is killed with the client mid-train, losing the final checkpoint even
#    if wandb shows "finished"; monitor via `modal app logs` or `modal volume ls assay-runs`).
modal run --detach modal_app.py --arm configs/correct.yaml --spawn

# 3. Held-out accuracy curve over its checkpoints (runs on the GPU against the Volume).
modal run modal_app.py --arm configs/correct.yaml --command heldout
#    This is the GSM8K-vs-MATH decision: read HELD-OUT ACCURACY, not the reward curve.
#    If held-out accuracy is flat after run 1, fall back to MATH-level training.
```

## Day 2: the `random` (placebo) arm

```bash
modal run modal_app.py --arm configs/random.yaml
modal run modal_app.py --arm configs/random.yaml --command heldout
```

## Check These, Never the Reward Curve

- **held-out accuracy** on the validation split (`--command heldout` → `heldout.json`)
- **completion length + entropy** (W&B): entropy collapse is reward hacking, and KL reacts too slowly to catch it
- **`completions/clipped_ratio` not persistently ≈1.0**: the EOS-sync warning (occasional clipping from long completions is normal and masked)

## After training → analysis

Checkpoints, `provenance.json`, and `heldout.json` live on the `assay-runs` Volume.

```bash
modal volume ls  assay-runs
modal volume get assay-runs correct-seed0 ./runs/correct-seed0    # pull for local analysis
```

Then Phase 2: `grpo-decomp generate --backend vllm` on base + both checkpoints across the eval
sets, and `grpo-decomp report` for the decomposition table (see the README).

## Reproduce the decomposition (CPU only, no training)

Every committed number in `results/` is derived from `CompletionSet`s on the `assay-runs` Volume
by the `grpo-decomp report-*` aggregators — deterministic (fixed bootstrap seed), so a clean
checkout reproduces them exactly. Two levels, cheapest first:

**Verify figures + docs (no Volume, no Modal account).** `make results` regenerates the committed
PNGs from the JSON and runs the docs↔JSON consistency test, so a reviewer can confirm every
headline number traces to its artifact:

```bash
make install        # CPU env
make results         # figures from results/*.json + the docs<->JSON consistency check
```

**Re-derive the JSON from the completions (needs the Volume).** Pull the completion dirs, then
`make aggregate`:

```bash
uv tool install modal && modal setup
modal volume get assay-runs passk-multiseed           runs/passk-multiseed
modal volume get assay-runs passk-multiseed-countdown runs/passk-multiseed-countdown
modal volume get assay-runs battery                    runs/battery
for s in 1 2 3 4 5; do modal volume get assay-runs battery-seed$s runs/battery-seed$s; done
make aggregate       # report-passk-seeds / report-mechanism / report-control-seeds -> the multi-seed panels

# The confirmatory placebo comparison + the seed-0 full decomposition aren't in `make aggregate`;
# they read the same battery dirs. (GSM8K shown; for Countdown, swap in the countdown battery dirs
# and --task-set countdown-test.)
uv run grpo-decomp report-seeds --task-set gsm8k-test \
  --battery-dirs runs/battery runs/battery-seed{1,2,3,4,5} --out results/seed-placebo-comparison.json
uv run grpo-decomp report --completions-dir runs/battery --task-set gsm8k-test --out results/

make results         # then regenerate figures + verify
```

`passk-multiseed` carries the gsm8k-test, gsm-symbolic, and gsm8k-platinum pass@8 panels (the
`__<set>` suffix keeps them disjoint); `battery` holds the seed-independent base plus the seed-0
correct/random arms, and `battery-seed{1..5}` the per-seed correct/random arms the placebo
comparison and the §3 controls aggregate. A byte-identical re-gen is the healthy result; a diff
means a completion set or the code moved. The 24 KB `tests/fixtures/mini` exercises the same
aggregators end-to-end without any download.

## Tuning notes

- `configs/*.yaml`: `max_steps=500` and `save_steps=100` are placeholders; anchor on the day-1 run.
- `max_completion_length=1024` (raised from 512 after the dry run clipped a chunk of rollouts).
- `vllm` is pinned to `==0.17.1` (TRL 1.0.0's supported max); don't bump past 0.17.x without re-checking TRL compat.
- Checkpoints are ~3–9 GB each (weights + optimizer state); ~5 of them ≈ 15–45 GB on the Volume.
- `vllm_gpu_memory_utilization=0.3` (colocate). Raise if rollouts are slow; lower (and cut
  `max_completion_length` / `num_generations`) before touching base weights on OOM.
