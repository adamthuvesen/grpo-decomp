# Pre-registration: off-policy placebo and same-base control

Written 2026-10-24, before any training run for these experiments. The git commit that adds
this file is the time stamp: it precedes every run, completion, and result it describes.
Budget: about $20 of Modal GPU time. If the timing smoke run puts one training run above $4,
the experiments stop and are re-quoted.

Both experiments change one variable from an existing arm.
`tests/test_train_config.py` pins that contract.

## Experiment 1: off-policy placebo

**Question.** Does the random-reward placebo gain once GRPO's PPO clip can bind? Shao et al.
2025 (_Spurious Rewards_) trace random-reward gains on Qwen2.5-Math to the clipping term. The
published placebo here is on-policy (one optimizer step per rollout batch), so its policy
ratio stays at 1 and the clip never binds.

**Arms.** `configs/offpolicy-random{,-seed1,-seed2}.yaml`: seeds 0, 1, 2. Identical to
`configs/random.yaml` except `steps_per_generation: 4`. Each rollout batch (4 prompts × 8
completions) feeds 4 optimizer steps. The run keeps the same 500 optimizer steps and the same
500 training prompts.

**Metric.** GSM8K test (n=1319), greedy pass@1, lenient grading, 1024-token budget: the
same metric as the published placebo comparison. Each seed is paired per problem with a
freshly generated base-model run. The 95% CI is a seed × problem bootstrap
(`grpo-decomp report-arm-vs-base`).

**Reference.** The on-policy placebo sits at 75.8% vs base 76.4% (6 seeds, seed-level).

**Decision rule.**

- **Supported at 1.5B:** CI lower bound > 0.
- **No detectable effect:** CI contains 0. This is not a refutation. The training budget
  here (500 prompts × 8 completions) is far below Spurious Rewards' (64 prompts × 16
  rollouts × 300 steps), and they report that the 1.5B model gains slowly.
- **Harmful:** CI upper bound < 0.

**Prediction.** A small effect, at most +3 pp. The most likely outcome is a CI that
contains 0.

## Experiment 2: same-base positive control

**Question.** Does the GSM8K/Countdown contrast come from the task, or from the base model?
The published Countdown arm uses general Qwen2.5-1.5B, and GSM8K uses Qwen2.5-Math-1.5B.

**Arms.** `configs/countdown-math-correct{,-seed1,-seed2}.yaml`: seeds 0, 1, 2. Identical
to `configs/countdown-correct.yaml` except the base model (Qwen2.5-Math-1.5B, the GSM8K
base, at the same pinned revision).

**Metric.** Countdown test (n=192), pass@8 with base n=16 and correct n=8, temperature
0.7, 1024-token budget: the same panel as the published Countdown result. Δ pass@8 =
mean correct − base, with the paired seed × problem CI (`delta_paired_ci_*`). The
new-capability share comes from `report-mechanism` (τ = 0.5).

**Decision rule.** The harness's existing verdict, applied unchanged:

- **Expansion:** Δ ≥ 10 pp and paired CI lower bound > 0. The task, not the base model,
  drives the contrast.
- **Bounded-small (elicitation):** otherwise. If base pass@8 is low, the contrast depends
  on the model, and the README's claim must be weakened.
- **Uninformative:** base pass@8 ≥ 90%. There is no headroom for expansion.

**Prediction.** Expansion.

**Not run.** A random-reward placebo for this base × task pair is out of budget. The
expansion verdict compares against the base model, not a placebo. The writeup will say so.
