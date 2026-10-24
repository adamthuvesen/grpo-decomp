# Pre-registered results: off-policy placebo and same-base control

Both experiments follow the plan in [`preregistration-2026-10.md`](preregistration-2026-10.md),
committed as `56848d4` before any run. Every run and completion records that commit.

| Experiment | Verdict (pre-registered rule) | Prediction held? |
| --- | --- | --- |
| Off-policy placebo | no detectable effect | yes |
| Same-base control | expansion, narrowly | yes, at a much smaller size |

## Off-policy placebo

_Qwen2.5-Math-1.5B · random reward · `steps_per_generation: 4` · 3 seeds · GSM8K test, n=1319_

Random − base = **+0.4 pp, 95% CI [-0.8, +1.6]** (seed × problem bootstrap, paired against
the base model at 76.1%). Per seed: +0.2, +0.8, +0.1 pp. The CI contains 0: no detectable
effect.

**This is a weak test of the clipping mechanism.**

- **The clip rarely bound.** In the retained training logs (steps 450–500 of 500) the PPO
  clip bound on at most 0.015% of tokens
  ([`offpolicy-clip-ratios.json`](offpolicy-clip-ratios.json)). The clip fraction over
  the first 450 steps is not known. At learning rate 1e-6 the policy moves little within the
  4 steps that share a rollout batch.
- **The arm differs from the on-policy placebo in two ways, not one.** In TRL 1.0.0,
  `steps_per_generation: 4` generates 4 prompts × 8 completions, shuffles them, and splits
  them across 4 optimizer steps (`GRPOTrainer._prepare_inputs`). Each step is off-policy and
  also mixes 4 prompts. Both arms train on the same 500 prompts.

So the flat placebo survives off-policy updates at this budget. The result does not settle
the Spurious Rewards mechanism at 1.5B. That needs a setting where the clip binds often.

Artifact: [`offpolicy-placebo.json`](offpolicy-placebo.json), with per-problem grades.

## Same-base control

_Qwen2.5-Math-1.5B · correct reward · 3 seeds · Countdown test, n=192 · base n=16, correct n=8_

| | General Qwen2.5-1.5B | Qwen2.5-Math-1.5B |
| --- | --- | --- |
| base pass@1 | 11.0% | 32.2% |
| base pass@8 | 53.6% | 84.8% |
| correct pass@8 | 94.6% | 96.5% |
| Δ pass@8 | +41.0 pp | **+11.8 pp** (paired CI [8.3, 15.5]) |
| new-capability share | 10.9% | **0.5%** |

Δ pass@8 = **+11.8 pp, paired 95% CI [8.3, 15.5]**: Δ ≥ 10 pp with a lower bound above 0,
so the verdict is expansion. Base pass@8 is below the 90% line where the test would be
uninformative.

**Reading.** On one base model, the task drives the contrast: Countdown coverage rises by
+11.8 pp, GSM8K by +0.7 pp. But the gain is about 3.5× smaller than on the general base
model, because the Math model already covers 84.8% of Countdown in 8 tries. 98% of the added
reliability is migration within the base model's reach, and only 0.5% of problems are
genuinely new.

**Post-hoc, not pre-registered:** the share of remaining headroom (100% − base pass@8)
that RL closes.

| Panel | base pass@8 | Δ pass@8 | Headroom closed |
| --- | --- | --- | --- |
| GSM8K, Math base | 94.0% | +0.7 pp | 12% |
| Countdown, Math base | 84.8% | +11.8 pp | 77% |
| Countdown, general base | 53.6% | +41.0 pp | 88% |

Countdown RL closes most of the remaining headroom on both bases; GSM8K RL closes little.
Most of the 3.5× gap in Δ comes from the smaller headroom (46.4 vs 15.2 pp). One base-model
swap cannot separate headroom from model identity, so this is a hypothesis.

**Not run:** a random-reward placebo for this pair. The verdict compares against the base
model.

Artifacts: [`countdown-math/pass8-multiseed.json`](countdown-math/pass8-multiseed.json)
(with per-problem counts) and [`countdown-math/mechanism.json`](countdown-math/mechanism.json).
