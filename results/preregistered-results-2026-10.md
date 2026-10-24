# Results: off-policy placebo and same-base control

These two experiments were pre-registered in
[`preregistration-2026-10.md`](preregistration-2026-10.md). That file is unchanged. All
runs and completions come from its commit (`56848d4`, `dirty=false`). Spend: $11.09 of
Modal GPU time, against a $20 budget.

| Experiment | Pre-registered verdict | Prediction | Held? |
| --- | --- | --- | --- |
| Off-policy placebo | No detectable effect | Small, CI likely contains 0 | Yes |
| Same-base control | Expansion (narrowly) | Expansion | Yes, at a much smaller size |

## 1. Off-policy placebo

_Qwen2.5-Math-1.5B · random reward · `steps_per_generation: 4` · 3 seeds · GSM8K test, n=1319_

Random − base = **+0.4 pp, 95% CI [-0.8, +1.6]** (seed × problem bootstrap, paired per
problem against a freshly generated base model at 76.1%). Per seed: +0.2, +0.8, +0.1 pp.
The CI contains 0, so the verdict is **no detectable effect**.

**How strong is this test?** It is weak. Off-policy training lets the PPO clip bind, but in
the logged training intervals it bound on only 0.002–0.015% of tokens. At learning rate
1e-6, the policy barely moves within the 4 optimizer steps that share one rollout batch. So
the clipping mechanism was hardly exercised. The result shows that the flat placebo
survives the switch to off-policy training at this budget. It does not confirm or refute
the Spurious Rewards mechanism at 1.5B. A real test needs the clip to bind often, through a
higher learning rate or more optimizer steps per rollout batch.

Artifact: [`offpolicy-placebo.json`](offpolicy-placebo.json), with the per-problem grades
for the base and every seed.

## 2. Same-base positive control

_Qwen2.5-Math-1.5B · correct reward · 3 seeds · Countdown test, n=192 · base n=16, correct n=8_

| | General Qwen2.5-1.5B (published) | Qwen2.5-Math-1.5B (this run) |
| --- | --- | --- |
| base pass@1 | 11.0% | 32.2% |
| base pass@8 | 53.6% | 84.8% |
| correct pass@8 | 94.6% | 96.5% |
| Δ pass@8 | +41.0 pp | **+11.8 pp** (paired CI [8.3, 15.5]) |
| new-capability share | 10.9% | **0.5%** |

Δ pass@8 = **+11.8 pp, paired 95% CI [8.3, 15.5]**. It clears the pre-registered rule
(Δ ≥ 10 pp and CI lower bound > 0), so the verdict is **expansion**. Base pass@8 is below
the 90% "uninformative" line.

**Reading.** The task effect survives on the same base model. On Qwen2.5-Math-1.5B,
Countdown coverage rises by +11.8 pp, against +0.7 pp on GSM8K. But the effect is about
3.5× smaller than the published contrast. The Math model already covers 84.8% of
Countdown in 8 tries, so there is less room to expand. Per problem, 98% of the added
reliability is migration within the base model's reach, and only 0.5% of problems are
genuinely new. The published +41.0 pp measured the task and the base model together.

**Revised thesis.** RL expands coverage in proportion to the headroom the base model leaves
on the task. GSM8K on the Math model leaves almost none (base pass@8 94.0%). Countdown on
the Math model leaves some (84.8%). Countdown on the general model leaves a lot (53.6%).

**Not run.** There is no random-reward placebo for this base × task pair. The verdict
compares against the base model, not a placebo.

Artifacts: [`countdown-math/pass8-multiseed.json`](countdown-math/pass8-multiseed.json)
(with per-problem counts) and [`countdown-math/mechanism.json`](countdown-math/mechanism.json).
