# Countdown findings: the positive control

_general Qwen2.5-1.5B · GRPO with a correctness reward · 3 training seeds per arm ·
generated Countdown test, n=192 (disjoint from training, so uncontaminated)_

**Result.** GRPO beats a random-reward placebo by **+46.5 pp, 95% CI [21.4, 71.6]**, and
this time pass@8 coverage rises with it. The gain is expansion: the trained model solves
problems the base model cannot solve in 8 tries. The same protocol that reports elicitation
on GSM8K reports expansion here, so its GSM8K verdict is not built in.

Settings match the GSM8K study: greedy pass@1 for the placebo comparison; pass@8 at
temperature 0.7 with base n=16 and correct n=8 for coverage. Artifacts:
`seed-placebo-comparison.json`, `pass8-multiseed.json`, `mechanism.json`.

## Placebo comparison

correct − random over 3 seeds: **+46.5 pp, 95% CI [21.4, 71.6]** (seed-level t, df=2).

| Seed | random | correct | Δ (pp) |
| --- | --- | --- | --- |
| 0 | 14.1% | 49.0% | +34.9 |
| 1 | 4.2% | 55.7% | +51.6 |
| 2 | 0.0% | 53.1% | +53.1 |

Random reward stays at or below the base model (11.0% pass@1). A reward that ignores
correctness cannot teach search.

## Coverage

| Arm | pass@1 (sampled) | pass@8 |
| --- | --- | --- |
| base | 11.0% | **53.6% [48.4, 58.9]** |
| correct, 3-seed mean | 58.0% | **94.6% [91.9, 97.3]** |

Coverage moves: Δ +41.0 pp, 95% CI [+35.1, +46.9] with the base model's sampling noise
added (seed-level [38.3, 43.7]).

![pass@8 coverage: GSM8K barely moves (+0.7 pp, elicitation) while Countdown rises from 53.6 to 94.6 (+41.0 pp, expansion)](../fig-passk-contrast.png)

## Per-problem mechanism

The base model solves almost nothing reliably on the first try (0.5%). The trained model
makes 47.9% of problems reliable within the base model's pass@8 reach, and 10.9% genuinely
new: reliable after RL, never solved by the base model in 8 tries. Completions get shorter
(366 → 315 words).

## Against GSM8K

| | GSM8K (Qwen2.5-Math-1.5B) | Countdown (Qwen2.5-1.5B) |
| --- | --- | --- |
| Placebo comparison (pass@1) | +3.9 pp [2.3, 5.6] | +46.5 pp [21.4, 71.6] |
| Δ pass@8 | +0.7 pp [-0.4, +1.9] | +41.0 pp [+35.1, +46.9] |
| base → correct pass@8 | 94.0 → 94.7 | 53.6 → 94.6 |
| New-capability share | 0.0% | 10.9% |
| Verdict | elicitation | expansion |

## Same base model

The table above changes the base model as well as the task. Trained from the GSM8K base
(Qwen2.5-Math-1.5B), Countdown still expands, by +11.8 pp (paired CI [8.3, 15.5]), but with
0.5% new capability. The Math model already covers 84.8% of Countdown in 8 tries. See the
[pre-registered results](../preregistered-results-2026-10.md).

## Limitations

- **3 seeds.** Both intervals clear zero, but with df=2 they are wide.
- **One model family.** The placebo is a within-Qwen lower bound.
- **Narrow task.** Countdown shows that the protocol detects expansion. It is not evidence
  of broad reasoning transfer.
- **No verifiable-chain check.** Countdown completions contain no `<<a op b = c>>` steps:
  chain coverage is 0.0%, so CoT-gated pass@8 is 0.0% for both arms.
