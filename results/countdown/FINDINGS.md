# grpo-gain-decomposition Countdown findings (general Qwen2.5-1.5B, GRPO on Countdown)

The **positive control**. Placebo comparison: **3 seeds per arm**; held-out `countdown-test`
(n=192, disjoint from train, procedurally generated so uncontaminated); pass@1 greedy. The
pass@k panel is seed 0 (n=8, temp 0.7). Commit-pinned artifacts:
`seed-placebo-comparison.json`,
`elicitation.json`, `summary.json`, `decomposition.md`.

## Headline (controlled)

A **large (+46.5pp), correctness-driven, statistically significant** gain that is **genuine
capability expansion, not elicitation**: the trained model solves at pass@8 problems the
base cannot solve in 8 tries. This is the *contrast* to GSM8K, and it validates the
decomposition: the same instrument reads **elicitation** on GSM8K (a saturated benchmark)
and **expansion** here (a search task the base genuinely lacks).

## 1. Placebo comparison across seeds (the pre-registered confirmatory test)

correct − random, 3 seeds → mean **+46.5 pp**, 95% CI **[21.4, 71.6]** (seed-level t, df=2).
**Significant**: the interval excludes zero by a wide margin, and all 3 seeds are strongly
positive.

| seed | random | correct | Δ (pp) |
| --- | --- | --- | --- |
| 0 | 14.1% | 49.0% | +34.9 |
| 1 | 4.2% | 55.7% | +51.6 |
| 2 | 0.0% | 53.1% | +53.1 |

- **The correct arm learned the task.** From a base of ~9% it reaches ~49–56% held-out
  pass@1, a skill the base genuinely lacked, not reliability on a skill it had.
- **The placebo genuinely doesn't help.** Random reward sits at 0–14% (≈ base or below; a
  correctness-blind reward can't teach search). Seed 2's random arm collapsed to 0%.
- Three seeds, not six: the effect is so large that the seed-level interval clears zero
  decisively (the GSM8K comparison needed six only because +3.9pp is small).

## 2. The pass@k curve: expansion, not elicitation (seed 0, n=8, temp 0.7)

| arm | pass@1 | pass@8 |
| --- | --- | --- |
| base | 9.0% | **50.5%** |
| correct | 59.6% | **95.3%** |

- **base pass@8 (50.5%) ≪ correct pass@8 (95.3%)**: pass@8 coverage **moved +44.8 pp**. The
  trained model solves problems at pass@8 that the base fails even with 8 attempts: *new
  capability*, by definition outside the base's pass@k envelope.
- Contrast GSM8K, where base pass@8 (93.6%) ≈ correct pass@8 (95.3%): pass@8 coverage barely
  moved (+1.7pp), so the gain was elicitation. **Same decomposition, opposite verdicts.**

## 3. Two-sided validation (the point of the study)

| | GSM8K (Qwen2.5-Math-1.5B) | Countdown (general Qwen2.5-1.5B) |
| --- | --- | --- |
| placebo comparison | +3.9 pp [2.3, 5.6] | **+46.5 pp [21.4, 71.6]** |
| pass@8 coverage (base → correct) | 93.6 → 95.3 (**+1.7**) | 50.5 → 95.3 (**+44.8**) |
| verdict | **elicitation** (saturated base) | **expansion** (base lacked the skill) |

The decomposition isn't biased toward "it's all fake": on a task where RL genuinely teaches
new ability, it reports expansion with higher pass@8 coverage; on a saturated benchmark it reports
elicitation. That two-sidedness is what makes the GSM8K "mostly elicitation" finding
trustworthy rather than a null-by-construction artifact.

## Bottom line

On Countdown — a verifiable, uncontaminated search task the base cannot do — GRPO with a
correctness reward delivers a **large, significant, genuinely-new-capability** gain (+46.5pp
placebo comparison; pass@8 coverage 50.5 → 95.3). It is the positive control that proves the
decomposition can detect expansion when it exists. Paired with GSM8K's controlled
elicitation result, the instrument is validated from both sides.

## Caveats

- **3 seeds**: the placebo comparison clears zero decisively, but the CI is wide (df=2);
  the point estimate is less precise than GSM8K's 6-seed comparison.
- **Within-Qwen, single base**: the placebo is a within-model lower bound on
  non-correctness-driven gain, not a cross-family verdict (a Llama arm is the follow-up).
- **Narrow task**: Countdown certifies that the *instrument* detects expansion; it does not
  claim broad reasoning transfer. It is the control, not a marquee capability result.
