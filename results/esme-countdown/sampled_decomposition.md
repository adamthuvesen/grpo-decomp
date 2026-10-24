# Sampled decomposition: Esme-214M-RL on esme-countdown

The retired greedy `decomposition.md` artifact scored one deterministic sample per problem
on exact-solve only. Held-out Countdown-Lite is scored here with sampling
(**n=16, temperature 1.0**) across two axes:

- **valid-expression rate**: a legal Countdown-Lite expression using each supplied number
  once with `+ - *`, regardless of whether it reaches the target.
- **exact-solve pass@k**: the accepted solve metric, restored to a sampled estimate.

The 2026-07-05 result aggregates six real-reward GRPO seeds against six same-budget
random-reward placebo seeds. The report is produced by `scripts/esme_sampled_decomp.py`,
with raw numbers stored in `sampled_multiseed_summary.json`. The raw completions are in
`completions/`, and `make reproduce-esme` recomputes the summary from them.

![Esme-214M-RL form vs exact solving: real reward separates from the random-reward placebo on valid-expression rate first, exact solving second](fig-sampled-form-vs-exact.svg)

## Why this exists

For a 214M model on Countdown, greedy exact pass@1 is the lowest-power slice available:
the whole dynamic range was one or two solved problems. That made the original strict
report read as non-separable even though the accepted run's own sampled eval showed a
large validity/form shift.

Sampling and scoring validity match the reward ladder better. GRPO pays
`invalid 0.0 < valid-expression 0.3 < exact 1.0`, so the first place a real verifier
reward should separate from a random reward is well-formedness. The random-reward arm has
the same recipe and budget, but its reward is independent of task correctness.

## Result

Seed-level result, each seed on the same 30 held-out problems with 16 samples/problem:

| Seed | correct valid | random valid | Δ valid | correct any-exact | random any-exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 214 | 27.1% | 0.8% | +26.2pp | 4/30 | 0/30 |
| 215 | 97.7% | 0.4% | +97.3pp | 3/30 | 0/30 |
| 216 | 97.1% | 0.8% | +96.2pp | 3/30 | 1/30 |
| 217 | 97.1% | 0.6% | +96.5pp | 3/30 | 0/30 |
| 218 | 96.7% | 0.6% | +96.0pp | 3/30 | 1/30 |
| 219 | 96.9% | 1.2% | +95.6pp | 3/30 | 1/30 |

Aggregate arm means:

| Arm | valid-expr rate | pass@1 | pass@8 | pass@16 | any-exact solved |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (Esme-214M-Chat) | 0.8% | 0.2% | 1.7% | 3.3% | 1/30 |
| correct (Esme-214M-RL, real reward) | 85.4% | 9.0% | 10.3% | 10.6% | 3.17/30 |
| random (placebo, random reward) | 0.8% | 0.1% | 0.8% | 1.7% | 0.50/30 |

Tests (paired per problem against each seed's own placebo; seeds and problems resampled
together, 6 seeds × 30 problems):

| Axis | Comparison | Δ | 95% CI | test |
| --- | --- | ---: | --- | --- |
| valid-expr rate | correct vs random | **+84.7 pp** | **[+60.1, +97.5]** pp | seed × problem bootstrap |
| any-exact solve | correct vs random | **+8.9 pp** | **[+0.6, +19.4]** pp | seed × problem bootstrap |

Traceable headline strings: valid-expression rates are **85.4% vs 0.8%** for real reward
vs placebo; valid-expression separation is **+84.7 pp, 95% CI [+60.1, +97.5]**; any-exact
separation is **+8.9 pp, 95% CI [+0.6, +19.4]**.

Conclusion: **supported** for validity. Real verifier reward separates from the
random-reward placebo on sampled held-out Countdown validity. Exact-any moves the same way
but only just clears zero, so validity is the headline: it is the reward rung with the
strongest signal and the most direct mechanism.

## Reading

The six-seed result is consistent: real reward lands near 97% valid-expression rate in five
of six seeds, while placebo stays near base in all six. The validity interval clears zero
with room to spare despite the low-validity seed 214.

The exact-solve axis is small and fragile. Real reward averages 3.17 held-out problems with
at least one exact sample, while placebo averages 0.50. Every real-reward seed solves the
same three problems (`easy_0000`, `easy_0002`, `medium_0004`; seed 214 adds `medium_0003`).
The effect is consistent across seeds but rests on three problems, which is why its interval
is wide once problems are resampled. This supports the "RL sharpened form first, exact
solving second" reading rather than a broad new reasoning claim for a 214M model.

## Caveats

- **Six seeds, 30 problems.** The interval resamples both, so it covers run-to-run and
  problem-sampling noise and stays inside the possible range. An earlier seed-level t
  interval ignored problem noise and reached +114.7 pp, past the 100 pp ceiling. With six
  seeds the seed resample is coarse, so read the bounds as approximate.
- **Validity is form, not target solving.** A valid expression may still hit the wrong value.
  That is not a bug: validity is the rung the reward shaped most strongly, and exact-solving is
  reported separately.
- **Held-out fresh, short decode.** These are held-out problems at temperature 1.0 with a
  12-token budget. The in-distribution acceptance eval is a different slice; the cross-arm
  comparison here is apples-to-apples.
