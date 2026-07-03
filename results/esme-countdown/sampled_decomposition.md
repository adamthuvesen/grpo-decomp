# Sampled decomposition — Esme-214M-RL on esme-countdown

Companion to the greedy `decomposition.md`. Same three arms (base / correct / random) on the
same 30 held-out Countdown problems, but sampled (**n=16, temperature 1.0**) instead of greedy
pass@1, and scored on two axes instead of one. Produced by `scripts/esme_sampled_decomp.py`;
raw numbers in `sampled_summary.json`.

![Esme-214M-RL form vs exact solving: real reward separates from the random-reward placebo on valid-expression rate first, exact solving second](fig-sampled-form-vs-exact.svg)

## Why this exists

The greedy report scored one deterministic sample per problem on **exact-solve only**. For a
214M model on Countdown that is the sparsest, lowest-power slice available — the whole dynamic
range is 1-2 solved problems — so real reward and the random-reward placebo came out identical
(+3.3pp, McNemar p=1.0). That was a measurement artifact, not a null result. Two fixes:

- **Sample (n=16).** Restores the pass@k range a single greedy decode collapses — this is how
  the accepted acceptance eval (32 samples) saw the gain at all.
- **Score validity, not just exact-solve.** GRPO's reward pays a graded ladder
  (invalid 0.0 < valid-expression 0.3 < exact 1.0). Its clearest effect is on the
  **valid-expression** rung (the accepted run took validity 5.83% → 99.38%). A random reward
  has no gradient toward well-formedness, so validity is exactly where a real reward and a
  placebo should diverge — and the axis greedy-exact throws away.

## Result

| Arm | valid-expr rate | pass@1 | pass@8 | pass@16 | any-exact solved |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (Esme-214M-Chat) | 0.8% | 0.2% | 1.7% | 3.3% | 1/30 |
| correct (Esme-214M-RL, real reward) | 27.1% | 5.6% | 11.5% | 13.3% | 4/30 |
| random (placebo, random reward) | 0.8% | 0.0% | 0.0% | 0.0% | 0/30 |

Paired per-problem tests (unit = problem, n=30):

| Axis | Comparison | Δ | 95% CI | test |
| --- | --- | ---: | --- | --- |
| valid-expr rate | correct vs random | **+26.2pp** | [+17.3, +36.0]pp | paired bootstrap |
| valid-expr rate | correct vs base | **+26.2pp** | [+17.5, +35.8]pp | paired bootstrap |
| any-exact solve | correct vs random | +13.3pp | [+3.3, +26.7]pp | exact-binomial p=0.125, n_discordant=4 |

## Reading

**On the axis the reward actually shapes, real reward is cleanly and significantly separable
from a same-budget placebo.** Real verifier reward lifts the valid-expression rate 0.8% → 27.1%
(+26.2pp, 95% CI [+17.3, +36.0]pp — nowhere near zero). The random-reward placebo sits at
**0.8%, identical to base**: it reproduces none of the reward's contribution to well-formedness.
This is the opposite of the greedy table's "not separable" and it is the honest headline — the
earlier result was an artifact of measuring only the rarest rung with one deterministic decode.

**Exact-solve moves the same way but is underpowered here.** With sampling, correct solves
4/30 (pass@16 13.3%) vs the placebo's 0/30 and base's 1/30 — all four discordant problems favor
correct, but at n=30 with a rare event the exact-binomial gives p=0.125. The placebo landing
*below* base on exact-solve (0 vs 1) is consistent with a random reward eroding the little
exact ability base had while a real reward builds it. Significance rests on the validity axis;
exact-solve is directionally unanimous but needs more problems (or seeds) to confirm on its own.

**Consistency with the accepted run.** Same direction as the acceptance eval (in-distribution
sampled pass@1 3.33% → 16.67%) and the "RL sharpened form, did not create new reasoning
capability" finding: the largest, most significant effect is on validity/form; exact-solve
gains are real but small and easy-band-only. Held-out greedy just could not see either.

## Caveats

- **Single seed.** One placebo training run; CIs are eval-sampling (problem + draw) noise, not
  run-to-run seed variance. The validity separation is large enough (+26pp) that seed noise is
  very unlikely to erase it, but a multi-seed placebo (`report-control-seeds`, ≥3 seeds) is
  still the bar for a fully headline claim, and now the cheap axis to run it on is validity.
- **Valid-expr rate 27.1% here vs 99.38% in the accepted run** is expected: this is held-out
  fresh problems at temperature 1.0 with a 12-token budget, not the in-distribution eval split.
  The cross-arm comparison is what matters and it is apples-to-apples.
- `valid-expr rate` is target-independent well-formedness (each supplied number used once,
  `+ - *` only, parses); `any-exact solved` and pass@k use the exact-solve verifier. Both use
  the harness's lenient extraction on the emitted `\boxed{...}`.
