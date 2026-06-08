# Decomposition — Qwen/Qwen2.5-1.5B on countdown-test [PRELIMINARY]

Artifact scope: Single-seed descriptive decomposition (1 seed). Treat the comparison below as a per-seed diagnostic; the headline claim must come from seed-level aggregation such as `seed-placebo-comparison.json`.

Confirmatory comparison: correct beats random by 34.9% (95% CI [26.6, 43.2]; McNemar p=8.14e-13; n=192)

| Control | Probes | Δ (pp) | 95% CI (pp) | McNemar p | n |
| --- | --- | --- | --- | --- | --- |
| raw gain | correct vs base on countdown-test | +35.4 | [+28.1, +42.7] | 6.84e-14 | 192 |
| format sensitivity | lenient vs strict (same completions) | +0.0 | [+0.0, +0.0] | 1 | 192 |

Elicitation (separate panel): pass@1: base=0.14, correct=0.49; high-n pass@k coverage deferred to a Phase-2 sampling run

Caveats:
- Rows are independent re-measurements of the raw gain under each control; they overlap and MUST NOT be summed into an additive partition.
- The placebo (correct - random) delta is a within-Qwen lower bound on non-correctness-driven gain, not a cross-family artifact verdict (needs the v2 Llama arm).
- Only the placebo comparison is the pre-registered confirmatory test; every table row is descriptive/exploratory and its 95% CI is marginal (per-row), NOT family-wise corrected — do not read a single row's p<0.05 as confirmed.
- PRELIMINARY: aggregated over 1 seed(s) (< 3); CIs reflect eval-sampling noise only, not run-to-run / seed variance — not a headline claim.
