# Decomposition: Qwen/Qwen2.5-Math-1.5B on gsm8k-test [PRELIMINARY]

Artifact scope: Single-seed descriptive decomposition (1 seed). Treat the comparison below as a per-seed diagnostic; the headline claim must come from seed-level aggregation such as `seed-placebo-comparison.json`.

Confirmatory comparison: correct beats random by 6.1% (95% CI [4.2, 8.2]; McNemar p=3.34e-09; n=1319)

| Control                  | Check                                 | Δ (pp) | 95% CI (pp)   | McNemar p | n    |
| ------------------------ | ------------------------------------- | ------ | ------------- | --------- | ---- |
| raw gain                 | correct vs base on gsm8k-test         | +5.0   | [+3.2, +7.0]  | 6.19e-07  | 1319 |
| control (gsm-plus)       | robustness (adversarial perturbation) | +4.9   | [+4.2, +5.7]  | 9.63e-37  | 9233 |
| control (gsm-symbolic)   | memorization (templated renumbering)  | +10.5  | [+9.3, +11.7] | 1.93e-61  | 5000 |
| control (gsm8k-platinum) | label noise (cleaned labels)          | +4.8   | [+2.8, +6.8]  | 3.26e-06  | 1209 |
| format sensitivity       | lenient vs strict (same completions)  | +0.8   | [+0.4, +1.4]  | 0.000977  | 1319 |

Elicitation (separate panel): pass@1: base=0.76, correct=0.81; pass@k coverage is reported in the separate multi-seed panel (pass8-multiseed.json via grpo-decomp report-passk-seeds)

Caveats:

- Rows re-measure the raw gain under each control. They overlap, so do not sum them into an additive partition.
- The placebo (correct - random) delta is a within-Qwen lower bound on non-correctness-driven gain, not a cross-family verdict.
- Only the placebo comparison is the pre-registered confirmatory test. Every other row is descriptive, and its 95% CI is marginal (per-row), not family-wise corrected. Do not read a single row's p<0.05 as confirmed.
- PRELIMINARY: aggregated over 1 seed(s) (< 3). CIs reflect eval-sampling noise only; they do not include run-to-run seed variance, so this is not a headline claim.
