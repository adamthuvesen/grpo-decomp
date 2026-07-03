# Decomposition — Esme-214M-Chat on esme-countdown [PRELIMINARY]

Artifact scope: Single-seed descriptive decomposition (1 seed). Treat the comparison below as a per-seed diagnostic; the headline claim must come from seed-level aggregation such as `seed-placebo-comparison.json`.

Confirmatory comparison: correct beats random by 3.3% (95% CI [0.0, 10.0]; McNemar p=1; n=30)

| Control | Probes | Δ (pp) | 95% CI (pp) | McNemar p | n |
| --- | --- | --- | --- | --- | --- |
| raw gain | correct vs base on esme-countdown | +3.3 | [+0.0, +10.0] | 1 | 30 |
| format sensitivity | lenient vs strict (same completions) | +0.0 | [+0.0, +0.0] | 1 | 30 |

Elicitation (separate panel): pass@1: base=0.03, correct=0.07; pass@k coverage is reported in the separate multi-seed panel (pass8-multiseed.json via grpo-decomp report-passk-seeds)

Caveats:
- Rows are independent re-measurements of the raw gain under each control; they overlap and MUST NOT be summed into an additive partition.
- The placebo (correct - random) delta is a within-Qwen lower bound on non-correctness-driven gain, not a cross-family artifact verdict (needs the v2 Llama arm).
- Only the placebo comparison is the pre-registered confirmatory test; every table row is descriptive/exploratory and its 95% CI is marginal (per-row), NOT family-wise corrected — do not read a single row's p<0.05 as confirmed.
- PRELIMINARY: aggregated over 1 seed(s) (< 3); CIs reflect eval-sampling noise only, not run-to-run / seed variance — not a headline claim.
