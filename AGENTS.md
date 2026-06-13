# AGENTS.md

Repo-local instructions for AI coding agents working in `grpo-gain-decomposition`.

`grpo-gain-decomposition` is a controlled study: train a small model with GRPO on math, then run an
adversarial eval battery that decomposes the benchmark gain into real reasoning vs.
contamination, formatting, and elicitation. Its taste is the same as a good
experiment — the result is only as trustworthy as the controls. A finding that
says "most of this gain was elicitation" is more valuable than a confident,
uncontrolled "RL works."

## Working Contract

- Start from repo truth: read `README.md`, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  `pyproject.toml`, and nearby tests before inventing patterns.
- Prefer the smallest coherent change. This repo is a focused, narrow study, not an
  RL framework. Do not turn it into a platform.
- **Preserve controlled evaluation.** Never report a gain without its controls. Every
  headline number carries a confidence interval and a paired significance test.
  Never claim from a single training run — aggregate over seeds.
- The training reward curve is not evidence. Only held-out eval, completion-length,
  and entropy are. Treat a rising reward with suspicion.
- Determinism where it matters: fixed seeds, pinned dataset revisions, recorded
  model/commit/config hashes on every result artifact.
- Do not add runtime dependencies without asking. The intended stack is TRL +
  transformers + vLLM + datasets for training, and `eval-audit` for statistics.
  No broad RL/agent frameworks beyond TRL.

## Repo Map

Full architecture + data flow + diagrams: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

- `src/grpo_gain_decomp/data/` — GSM8K + perturbation-set loaders (GSM-Symbolic, GSM-Plus, GSM8K-Platinum), pinned revisions.
- `src/grpo_gain_decomp/rewards/` — verifiable reward functions sharing one signature: `correct`, `random` (placebo), `format`.
- `src/grpo_gain_decomp/train/` — TRL `GRPOConfig` + run launcher (one arm per config).
- `src/grpo_gain_decomp/eval/` — answer extraction (strict/lenient), pass@k + CoT-pass@k estimators, the "code-reasoning" detector.
- `src/grpo_gain_decomp/stats/` — McNemar + paired bootstrap CIs, or a thin adapter over `eval-audit`.
- `src/grpo_gain_decomp/report/` — deterministic decomposition-table generator.
- `configs/` — one YAML per arm + seed.
- `results/` — committed: the headline decomposition table, plots, `summary.json`. (`runs/` checkpoints are gitignored.)

## Conventions

- Python 3.11+, `uv`, ruff (line length 100), strict Pydantic for result schemas.
- `CLAUDE.md` is a symlink to this file.
- Pin TRL — its GRPO defaults move between versions; re-verify config at build time.
