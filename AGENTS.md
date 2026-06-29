# AGENTS.md — llm-grpo-gains

`llm-grpo-gains` is a controlled study: train a small model with GRPO on math,
then run an adversarial eval battery that decomposes the benchmark gain into real reasoning
vs. contamination, formatting, and elicitation. The result is only as trustworthy as the
controls — "most of this gain was elicitation" beats a confident, uncontrolled "RL works."

This is a focused, narrow study, not an RL framework. Keep it that way; do not turn it into
a platform. User-level guidance (tone, principles, git etiquette) lives in the user's agent
defaults and is _not_ duplicated here.

## Quickstart

```bash
make install        # CPU env: data, rewards, eval, stats, report
make check          # ruff + unit tests + docs consistency
make demo           # score committed mini CompletionSets; no model load
make results        # rebuild figures from results/*.json + docs<->JSON consistency check

modal run --detach modal_app.py --arm configs/correct.yaml  # train one arm on an A100 (see docs/RUNBOOK.md)
grpo-decomp battery --completions runs/base__dev --k 1      # grade a CompletionSet (CPU)
grpo-decomp report  --completions-dir runs/ --out results/  # <arm>__<set> dirs -> table + summary.json
```

Only `generate` (and training) needs a model/GPU; `battery`, `report`, and `make results` run on CPU.

## Conventions

- Start from repo truth: read `README.md`, [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), `pyproject.toml`, and nearby tests before inventing patterns.
- Prefer the smallest coherent change; if a doc disagrees with code, fix the doc in the same change.
- **Controls before gains.** Never report a gain without its controls, a confidence interval, and a paired significance test.
- **The reward curve is not evidence.** Only held-out eval, completion length, and entropy are; treat a rising reward with suspicion.
- **Never claim from one run** — aggregate over seeds (below three seeds, label preliminary).
- Determinism: fixed seeds, pinned dataset revisions, recorded model/commit/config/dependency hashes on every artifact.
- **Pin TRL** — its GRPO defaults move between versions; re-verify config at build time. Same for vLLM (`==0.17.1`, TRL 1.0.0's supported max).
- No new runtime deps without asking. Stack is TRL + transformers + vLLM + datasets (train) and `eval-audit` (stats); no broad RL/agent frameworks.
- Python 3.11+, `uv`, ruff (line length 100), strict Pydantic for result schemas. `CLAUDE.md` is a symlink to this file.

The rationale behind these rules (why the reward curve lies, why seeds, why strict schemas)
lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#design-rules-that-shaped-the-code).

## Read The Docs First

| Topic                                                | Doc                                          |
| ---------------------------------------------------- | -------------------------------------------- |
| Architecture, data flow, module map, Modal execution | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Training a GRPO arm on Modal, reproducing results    | [docs/RUNBOOK.md](docs/RUNBOOK.md)           |
