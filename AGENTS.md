# AGENTS.md: grpo-decomp + llm_grpo_gains

This repo has **two packages and one boundary**:

- **`grpo_decomp`** (`src/grpo_decomp/`): the harness, a controls-first method for
  decomposing a GRPO benchmark gain into real reasoning vs. contamination, formatting, and
  elicitation. Task- and model-agnostic; knows nothing about GSM8K or Countdown. Tasks plug
  in through the registries in `grpo_decomp/registries.py`.
- **`llm_grpo_gains`** (`src/llm_grpo_gains/`): the reference study that exercises the
  harness: GSM8K (primary) plus a generated Countdown positive control. It supplies datasets,
  the `correct`/`countdown` rewards, configs, results, and a `registration.py` that wires
  itself into the harness via a `grpo_decomp.plugins` entry point.

The result is only as trustworthy as the controls. "Most of this gain was elicitation"
beats a confident, uncontrolled "RL works."

**Keep the boundary one-way.** The harness must never import the study (verified by a
standalone-import smoke). Put generic measurement machinery in `grpo_decomp`; put anything
GSM8K/Countdown-specific in `llm_grpo_gains`. The harness is a narrow measurement harness.
Add extension points (registry entries), not a framework.
User-level guidance (tone, principles, git etiquette) lives in the user's agent defaults
and is _not_ duplicated here.

## Quickstart

```bash
make install        # CPU env: data, rewards, eval, stats, report
make check          # ruff + unit tests + docs consistency
make demo           # score committed mini CompletionSets; no model load
make results        # rebuild figures from results/*.json + docs<->JSON consistency check

modal run --detach modal_app.py --arm configs/correct.yaml  # train one arm on an A100 (see docs/runbook.md)
grpo-decomp battery --completions runs/base__dev --k 1      # grade a CompletionSet (CPU)
grpo-decomp report  --completions-dir runs/ --out results/  # <arm>__<set> dirs -> table + summary.json
```

Only `generate` (and training) needs a model/GPU; `battery`, `report`, and `make results` run on CPU.

## Conventions

- Start from repo truth: read `README.md`, [`docs/architecture.md`](docs/architecture.md), `pyproject.toml`, and nearby tests before inventing patterns.
- Prefer the smallest coherent change; if a doc disagrees with code, fix the doc in the same change.
- **Controls before gains.** Never report a gain without its controls, a confidence interval, and a paired significance test.
- **The reward curve is not evidence.** Only held-out eval, completion length, and entropy are; treat a rising reward with suspicion.
- **Never claim from one run.** Aggregate over seeds; below three seeds, label preliminary.
- Determinism: fixed seeds, pinned dataset revisions, recorded model/commit/config/dependency hashes on every artifact.
- **Pin TRL.** Its GRPO defaults move between versions; re-check config at build time. Same for vLLM (`==0.17.1`, TRL 1.0.0's supported max).
- No new runtime deps without asking. Stack is TRL + transformers + vLLM + datasets (train) and `eval-audit` (stats); no broad RL/agent frameworks.
- Python 3.11+, `uv`, ruff (line length 100), strict Pydantic for result schemas. `CLAUDE.md` is a symlink to this file.
- **Respect the boundary.** Harness code imports only `grpo_decomp.*`; study code may import both. A new GSM8K/Countdown-shaped knob is a registry entry in `llm_grpo_gains/registration.py`, not an `if task == ...` in the harness.

The rationale behind these rules (why the reward curve lies, why seeds, why strict schemas)
lives in [`docs/architecture.md`](docs/architecture.md#design-rules-that-shaped-the-code).

## Plug In Your Own Model + Task

The harness already takes any model (`base_model` / `--model` is a free HF id or path) and
any task through registries. No fork is required:

1. Write a `register()` that calls `register_eval_set`, `register_train_dataset`,
   `register_reward`, `register_verifier` (if not boxed-math), and
   `register_validation_reconstructor` (for held-out selection). Optionally
   `register_prompt_strategy` for a chat-template model; the harness default is `r1_zero`.
2. Declare it under the `grpo_decomp.plugins` entry-point group so the CLI/Modal discover it.
3. Point an `ArmConfig` at your `base_model` + your registered `reward`/`dataset`/`prompt_strategy`.

`llm_grpo_gains/registration.py` is the worked example.

## Read the docs first

| Topic                                                | Doc                                          |
| ---------------------------------------------------- | -------------------------------------------- |
| Architecture, data flow, module map, Modal execution | [docs/architecture.md](docs/architecture.md) |
| Training a GRPO arm on Modal, reproducing results    | [docs/runbook.md](docs/runbook.md)           |
