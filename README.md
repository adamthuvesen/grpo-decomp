# grpo-decomp

`grpo-decomp` is a measurement framework for GRPO gains. It asks a
simple question:

Did reinforcement learning teach the model new reasoning, or did it mostly make
answers the base model could already produce show up more reliably?

The repo contains two packages:

- **`grpo_decomp`**: the task-agnostic harness. It trains GRPO arms, samples
  completions, freezes them as artifacts, grades them, and reports controlled
  comparisons.
- **`llm_grpo_gains`**: the reference study. It plugs GSM8K and a generated
  Countdown positive control into the harness.

The harness is not a general RL platform. It is a small measurement system built
around controls, confidence intervals, paired tests, and reproducible artifacts.

## Current Result

On GSM8K with `Qwen/Qwen2.5-Math-1.5B`, a correctness reward gives a real but
modest gain over a random-reward placebo: **+3.9 pp, 95% CI [2.3, 5.6]** across
six seeds.

That gain is mostly reliability, not new coverage:

- The base already has the envelope: base pass@8 (94.0%) is above correct pass@1
  (76.2%).
- Coverage barely moves: Δ +0.7 pp, 95% CI [-0.4, +1.9].
- Per problem, 0.0% of the GSM8K gain is new capability; added reliability comes
  from problems inside the base pass@8 envelope.
- The envelope survives decontamination: on renumbered GSM-Symbolic problems,
  base pass@8 holds at 90.8%.
- CoT-gated pass@k is not informative here because these completions have 0.0%
  verifiable-chain coverage in the `<<a op b = c>>` format.

![GSM8K decomposition](results/fig-gsm8k-decomposition.svg)

Countdown is the positive control. On a generated search task where the base
model lacks coverage, the same protocol detects real expansion: the placebo
comparison is **+46.5 pp, 95% CI [21.4, 71.6]**, pass@8 moves by **Δ +41.0 pp,
95% CI [35.1, 46.9]**, and coverage goes from base 53.6% → correct 94.6%.
Per problem, 10.9% on Countdown is genuinely new capability.

![GSM8K vs Countdown](results/fig-task-contrast.svg)

The committed tables, JSON summaries, and figures live under `results/`.

## How It Works

Training and analysis are separated by a committed artifact boundary:

1. Train one arm with a recorded base model, reward, seed, dataset, and config.
2. Generate completions from the base model and trained checkpoints.
3. Freeze answers as `CompletionSet` directories.
4. Grade and report offline on CPU.

Only training and generation need a model backend. Once a `CompletionSet` exists,
analysis is deterministic and needs no GPU, network, or Hugging Face access.

The harness stays task-agnostic through registries in `grpo_decomp/registries.py`.
The study package registers datasets, rewards, verifiers, prompt strategies, and
task profiles through the `grpo_decomp.plugins` entry point.

For the module map and data flow, read [`docs/architecture.md`](docs/architecture.md).
For local checks and reproducibility commands, read [`docs/runbook.md`](docs/runbook.md).

## Install

```bash
make install
```

This installs the CPU environment for data loading, rewards, evaluation,
statistics, reports, and tests.

Optional extras:

```bash
uv sync --extra generate  # local transformers generation
uv sync --extra train     # GPU training stack: TRL + vLLM + W&B
```

## Check The Published Numbers

```bash
make results
```

This rebuilds the committed figures from JSON and runs the docs-to-JSON
consistency check. The normal local gate is:

```bash
make check
```

## Try The CPU Eval Path

`make demo` scores two tiny committed `CompletionSet` fixtures. It does not load a
model.

```bash
make demo
```

Expected strict accuracy:

- base fixture: `0.3333333333333333`
- correct fixture: `0.5`

## Generate A Completion Set

Local generation uses the optional `generate` extra:

```bash
uv sync --extra generate
grpo-decomp generate \
  --model Qwen/Qwen2.5-Math-1.5B \
  --set dev \
  --backend transformers \
  --out runs/base__dev
grpo-decomp battery --completions runs/base__dev --k 1
```

For GPU training on Modal:

```bash
modal run --detach modal_app.py --arm configs/correct.yaml
```

## Main CLI Commands

- `generate`: load a model and write a `CompletionSet`
- `battery`: score a `CompletionSet`
- `report`: build a single-seed decomposition table
- `report-seeds`: aggregate the placebo comparison
- `report-passk-seeds`: aggregate pass@k coverage
- `report-mechanism`: report per-problem migration vs. new capability
- `report-control-seeds`: aggregate control-set results
- `heldout`: score checkpoints on a validation split

Example report layout:

```bash
grpo-decomp report --completions-dir runs/ --out results/
```

`runs/` should contain directories named `<arm>__<set>`, such as
`base__gsm8k-test` and `correct__gsm-symbolic`.

## Plug In Your Own Model And Task

Point `--model` or `ArmConfig.base_model` at any Hugging Face id or local
checkpoint path, then register the task:

```python
from grpo_decomp.registries import register_eval_set, register_reward, register_train_dataset

def register() -> None:
    register_eval_set("my-test", load_my_test)
    register_train_dataset(TrainDataset("my-task", load_my_train_and_validation))
    register_reward("my-reward", lambda seed: my_reward)
```

Most tasks also register a verifier and validation reconstructor; chat-template
models can register a prompt strategy. Declare the function under the
`grpo_decomp.plugins` entry-point group so the CLI and Modal app discover it.

[`src/llm_grpo_gains/registration.py`](src/llm_grpo_gains/registration.py) is the
worked example.

## Study Rules

- No headline gain without controls, confidence intervals, and paired tests.
- Aggregate over seeds before making a claim.
- Treat reward curves as training diagnostics, not evidence.
- Record dataset revisions, model revisions, config, commit, dependency versions,
  seeds, and sampling settings.
- Make skipped records, malformed artifacts, and unparseable completions visible.

## Related Repositories

These repositories are separate codebases connected by model artifacts and
measurement questions:

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains
  `Esme-214M-Base` from scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts
  the base checkpoint with SFT, DPO, and verifier-backed RLVR.
- [`llm-infer`](https://github.com/adamthuvesen/llm-infer): loads, serves, and
  benchmarks exported Esme checkpoints.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): provides a reusable
  RLVR harness with text-to-SQL as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where
  GRPO gains come from, separating reliability from new capability.

## Stack

Python 3.11+ · `uv` · Pydantic · TRL GRPO · transformers · vLLM · Modal ·
[`eval-audit`](https://github.com/adamthuvesen/eval-audit)

## References

- Shao et al., [_DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_](https://arxiv.org/abs/2402.03300), 2024.
- Guo et al., [_DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning_](https://www.nature.com/articles/s41586-025-09422-z), 2025.
- Yang et al., [_Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement_](https://arxiv.org/abs/2409.12122), 2024.
- Cobbe et al., [_Training Verifiers to Solve Math Word Problems_](https://arxiv.org/abs/2110.14168), 2021.
- Mirzadeh et al., [_GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models_](https://arxiv.org/abs/2410.05229), 2024.
- Vendrow et al., [_Do Large Language Model Benchmarks Test Reliability?_](https://arxiv.org/abs/2502.03461), 2025.
- Wen et al., [_Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs_](https://arxiv.org/abs/2506.14245), 2025.
- Chen et al., [_Evaluating Large Language Models Trained on Code_](https://arxiv.org/abs/2107.03374), 2021.
- Pan, [_TinyZero_](https://github.com/Jiayi-Pan/TinyZero), 2025.

## License

MIT
