# grpo-decomp

![License](https://img.shields.io/github/license/adamthuvesen/grpo-decomp) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A tool for measuring GRPO gains. It asks one question: did reinforcement
learning teach the model new reasoning, or did it mostly make answers the base
model could already produce show up more reliably?

Two packages:

- `grpo_decomp`: the task-agnostic harness. It trains GRPO arms, samples and
  freezes completions as artifacts, grades them, and reports controlled comparisons.
- `llm_grpo_gains`: the reference study. It plugs GSM8K and a generated Countdown
  positive control into the harness.

It is not a general RL platform. It is a small measurement system built on controls,
confidence intervals, paired tests, and reproducible artifacts.

## Result

Three studies use the same controlled protocol: placebo, confidence intervals, and
paired tests. Each measures a different thing GRPO can do. Only one shows new reasoning.

**Reading the numbers:** _pass@1_ is reliability (does the base solve it on the first
try); _pass@8_ is coverage (can it solve it in _any_ of 8 tries). A gain in pass@1
without pass@8 is answers the model already had showing up more often, not new capability.

### 1 · GSM8K: reliability, not new coverage

_Qwen2.5-Math-1.5B · 6 seeds_

GRPO beats a random-reward placebo by **+3.9 pp, 95% CI [2.3, 5.6]**, real but modest.
Coverage barely moves: Δ +0.7 pp [-0.4, +1.9]. The answers were already latent, since
base pass@8 (94.0%) already exceeds correct pass@1 (76.2%), so **0.0% of the GSM8K gain
is new capability**. RL makes latent answers show up more reliably. Controls hold:
base pass@8 holds at 90.8% on renumbered GSM-Symbolic, with 0.0% verifiable-chain coverage
in the CoT-gated check.

### 2 · Countdown: real expansion (positive control)

_generated search task · 3 seeds_

Countdown is a task the base model can't cover, so the same protocol should catch real
capability. It does. Gain over placebo: **+46.5 pp, 95% CI [21.4, 71.6]**. This time
coverage moves with it: **Δ +41.0 pp, 95% CI [35.1, 46.9]** (base 53.6% → correct
94.6%). Per problem, 10.9% on Countdown is genuinely new. This is the control that
shows the GSM8K null is real.

### 3 · Esme-214M-RL: reward sharpens form

_trained from scratch · 6 seeds_

A different axis: not coverage or reliability, but whether outputs are even well-formed.
Seed-aggregated valid-expression rate is 85.4% for real reward vs 0.8% for the
random-reward placebo. Real reward separates from placebo on held-out Countdown validity:
**+84.7 pp, 95% CI [+54.6, +114.7]** across six training seeds. Exact-any also clears
across seeds: **+8.9 pp, 95% CI [+6.0, +11.7]**.

Full writeups, figures, and the decontamination / mechanism / CoT-gated checks:
[GSM8K](results/FINDINGS.md) · [Countdown](results/countdown/FINDINGS.md) ·
[Esme](results/esme-countdown/sampled_decomposition.md). Committed tables, JSON summaries,
and figures are under `results/`. The cross-repo
[Esme retrospective](https://github.com/adamthuvesen/esme-pretrain/blob/main/docs/retrospective.md)
places these results in the full pretrain-to-serving chain.

## How It Works

Training and analysis meet at a committed artifact boundary:

1. Train one arm with a recorded base model, reward, seed, dataset, and config.
2. Generate completions from the base model and trained checkpoints.
3. Freeze them as `CompletionSet` directories.
4. Grade and aggregate across seeds offline on CPU.

Only training and generation need a model backend. Once a `CompletionSet` exists,
analysis is deterministic: no GPU, network, or Hugging Face access.

The harness stays task-agnostic through registries in `grpo_decomp/registries.py`. The
study registers its datasets, rewards, verifiers, and prompt strategies through the
`grpo_decomp.plugins` entry point. Published runs used Modal for training/generation
and W&B for curves. The public repo ships the derived result JSON and figures, not the
run volumes or checkpoints. Module map and data flow: [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
make install   # CPU env: data, rewards, eval, stats, reports, tests
make check     # ruff + unit tests + docs consistency
make results   # rebuild committed figures from JSON, then docs <-> JSON check
make demo      # score two tiny committed CompletionSets; no model load
```

Generation and training need a model backend; everything else runs on CPU:

```bash
uv sync --extra generate
grpo-decomp generate --model Qwen/Qwen2.5-Math-1.5B --set dev --backend transformers --out runs/base__dev
grpo-decomp battery  --completions runs/base__dev --k 1
```

Run `grpo-decomp --help` for the full command set. To train an arm on Modal and
reproduce the published runs, see [docs/runbook.md](docs/runbook.md).

## Plug In Your Own Model And Task

`--model` / `ArmConfig.base_model` takes any Hugging Face id or local checkpoint path.
To add a task, write a `register()` that registers your eval set, train dataset, and
reward (plus a verifier and validation reconstructor for non-boxed-math tasks), then
declare it under the `grpo_decomp.plugins` entry point.
[src/llm_grpo_gains/registration.py](src/llm_grpo_gains/registration.py) is the worked
example.

## Study Rules

- No headline gain without controls, confidence intervals, and paired tests.
- Aggregate over seeds before making a claim; below three seeds, label it preliminary.
- Treat reward curves as training diagnostics, not evidence.
- Record dataset and model revisions, config, commit, dependency versions, seeds, and
  sampling settings on every artifact.
- Make skipped records, malformed artifacts, and unparseable completions visible.

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
