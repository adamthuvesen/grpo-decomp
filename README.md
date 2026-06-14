# grpo-gain-decomposition

How much of an LLM's RL "reasoning" gain is real? `grpo-gain-decomposition` trains a small language model with GRPO (the DeepSeek-R1-Zero
recipe) on grade-school math, then runs an adversarial evaluation that
decomposes the benchmark gain into its parts: genuine learning vs. contamination,
answer formatting, and elicitation of capability the base model already had.

## The question

RLVR (RL with verifiable rewards) produces large, cheap gains on math benchmarks,
but several papers argue the gain is "illusory":

- Random rewards still lift Qwen models on math ([Spurious Rewards, 2506.10947](https://arxiv.org/abs/2506.10947))
- The base model often matches RL at high pass@k ([Yue et al., 2504.13837](https://arxiv.org/abs/2504.13837))
- Benchmarks are contaminated in modern base models ([Wu et al., 2507.10532](https://arxiv.org/abs/2507.10532))

While others show RL does expand reasoning under the right conditions
([ProRL, 2505.24864](https://arxiv.org/abs/2505.24864)) and that the pass@k yardstick
itself is flawed ([CoT-Pass@K, 2506.14245](https://arxiv.org/abs/2506.14245)).

`grpo-gain-decomposition` builds the measurement that decomposes
the gain, on the exact models the debate centers on (Qwen). v1 reproduces and
decomposes the gain within Qwen (6 seeds per arm); the cross-family control (Llama)
that tests whether a result is real or a Qwen artifact is the follow-up.

## Status

### GSM8K: Real Gain, Mostly Elicitation

**GSM8K study complete: 6 seeds per arm.** The committed results are in
[`results/`](results/FINDINGS.md).

Correct reward beats random reward by **+3.9 pp, 95% CI [2.3, 5.6]**. That is a real,
statistically clear gain, but the mechanism is mostly reliability on problems the base model
could already solve:

- base pass@8 (94.0%) is already above correct pass@1 (76.2%).
- pass@8 coverage barely moves after RL: Δ +0.7 pp, 95% CI [−0.4, +1.9].
- Per problem, **0.0% of the GSM8K gain is new capability**.
- On renumbered GSM-Symbolic problems, base pass@8 holds at 90.8%.
- CoT-gated pass@k is not useful here: Qwen has **0.0% verifiable-chain coverage** in these completions.

![GSM8K decomposition](results/fig-gsm8k-decomposition.svg)

### Positive Control: Countdown Expands

Countdown is the counterexample that makes the measurement credible. Same protocol, different
task: the base model lacks coverage, and RL really expands what it can solve. The placebo
comparison is **+46.5 pp, 95% CI [21.4, 71.6]**, while pass@8 coverage moves by
**Δ +41.0 pp, 95% CI [38.3, 43.7]**: base 53.6% → correct 94.6%. Per problem,
**10.9% on Countdown** is genuinely new capability.

![Countdown expansion](results/fig-task-contrast.svg)

That two-sidedness is the point. The instrument reports elicitation on a saturated benchmark
and expansion when coverage actually moves. The cross-family Llama and format-reward arms remain
follow-ups.

Everything that runs on CPU is built and tested: data loaders, reward functions, the eval
battery, paired statistics, the decomposition report, the GRPO training launcher + Modal
runner, and the completion-generation backend (transformers on CPU/MPS · vLLM on GPU) behind
the `grpo-decomp` CLI.

## Architecture

How the pieces fit is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): modules
and their one-way dependencies, the data flow, the GRPO training loop, the reward functions,
the decomposition and statistics, and the Modal execution model, all with diagrams.

## Usage

```bash
make install           # CPU env: data, rewards, eval, stats, report
make check             # ruff + unit tests (the Phase-0 check)
make test-integration  # loads the pinned datasets from HuggingFace

uv sync --extra train  # GPU stack (Linux/CUDA)
modal run modal_app.py --arm configs/correct.yaml  # one training arm on an A100
```

The full Phase-1 sequence is in [RUNBOOK.md](RUNBOOK.md): Modal auth, the W&B secret, the
day-1 smoke, both arms, and the held-out accuracy curve.

Evaluation runs through `grpo-decomp`: `generate` (the only model-loading step) writes a
`CompletionSet`; `battery` and `report` read it on a cheap CPU box, no backend needed.

```bash
uv sync --extra generate  # CPU/MPS generation backend (transformers; no CUDA needed)

# Phase-0 base-model smoke: sample the base model, then score it end-to-end on CPU.
grpo-decomp generate --model Qwen/Qwen2.5-Math-1.5B --set dev --backend transformers --out runs/base__dev
grpo-decomp battery  --completions runs/base__dev --k 1

# Phase-2: generate each arm on a GPU box (--backend vllm, high --n), then decompose on CPU.
grpo-decomp generate --model <correct-ckpt> --set gsm8k-test --backend vllm --n 256 --temperature 0.8 --out runs/correct__gsm8k-test
grpo-decomp report   --completions-dir runs/ --out results/   # <arm>__<set> dirs -> table + summary.json
```

## Reproduce

Trained checkpoints are not in this repo — they live on the Modal `assay-runs` volume, so
re-running generation means training the arms (RUNBOOK) or pulling the completions back. The
committed `results/` JSON makes that unnecessary for verifying the numbers:

```bash
make results   # regenerate the figures from results/*.json + check every headline number traces to its JSON
```

`make results` needs no GPU and no Modal account. To re-derive the JSON from the raw completions
(needs the volume) see [RUNBOOK.md](RUNBOOK.md) → "Reproduce the decomposition".

## Stack

Python 3.11+ · `uv` · TRL (GRPO) · vLLM (rollouts) · single-GPU (A100/H100-80GB) ·
[`eval-audit`](https://github.com/adamthuvesen/eval-audit) for the statistics layer.

## License

MIT
