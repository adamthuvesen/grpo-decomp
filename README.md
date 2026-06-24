# llm-grpo-gains

How much of an LLM's RL "reasoning" gain is real? `llm-grpo-gains` trains a small language model with GRPO (the DeepSeek-R1-Zero
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

`llm-grpo-gains` builds the measurement that decomposes
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
make demo              # score committed mini completions; no model load
make results           # rebuild figures from committed JSON and check docs
make test-integration  # loads the pinned datasets from HuggingFace

uv sync --extra train  # GPU stack (Linux/CUDA)
modal run modal_app.py --arm configs/correct.yaml  # one training arm on an A100
```

The full Phase-1 sequence is in [RUNBOOK.md](RUNBOOK.md): Modal auth, the W&B secret, the
day-1 smoke, both arms, and the held-out accuracy curve.

Evaluation runs through `grpo-decomp`:

- `generate` loads a model and writes a `CompletionSet`.
- `battery` scores one `CompletionSet`.
- `report` reads completion directories and writes the decomposition table.

Only `generate` needs a model backend. `battery`, `report`, and `make results` run on CPU.

```bash
uv sync --extra generate  # CPU/MPS generation backend (transformers; no CUDA needed)

# Phase-0 base-model smoke: sample the base model, then score it end-to-end on CPU.
grpo-decomp generate --model Qwen/Qwen2.5-Math-1.5B --set dev --backend transformers --out runs/base__dev
grpo-decomp battery  --completions runs/base__dev --k 1

# Phase-2: generate each arm on a GPU box (--backend vllm, high --n), then decompose on CPU.
grpo-decomp generate --model <correct-ckpt> --set gsm8k-test --backend vllm --n 256 --temperature 0.8 --out runs/correct__gsm8k-test
grpo-decomp report   --completions-dir runs/ --out results/   # <arm>__<set> dirs -> table + summary.json
```

## No-GPU Demo

`make demo` scores two committed mini `CompletionSet` fixtures. It does not load a model,
use the network, use Modal, or need a GPU.

```bash
make demo
```

Expected: base `strict_accuracy = 0.3333333333333333`; correct `strict_accuracy = 0.5`.
Both fixtures have 12 problems and 4 samples per problem.

## Verify Committed Results

Trained checkpoints are not in this repo. They live on the Modal `assay-runs` volume. The
committed `results/` JSON is enough to verify the published numbers:

```bash
make results
```

This rebuilds figures from `results/*.json` and checks that headline doc numbers trace to
JSON. It needs no GPU and no Modal account. It does not re-derive checkpoints or completions.

## Re-Derive From Checkpoints

Re-deriving `results/*.json` needs the off-repo checkpoints, full completion sets, Modal run
volume, and GPU generation path. See [RUNBOOK.md](RUNBOOK.md) → "Reproduce the decomposition".

## Stack

Python 3.11+ · `uv` · TRL (GRPO) · vLLM (rollouts) · single-GPU (A100/H100-80GB) ·
[`eval-audit`](https://github.com/adamthuvesen/eval-audit) for the statistics layer.

## License

MIT
