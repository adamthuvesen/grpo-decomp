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

**GSM8K study complete: 6 seeds per arm.** The result is committed under
[`results/`](results/FINDINGS.md): the placebo comparison (correct − random) is **+3.9 pp,
95% CI [2.3, 5.6]** (seed-level *t*, df=5): a small, correctness-driven, statistically
significant gain. But it is mostly elicitation, not new reasoning: base pass@8 (94.0%)
≫ correct pass@1 (76.2%), and pass@8 coverage barely moves — **Δ +0.7 pp** over 6 seeds
(propagated CI [−0.4, +1.9]: consistent with zero once the base anchor's own ±1.1 pp sampling
CI is folded in). RL made the model more *reliable* at problems the base could already solve. Two
confident single-seed numbers settled under aggregation: the placebo "+6.1 pp, p=3e-9" → +3.9 pp
[2.3, 5.6], and the pass@8 panel's "+1.7 pp" → +0.7 pp [−0.4, +1.9]. We also apply the CoT-gated
yardstick the last critique recommends: it has **0.0% verifiable-chain coverage** on these Qwen
completions (they reason in code, not `<<a op b=c>>` annotations), so it cannot discriminate base
from RL here — a proxy limitation we report in [FINDINGS](results/FINDINGS.md), not quietly drop.

| The confirmatory test | The elicitation finding |
| --- | --- |
| ![Placebo comparison over 6 seeds: +3.9 pp [2.3, 5.6]](results/fig-placebo-comparison.png) | ![RL improves pass@1 reliability while pass@8 coverage barely moves](results/fig-passk-curve.png) |

Everything that runs on CPU is built and tested: data loaders, reward functions, the eval
battery, paired statistics, the decomposition report, the GRPO training launcher + Modal
runner, and the completion-generation backend (transformers on CPU/MPS · vLLM on GPU) behind
the `grpo-decomp` CLI.

**The positive control: expansion, confirmed.** GSM8K is near-saturated for this base (base
pass@8 = 94%), so to prove the decomposition can *detect* genuine expansion, a second study
trains the **general `Qwen2.5-1.5B`** with GRPO on **Countdown** (a TinyZero-style search task
the base genuinely lacks). The result is committed under
[`results/countdown/`](results/countdown/FINDINGS.md): a **+46.5 pp** placebo comparison (95% CI
[21.4, 71.6], 3 seeds) and pass@8 coverage that **moves** — **Δ +41.0 pp, 95% CI [38.3, 43.7]**
(3 seeds): base 53.6% → correct 94.6%. Same decomposition, same protocol, opposite verdict from
GSM8K's. **Elicitation there, expansion here:**

![Pass@8 coverage: flat on GSM8K (elicitation), expanded on Countdown (expansion)](results/fig-passk-contrast.png)

That two-sidedness is the point. The instrument isn't biased toward "it's all fake": it
reports expansion when RL genuinely teaches new ability. The cross-family (Llama) and
format-reward arms remain follow-ups.

## Architecture

How the pieces fit is in [agents/docs/ARCHITECTURE.md](agents/docs/ARCHITECTURE.md): modules
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

## Stack

Python 3.11+ · `uv` · TRL (GRPO) · vLLM (rollouts) · single-GPU (A100/H100-80GB) ·
[`eval-audit`](https://github.com/adamthuvesen/eval-audit) for the statistics layer.

## License

MIT
