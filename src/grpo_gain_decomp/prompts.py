"""The single prompt format shared by training and evaluation.

Training and eval MUST draw completions on the *same* prompt, or the decomposition
measures the model on a distribution it never trained on. Both
`grpo_gain_decomp.train.launcher` and `grpo_gain_decomp.eval.generate` build prompts from here.
"""

from __future__ import annotations

#: R1-Zero-style prompt. The base model has no chat template, so the answer format
#: is supplied explicitly — and it is the same ``\boxed{}`` strict eval extracts.
PROMPT_TEMPLATE = "{question}\n\nReason step by step, and put your final answer within \\boxed{{}}."


def build_prompt(question: str) -> str:
    """Wrap a question in the explicit reason-then-box prompt."""
    return PROMPT_TEMPLATE.format(question=question)
