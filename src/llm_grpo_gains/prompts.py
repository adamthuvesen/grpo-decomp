"""The single prompt format shared by training and evaluation.

Training and eval MUST draw completions on the *same* prompt, or the decomposition
measures the model on a distribution it never trained on. Both
`llm_grpo_gains.train.launcher` and `llm_grpo_gains.eval.generate` build prompts from here.
"""

from __future__ import annotations

from typing import Any

#: R1-Zero-style prompt. The base model has no chat template, so the answer format
#: is supplied explicitly — and it is the same ``\boxed{}`` strict eval extracts.
PROMPT_TEMPLATE = "{question}\n\nReason step by step, and put your final answer within \\boxed{{}}."

#: Generation budget for eval and held-out curves — matches `GRPOSettings.max_completion_length`.
EVAL_MAX_NEW_TOKENS = 1024


def build_prompt(question: str) -> str:
    """Wrap a question in the explicit reason-then-box prompt."""
    return PROMPT_TEMPLATE.format(question=question)


def prepare_qwen_tokenizer(tokenizer: Any) -> None:
    """Default the pad token to EOS and left-pad for batched generation.

    v1 trains the base model on a raw (non-chat) prompt, so its **native** EOS
    (``<|endoftext|>`` for Qwen2.5-Math) is the correct terminator — and the one
    vLLM colocate stops on, since it loads the model's own ``generation_config``.
    Do not override EOS to a chat-template token: that would desync the trainer's
    stop token from vLLM's, and with ``mask_truncated_completions`` it silently
    masks the gradient.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
