"""Prompt strategies: how a task turns a question into a prompt and prepares the tokenizer.

Training and evaluation MUST draw completions on the *same* strategy, or the
decomposition measures the model on a distribution it never trained on. A strategy is
selected by name (``ArmConfig.prompt_strategy`` for training, ``--prompt-strategy`` for
``generate``) from the registry in :mod:`grpo_decomp.registries`.

The harness ships one built-in, ``r1_zero`` (the R1-Zero-style raw boxed prompt for a base
model with no chat template). A study or a new RL task registers its own — e.g. a
chat-template strategy for an instruct model — via ``register_prompt_strategy``.
"""

from __future__ import annotations

from typing import Any

from grpo_decomp.registries import PromptStrategy, register_prompt_strategy

#: Generation budget for eval and held-out curves — matches ``GRPOSettings.max_completion_length``.
EVAL_MAX_NEW_TOKENS = 1024

#: R1-Zero-style prompt. The base model has no chat template, so the answer format is
#: supplied explicitly — and it is the same ``\boxed{}`` strict eval extracts.
R1_ZERO_TEMPLATE = (
    "{question}\n\nReason step by step, and put your final answer within \\boxed{{}}."
)


def _r1_zero_build_prompt(question: str) -> str:
    """Wrap a question in the explicit reason-then-box prompt."""
    return R1_ZERO_TEMPLATE.format(question=question)


def _r1_zero_prepare_tokenizer(tokenizer: Any) -> None:
    """Default the pad token to EOS and left-pad for batched generation.

    A base model trained on a raw (non-chat) prompt should stop on its **native** EOS
    (``<|endoftext|>`` for Qwen2.5-Math) — the token vLLM colocate also stops on, since it
    loads the model's own ``generation_config``. Do not override EOS to a chat-template
    token: that desyncs the trainer's stop token from vLLM's, and with
    ``mask_truncated_completions`` it silently masks the gradient.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


R1_ZERO = PromptStrategy(
    name="r1_zero",
    build_prompt=_r1_zero_build_prompt,
    prepare_tokenizer=_r1_zero_prepare_tokenizer,
)

register_prompt_strategy(R1_ZERO)
