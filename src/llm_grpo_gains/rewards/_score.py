"""Shared reward scoring helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from llm_grpo_gains.eval.answers import extract_strict
from llm_grpo_gains.rewards._warn import log_unparseable_fraction


def score_strict_boxed(
    completions: Sequence[str],
    gold_answer: Sequence[str],
    *,
    logger: logging.Logger,
    reward_name: str,
    is_correct: Callable[[str, str], bool],
) -> list[float]:
    """Score boxed completions as 1.0/0.0, logging unparseable training samples."""
    rewards: list[float] = []
    unparseable = 0
    for completion, gold in zip(completions, gold_answer, strict=True):
        extracted = extract_strict(completion)
        if extracted is None:
            unparseable += 1
            rewards.append(0.0)
            continue
        rewards.append(1.0 if is_correct(extracted, gold) else 0.0)

    if unparseable:
        log_unparseable_fraction(logger, reward_name, unparseable, len(rewards))
    return rewards
