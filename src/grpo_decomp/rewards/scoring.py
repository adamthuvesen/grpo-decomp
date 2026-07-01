"""Shared strict-boxed scoring for training rewards."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from grpo_decomp.grading import extract_strict

# Warn when more than this fraction of a batch is unparseable (reward-hacking signal).
UNPARSEABLE_WARN_RATE = 0.05


def log_unparseable_fraction(logger: logging.Logger, label: str, count: int, total: int) -> None:
    """Log once per batch when a meaningful fraction of completions is unparseable."""
    rate = count / total
    log = logger.warning if rate > UNPARSEABLE_WARN_RATE else logger.debug
    log(
        "%s: %d/%d (%.0f%%) completions had no boxed answer",
        label,
        count,
        total,
        rate * 100,
    )


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
