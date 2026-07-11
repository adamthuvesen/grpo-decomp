"""The study's verifiable rewards (the real signals the placebo is compared against)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from grpo_decomp.grading import is_correct
from grpo_decomp.rewards import score_strict_boxed
from llm_grpo_gains.data.countdown import countdown_is_correct

_CORRECT_LOGGER = logging.getLogger(f"{__name__}.correct")
_COUNTDOWN_LOGGER = logging.getLogger(f"{__name__}.countdown")


def correct(
    completions: Sequence[str], gold_answer: Sequence[str], **kwargs: object
) -> list[float]:
    """Score strict-boxed math answers as correct or wrong."""
    return score_strict_boxed(
        completions,
        gold_answer,
        logger=_CORRECT_LOGGER,
        reward_name="correct",
        is_correct=is_correct,
    )


def countdown(
    completions: Sequence[str], gold_answer: Sequence[str], **kwargs: object
) -> list[float]:
    """Score strict-boxed expressions against the Countdown rules."""
    return score_strict_boxed(
        completions,
        gold_answer,
        logger=_COUNTDOWN_LOGGER,
        reward_name="countdown",
        is_correct=countdown_is_correct,
    )


__all__ = ["correct", "countdown"]
