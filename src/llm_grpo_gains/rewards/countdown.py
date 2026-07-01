"""The `countdown` reward: verifiable search correctness for the positive-control task."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from grpo_decomp.rewards import score_strict_boxed
from llm_grpo_gains.data.countdown import countdown_is_correct

logger = logging.getLogger(__name__)


def countdown(
    completions: Sequence[str], gold_answer: Sequence[str], **kwargs: object
) -> list[float]:
    """Score each completion 1.0 iff its boxed expression is a legal Countdown solution
    (each source number used at most once, only ``+ - * /``, evaluating exactly to the
    target), else 0.0. No partial credit.

    The expression is read from the final ``\\boxed{...}`` via the *same* `extract_strict`
    the eval grader uses, so the training reward and the decomposition agree on extraction.
    Verification is the restricted, `eval`-free Countdown checker.

    As the training reward under ``beta=0.0``, an unparseable (no box) or invalid completion
    scores **0.0**, not skipped — skipping would remove downward pressure on degenerate
    output. The per-call unparseable count is logged as a reward-hacking warning. The
    matching `gold_answer` (the `(numbers, target)` key) is forwarded by the TRL trainer.
    """
    return score_strict_boxed(
        completions,
        gold_answer,
        logger=logger,
        reward_name="countdown",
        is_correct=countdown_is_correct,
    )
