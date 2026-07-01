"""The `correct` reward: verifiable exact-match correctness, no partial credit."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from grpo_decomp.grading import is_correct
from grpo_decomp.rewards import score_strict_boxed

logger = logging.getLogger(__name__)


def correct(
    completions: Sequence[str], gold_answer: Sequence[str], **kwargs: object
) -> list[float]:
    """Score each completion 1.0 iff its boxed answer mathematically equals the
    gold, else 0.0. No partial credit.

    The answer is read from the final ``\\boxed{...}`` via the *same* `extract_strict`
    the eval grader uses for headline strict accuracy, so the training reward and the
    decomposition agree on extraction. Grading is delegated to math-verify via
    `is_correct` (so ``1,000`` == ``1000`` and ``3/4`` == ``0.75``).

    Unparseable completions score **0.0** (treated as wrong), not skipped: this is
    the *training* reward, and under ``beta=0.0`` (no KL anchor) skipping would
    remove downward pressure on degenerate output. The per-call unparseable count
    is logged as a reward-hacking warning.

    `completions` are generated answer strings; `gold_answer` is the matching
    normalized gold per completion, forwarded by the TRL trainer from the dataset.
    """
    return score_strict_boxed(
        completions,
        gold_answer,
        logger=logger,
        reward_name="correct",
        is_correct=is_correct,
    )
