"""The `countdown` reward: verifiable search correctness for the positive-control task."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from llm_grpo_gains.data.countdown import is_valid_countdown_solution, parse_countdown_key
from llm_grpo_gains.eval.answers import extract_strict

logger = logging.getLogger(__name__)

#: Warn (not debug) when more than this fraction of a batch is unparseable.
_UNPARSEABLE_WARN_RATE = 0.05


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
    rewards: list[float] = []
    unparseable = 0
    for completion, gold in zip(completions, gold_answer, strict=True):
        expression = extract_strict(completion)
        if expression is None:
            unparseable += 1
            rewards.append(0.0)
            continue
        numbers, target = parse_countdown_key(gold)
        rewards.append(1.0 if is_valid_countdown_solution(expression, numbers, target) else 0.0)

    if unparseable:
        rate = unparseable / len(rewards)
        log = logger.warning if rate > _UNPARSEABLE_WARN_RATE else logger.debug
        log(
            "countdown: %d/%d (%.0f%%) completions had no boxed expression",
            unparseable,
            len(rewards),
            rate * 100,
        )
    return rewards
