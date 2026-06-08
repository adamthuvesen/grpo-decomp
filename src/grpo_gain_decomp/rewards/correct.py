"""The `correct` reward: verifiable exact-match correctness, no partial credit."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from math_verify import parse, verify

logger = logging.getLogger(__name__)

#: Warn (not debug) when more than this fraction of a batch is unparseable.
_UNPARSEABLE_WARN_RATE = 0.05


def correct(
    completions: Sequence[str], gold_answer: Sequence[str], **kwargs: object
) -> list[float]:
    """Score each completion 1.0 iff its final answer mathematically equals the
    gold, else 0.0. No partial credit.

    Verification is delegated to math-verify, which extracts the answer from the
    completion text and compares for numeric equivalence (so ``1,000`` == ``1000``
    and ``3/4`` == ``0.75``).

    Unparseable completions score **0.0** (treated as wrong), not skipped: this is
    the *training* reward, and under ``beta=0.0`` (no KL anchor) skipping would
    remove downward pressure on degenerate output. The per-call unparseable count
    is logged as a reward-hacking warning. (Eval-time extraction, which may skip,
    lives in the eval layer.)

    `completions` are generated answer strings; `gold_answer` is the matching
    normalized gold per completion, forwarded by the TRL trainer from the dataset.
    """
    rewards: list[float] = []
    unparseable = 0
    for completion, gold in zip(completions, gold_answer, strict=True):
        parsed_completion = parse(completion)
        if not parsed_completion:
            unparseable += 1
            rewards.append(0.0)
            continue
        rewards.append(1.0 if verify(parse(gold), parsed_completion) else 0.0)

    if unparseable:
        # Warning: log once a real fraction is unparseable (reward-hacking /
        # degeneration under beta=0.0), but stay quiet on the odd stray completion.
        rate = unparseable / len(rewards)
        log = logger.warning if rate > _UNPARSEABLE_WARN_RATE else logger.debug
        log(
            "correct: %d/%d (%.0f%%) completions had no parseable answer",
            unparseable,
            len(rewards),
            rate * 100,
        )
    return rewards
