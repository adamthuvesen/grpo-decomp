"""Shared logging for unparseable training-reward batches."""

from __future__ import annotations

import logging

#: Warn when more than this fraction of a batch is unparseable (reward-hacking signal).
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
