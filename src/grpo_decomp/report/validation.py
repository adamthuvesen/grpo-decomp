"""Measurement invariants for persisted report records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from grpo_decomp.report.evidence import seed_labels
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS

Rate = Annotated[float, Field(ge=0, le=1, strict=True)]
Effect = Annotated[float, Field(ge=-1, le=1, strict=True)]
Count = Annotated[int, Field(ge=0, strict=True)]
PositiveCount = Annotated[int, Field(gt=0, strict=True)]


def require_close(name: str, actual: float, expected: float) -> None:
    # Numerical agreement tolerates arithmetic rounding, not changes to study thresholds.
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{name}={actual} disagrees with observations ({expected})")


def interval(name: str, low: float | None, high: float | None) -> None:
    if (low is None) != (high is None):
        raise ValueError(f"{name} requires both interval bounds")
    if low is not None and high is not None and low > high:
        raise ValueError(f"{name}: lower bound {low} exceeds upper bound {high}")


def seed_axis(n: int, seeds: Sequence[str], preliminary: bool, *rows: Sequence[object]) -> None:
    seed_labels(seeds)
    if n != len(seeds) or any(len(row) != n for row in rows):
        raise ValueError("n_seeds, seed labels, and per-seed observations must have equal lengths")
    if preliminary != (n < MIN_HEADLINE_SEEDS):
        raise ValueError(f"preliminary must be {n < MIN_HEADLINE_SEEDS} for {n} seeds")


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("observations must be nonempty")
    return sum(values) / len(values)
