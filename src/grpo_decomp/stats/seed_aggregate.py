"""Seed-level mean and t-interval aggregation shared by report aggregators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import t

#: Two-sided 95% t critical value multiplier (df supplied per call).
T_CRIT_ALPHA = 0.975


def seed_level_mean_ci(
    deltas: Sequence[float],
    *,
    single_seed_ci: tuple[float, float] | None = None,
) -> tuple[float, float | None, float, float, str]:
    """Return ``(mean, sem, ci_low, ci_high, ci_kind)`` over per-seed deltas.

    With >=2 seeds the CI is a t-interval over the deltas (run-to-run variance).
    At one seed, falls back to ``single_seed_ci`` when provided.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        raise ValueError("no deltas to aggregate")
    mean = float(arr.mean())
    n = len(arr)
    if n >= 2:
        sem = float(arr.std(ddof=1) / np.sqrt(n))
        half = float(t.ppf(T_CRIT_ALPHA, n - 1)) * sem
        return mean, sem, mean - half, mean + half, f"seed-level t, df={n - 1}"
    if single_seed_ci is None:
        raise ValueError("single_seed_ci required when aggregating one seed")
    return (
        mean,
        None,
        single_seed_ci[0],
        single_seed_ci[1],
        "single-seed eval bootstrap (no seed variance)",
    )


def seed_level_t_half_width(values: Sequence[float]) -> tuple[float, str]:
    """Half-width of a seed-level t interval for ``mean(values)``."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n >= 2:
        half = float(t.ppf(T_CRIT_ALPHA, n - 1)) * float(arr.std(ddof=1) / np.sqrt(n))
        return half, f"seed-level t, df={n - 1}"
    return 0.0, "single-seed (no seed variance)"
