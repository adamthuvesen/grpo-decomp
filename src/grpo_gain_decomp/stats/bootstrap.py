"""Paired bootstrap CI on an accuracy delta, delegated to eval-audit.

This is the portfolio touch: grpo_gain_decomp audits its own RL run with Adam's own eval
tooling. eval-audit resamples the shared examples (percentile bootstrap) and
reports the delta with a CI; we map our two correctness vectors onto its
``task_id`` + outcome frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_ci(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    *,
    n_iter: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(delta, ci_low, ci_high)`` for ``acc_b - acc_a`` (b vs a).

    eval-audit computes ``delta = mean(arm_a) - mean(arm_b)``, so we pass B as
    `arm_a` and A as `arm_b` to get ``acc_b - acc_a``.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired correctness vectors must align")
    if not correct_a:
        raise ValueError("empty correctness vectors")

    # Lazy import: polars + eval-audit live in the `eval` extra, so the rest of
    # grpo_gain_decomp.stats / grpo_gain_decomp.report stays importable on a core install.
    try:
        import polars as pl
        from eval_audit.stats import paired_task_bootstrap
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "paired_bootstrap_ci needs the 'eval' extra: uv sync --extra eval"
        ) from exc

    ids = [str(i) for i in range(len(correct_a))]
    frame_b = pl.DataFrame({"task_id": ids, "outcome": [float(x) for x in correct_b]})
    frame_a = pl.DataFrame({"task_id": ids, "outcome": [float(x) for x in correct_a]})
    result = paired_task_bootstrap(
        frame_b, frame_a, outcome="outcome", n_iter=n_iter, alpha=alpha, seed=seed
    )
    return result.delta_point_estimate, result.delta_ci_low, result.delta_ci_high


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_iter: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for the mean of `values`, via a one-sample
    percentile bootstrap (elements resampled with replacement).

    The base pass@k anchor is a single, seed-independent model, so it carries no
    training-seed variance — only problem-sampling uncertainty. Each element here is one
    problem's pass@k estimate; resampling problems puts a CI on the anchor so it is not
    treated as noiseless. Deterministic given `seed` (no polars / eval-audit needed —
    this is a plain one-sample mean, unlike the paired delta above).
    """
    if len(values) == 0:
        raise ValueError("empty values")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_iter, arr.size))].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
