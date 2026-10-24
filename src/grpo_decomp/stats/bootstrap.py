"""Paired bootstrap CI on an accuracy delta, delegated to eval-audit.

Maps paired per-problem correctness vectors onto eval-audit's resampling frame and
reports the delta with a percentile bootstrap CI.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Fixed seed for reproducible paired bootstrap CIs across runs.
PAIRED_BOOTSTRAP_SEED = 42

#: Bootstrap resamples per paired comparison.
BOOTSTRAP_ITERATIONS = 10_000


def paired_bootstrap_ci(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    *,
    n_iter: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(delta, ci_low, ci_high)`` for ``acc_b - acc_a`` (b vs a).

    eval-audit computes ``delta = mean(arm_a) - mean(arm_b)``, so we pass B as
    `arm_a` and A as `arm_b` to get ``acc_b - acc_a``.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired correctness vectors must align")
    if not correct_a:
        raise ValueError("empty correctness vectors")

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
    n_iter: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for the mean of `values`, via a one-sample
    percentile bootstrap (elements resampled with replacement).

    The base pass@k anchor is a single, seed-independent model, so it carries no
    training-seed variance — only problem-sampling uncertainty. Each element here is one
    problem's pass@k estimate; resampling problems puts a CI on the anchor so it is not
    treated as noiseless.
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


def seed_problem_bootstrap_ci(
    deltas: Sequence[Sequence[float]],
    *,
    n_iter: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for the grand mean of a seeds x problems matrix.

    Row = training seed, column = problem, one problem axis shared by every seed (a crossed
    design). Each cell is already a paired per-problem delta. A resample draws seeds and
    problems independently with replacement, so the interval carries run-to-run and
    problem-sampling uncertainty together. A seed-level t interval drops the second term and
    can leave the [-1, 1] range of a rate delta; a percentile interval of resampled means cannot.
    With one seed this reduces to a problem bootstrap.
    """
    arr = np.asarray(deltas, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("deltas must be a non-empty seeds x problems matrix")
    n_seeds, n_problems = arr.shape
    rng = np.random.default_rng(seed)
    seed_weights = rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds), size=n_iter)
    problem_weights = rng.multinomial(n_problems, np.full(n_problems, 1 / n_problems), size=n_iter)
    means = np.einsum("is,sp,ip->i", seed_weights, arr, problem_weights, optimize=True) / (
        n_seeds * n_problems
    )
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
