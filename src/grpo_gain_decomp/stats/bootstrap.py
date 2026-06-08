"""Paired bootstrap CI on an accuracy delta, delegated to eval-audit.

This is the portfolio touch: grpo_gain_decomp audits its own RL run with Adam's own eval
tooling. eval-audit resamples the shared examples (percentile bootstrap) and
reports the delta with a CI; we map our two correctness vectors onto its
``task_id`` + outcome frame.
"""

from __future__ import annotations

from collections.abc import Sequence


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
