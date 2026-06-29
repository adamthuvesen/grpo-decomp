"""Aggregate the placebo comparison (correct - random) across seed replicates.

A single run's CI reflects eval-sampling noise only. The headline must also clear
run-to-run (seed) variance, so we compute the placebo delta per seed and aggregate at
the seed level: the mean delta with a t-based CI over seeds. Below `MIN_HEADLINE_SEEDS`
the seed-level interval is wide (or undefined at one seed) and the result stays preliminary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import Field

from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.schemas import Record
from grpo_decomp.stats.compare import Comparison
from grpo_decomp.stats.seed_aggregate import seed_level_mean_ci


class SeedPlaceboComparison(Record):
    """The placebo comparison aggregated over seed replicates."""

    task: str
    n_seeds: int
    seeds: tuple[str, ...] = Field(description="Per-seed labels, in input order.")
    per_seed_delta: tuple[float, ...] = Field(description="correct - random per seed.")
    per_seed_correct_acc: tuple[float, ...]
    per_seed_random_acc: tuple[float, ...]
    mean_delta: float
    sem: float | None = Field(description="Std error of the mean over seeds (None at n=1).")
    ci_low: float
    ci_high: float
    ci_kind: str = Field(description="How the CI was formed (seed-level t, or single-seed eval).")
    preliminary: bool = Field(description=f"True below {MIN_HEADLINE_SEEDS} seeds.")

    def headline(self) -> str:
        """The atomic seed-aggregated claim for the report header."""
        verb = "beats" if self.mean_delta >= 0 else "trails"
        tag = f"  [PRELIMINARY <{MIN_HEADLINE_SEEDS} seeds]" if self.preliminary else ""
        return (
            f"over {self.n_seeds} seed(s) on {self.task}, correct {verb} random by "
            f"{abs(self.mean_delta) * 100:.1f}% "
            f"(95% CI [{self.ci_low * 100:.1f}, {self.ci_high * 100:.1f}]; {self.ci_kind}){tag}"
        )


def aggregate_placebo_comparison(
    comparisons: Sequence[Comparison], seeds: Sequence[object], *, task: str = "gsm8k-test"
) -> SeedPlaceboComparison:
    """Aggregate per-seed placebo `Comparison`s into a seed-level mean delta + CI.

    Each `Comparison` is correct-vs-random on one seed (delta = acc_correct - acc_random).
    With >=2 seeds the CI is a t-interval over the per-seed deltas (capturing run-to-run
    variance); at 1 seed it falls back to that seed's eval-bootstrap CI, clearly labelled.
    A seed/comparison count mismatch raises an explicit error instead of zip-truncating.
    """
    if not comparisons:
        raise ValueError("no per-seed comparisons to aggregate")
    if len(comparisons) != len(seeds):
        raise ValueError(f"{len(comparisons)} comparisons but {len(seeds)} seed labels")

    deltas = np.array([c.delta for c in comparisons], dtype=float)
    n = len(deltas)
    single_ci = (comparisons[0].ci_low, comparisons[0].ci_high) if n == 1 else None
    mean, sem, ci_low, ci_high, ci_kind = seed_level_mean_ci(deltas, single_seed_ci=single_ci)

    return SeedPlaceboComparison(
        task=task,
        n_seeds=n,
        seeds=tuple(str(s) for s in seeds),
        per_seed_delta=tuple(float(d) for d in deltas),
        per_seed_correct_acc=tuple(float(c.accuracy_b) for c in comparisons),
        per_seed_random_acc=tuple(float(c.accuracy_a) for c in comparisons),
        mean_delta=mean,
        sem=sem,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        ci_kind=ci_kind,
        preliminary=n < MIN_HEADLINE_SEEDS,
    )
