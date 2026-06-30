"""Aggregate the section-3 control rows across seed replicates, with family-wise correction.

The seed-0 controls (contamination / robustness / label-noise) are descriptive: each row's CI is
marginal and its McNemar p is per-row, not family-wise corrected. This upgrades them to
confirmatory grade — the same seed-level aggregation the placebo comparison and pass@k panel use
— by computing the base-vs-correct delta per seed (base seed-independent, correct per training
seed), a seed-level t CI, a one-sample t p-value per row, and Holm-Bonferroni across the rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import Field
from scipy.stats import ttest_1samp

from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.schemas import Record
from grpo_decomp.stats.compare import Comparison
from grpo_decomp.stats.seed_aggregate import seed_level_mean_ci
from grpo_decomp.stats.significance import holm_correction


class ControlRow(Record):
    """One control set's base-vs-correct gain, aggregated over seeds and Holm-corrected."""

    control: str = Field(description="The control set, e.g. 'gsm-symbolic'.")
    probes: str
    n_seeds: int
    per_seed_delta: tuple[float, ...] = Field(description="correct(seed) - base, per seed.")
    per_seed_correct_acc: tuple[float, ...]
    base_acc: float = Field(description="Seed-independent base accuracy on this control.")
    mean_delta: float
    ci_low: float = Field(description="Seed-level t CI low on the mean delta.")
    ci_high: float
    p_value: float = Field(description="One-sample t-test on the per-seed deltas (H0: delta=0).")
    p_value_holm: float = Field(description="Holm-Bonferroni corrected across the control rows.")
    significant: bool = Field(description="p_value_holm < 0.05.")


class ControlDecomposition(Record):
    """The section-3 controls, multi-seeded and family-wise corrected."""

    task: str
    n_seeds: int
    family_size: int = Field(description="Number of control rows Holm corrects across.")
    ci_kind: str
    rows: tuple[ControlRow, ...]
    preliminary: bool = Field(description=f"True below {MIN_HEADLINE_SEEDS} seeds.")

    def headline(self) -> str:
        """The atomic claim: how many control rows survive family-wise correction."""
        tag = f"  [PRELIMINARY <{MIN_HEADLINE_SEEDS} seeds]" if self.preliminary else ""
        sig = sum(row.significant for row in self.rows)
        return (
            f"{self.task}: {len(self.rows)} control rows over {self.n_seeds} seeds, "
            f"{sig}/{len(self.rows)} significant after Holm (FWER){tag}"
        )


class _ControlRowStats(NamedTuple):
    control: str
    probes: str
    per_seed_delta: tuple[float, ...]
    per_seed_correct_acc: tuple[float, ...]
    base_acc: float
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    ci_kind: str


def _aggregate_one_control(
    control: str, probes: str, comparisons: Sequence[Comparison], *, n_seeds: int
) -> _ControlRowStats:
    if len(comparisons) != n_seeds:
        raise ValueError(f"control {control!r}: {len(comparisons)} comparisons but {n_seeds} seeds")
    deltas = np.array([comparison.delta for comparison in comparisons], dtype=float)
    single_ci = (comparisons[0].ci_low, comparisons[0].ci_high) if n_seeds == 1 else None
    mean, _sem, ci_low, ci_high, ci_kind = seed_level_mean_ci(deltas, single_seed_ci=single_ci)
    p_value = float(ttest_1samp(deltas, 0.0).pvalue) if n_seeds >= 2 else comparisons[0].p_value
    return _ControlRowStats(
        control=control,
        probes=probes,
        per_seed_delta=tuple(float(delta) for delta in deltas),
        per_seed_correct_acc=tuple(float(comparison.accuracy_b) for comparison in comparisons),
        base_acc=float(comparisons[0].accuracy_a),
        mean_delta=mean,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_value,
        ci_kind=ci_kind,
    )


def aggregate_control_rows(
    rows: Sequence[tuple[str, str, Sequence[Comparison]]],
    seeds: Sequence[object],
    *,
    task: str = "gsm8k-test",
) -> ControlDecomposition:
    """Aggregate per-control per-seed base-vs-correct `Comparison`s into a corrected table.

    Each input row is ``(control, probes, comparisons)`` where ``comparisons[i]`` is
    base-vs-correct on seed i (delta = acc_correct - acc_base; base is the same seed-0 anchor
    across seeds). Every row must carry one comparison per seed. Each row gets a seed-level t CI
    over its per-seed deltas and a one-sample t p-value; Holm-Bonferroni then corrects the
    p-values across the family of rows.
    """
    if not rows:
        raise ValueError("no control rows to aggregate")
    n_seeds = len(seeds)
    if n_seeds < 1:
        raise ValueError("no seeds")

    built = tuple(
        _aggregate_one_control(control, probes, comparisons, n_seeds=n_seeds)
        for control, probes, comparisons in rows
    )

    holm = holm_correction([row.p_value for row in built])
    control_rows = tuple(
        ControlRow(
            control=row.control,
            probes=row.probes,
            n_seeds=n_seeds,
            per_seed_delta=row.per_seed_delta,
            per_seed_correct_acc=row.per_seed_correct_acc,
            base_acc=row.base_acc,
            mean_delta=row.mean_delta,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            p_value=row.p_value,
            p_value_holm=float(corrected_p),
            significant=corrected_p < 0.05,
        )
        for row, corrected_p in zip(built, holm, strict=True)
    )
    return ControlDecomposition(
        task=task,
        n_seeds=n_seeds,
        family_size=len(control_rows),
        ci_kind=built[-1].ci_kind,
        rows=control_rows,
        preliminary=n_seeds < MIN_HEADLINE_SEEDS,
    )
