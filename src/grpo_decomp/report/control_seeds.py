"""Aggregate the section-3 control rows across seed replicates, with family-wise correction.

The seed-0 controls (contamination / robustness / label-noise) are descriptive: each row's CI is
marginal and its McNemar p is per-row, not family-wise corrected. This version uses the same
seed-level aggregation as the placebo comparison and pass@k panel: base-vs-correct delta per
seed (base seed-independent, correct per training seed), a seed-level t CI, a one-sample t
p-value per row, and Holm-Bonferroni across the rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import ttest_1samp

from grpo_decomp.registries import default_task_set
from grpo_decomp.report.evidence import AnalysisProvenance, seed_labels
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.report.validation import Effect, PositiveCount, Rate, interval, mean, require_close
from grpo_decomp.schemas import Record
from grpo_decomp.stats.compare import Comparison
from grpo_decomp.stats.seed_aggregate import seed_level_mean_ci
from grpo_decomp.stats.significance import holm_correction


def _control_p_value(deltas: Sequence[float]) -> float:
    values = np.asarray(deltas, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    if np.allclose(values, values[0]):
        return 0.0
    return float(ttest_1samp(values, 0.0).pvalue)


class ControlRow(Record):
    """One control set's base-vs-correct gain, aggregated over seeds and Holm-corrected."""

    control: str = Field(description="The control set, e.g. 'gsm-symbolic'.")
    probes: str
    n_seeds: PositiveCount
    per_seed_delta: tuple[Effect, ...] = Field(description="correct(seed) - base, per seed.")
    per_seed_correct_acc: tuple[Rate, ...]
    base_acc: Rate = Field(description="Seed-independent base accuracy on this control.")
    mean_delta: Effect
    ci_low: float = Field(description="Seed-level t CI low on the mean delta.")
    ci_high: float
    p_value: Rate = Field(description="One-sample t-test on the per-seed deltas (H0: delta=0).")
    p_value_holm: Rate = Field(description="Holm-Bonferroni corrected across the control rows.")
    significant: bool = Field(description="p_value_holm < 0.05.")

    @model_validator(mode="after")
    def _consistent(self) -> ControlRow:
        if (
            len(self.per_seed_delta) != self.n_seeds
            or len(self.per_seed_correct_acc) != self.n_seeds
        ):
            raise ValueError("control observations must align with n_seeds")
        for delta, acc in zip(self.per_seed_delta, self.per_seed_correct_acc, strict=True):
            require_close("per_seed_delta", delta, acc - self.base_acc)
        require_close("mean_delta", self.mean_delta, mean(self.per_seed_delta))
        interval("control", self.ci_low, self.ci_high)
        if self.n_seeds >= 2:
            require_close("p_value", self.p_value, _control_p_value(self.per_seed_delta))
        if self.significant != (self.p_value_holm < 0.05):
            raise ValueError("significant disagrees with the declared 0.05 Holm threshold")
        return self


class ControlDecomposition(Record):
    """The section-3 controls, multi-seeded and family-wise corrected."""

    task: str
    n_seeds: PositiveCount
    family_size: PositiveCount = Field(description="Number of control rows Holm corrects across.")
    ci_kind: str
    rows: tuple[ControlRow, ...]
    preliminary: bool = Field(description=f"True below {MIN_HEADLINE_SEEDS} seeds.")

    analyses: tuple[AnalysisProvenance, ...] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ControlDecomposition:
        if self.family_size != len(self.rows) or len({row.control for row in self.rows}) != len(
            self.rows
        ):
            raise ValueError("family_size must match unique control rows")
        if any(row.n_seeds != self.n_seeds for row in self.rows):
            raise ValueError("control rows must share n_seeds")
        if self.preliminary != (self.n_seeds < MIN_HEADLINE_SEEDS):
            raise ValueError("preliminary disagrees with n_seeds")
        for row, adjusted in zip(
            self.rows, holm_correction([row.p_value for row in self.rows]), strict=True
        ):
            require_close("p_value_holm", row.p_value_holm, adjusted)
        return self

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
    per_seed_delta: tuple[Effect, ...]
    per_seed_correct_acc: tuple[Rate, ...]
    base_acc: Rate
    mean_delta: Effect
    ci_low: float
    ci_high: float
    p_value: Rate
    ci_kind: str


def _aggregate_one_control(
    control: str, probes: str, comparisons: Sequence[Comparison], *, n_seeds: int
) -> _ControlRowStats:
    if len(comparisons) != n_seeds:
        raise ValueError(f"control {control!r}: {len(comparisons)} comparisons but {n_seeds} seeds")
    if len({(c.n, c.accuracy_a) for c in comparisons}) != 1:
        raise ValueError(f"control {control!r}: comparisons must share the same base anchor")
    deltas = np.array([comparison.delta for comparison in comparisons], dtype=float)
    single_ci = (comparisons[0].ci_low, comparisons[0].ci_high) if n_seeds == 1 else None
    mean, _sem, ci_low, ci_high, ci_kind = seed_level_mean_ci(
        deltas.tolist(), single_seed_ci=single_ci
    )
    p_value = _control_p_value(deltas.tolist()) if n_seeds >= 2 else comparisons[0].p_value
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
    task: str | None = None,
    analyses: tuple[AnalysisProvenance, ...] | None = None,
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

    seed_labels(seeds)
    if len({control for control, _, _ in rows}) != len(rows):
        raise ValueError("duplicate control rows change the test family")
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
        analyses=analyses,
        task=task or default_task_set() or "unspecified",
        n_seeds=n_seeds,
        family_size=len(control_rows),
        ci_kind=built[-1].ci_kind,
        rows=control_rows,
        preliminary=n_seeds < MIN_HEADLINE_SEEDS,
    )
