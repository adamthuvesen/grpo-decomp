"""Aggregate the section-3 control rows across seed replicates, with family-wise correction.

The seed-0 controls (contamination / robustness / label-noise) are descriptive: each row's CI is
marginal and its McNemar p is per-row, not family-wise corrected. This upgrades them to
confirmatory grade — the same seed-level aggregation the placebo comparison and pass@k panel use
— by computing the base-vs-correct delta per seed (base seed-independent, correct per training
seed), a seed-level t CI, a one-sample t p-value per row, and Holm-Bonferroni across the rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import Field
from scipy.stats import t, ttest_1samp

from grpo_gain_decomp.report.decomposition import MIN_SEEDS
from grpo_gain_decomp.schemas import Record
from grpo_gain_decomp.stats.compare import Comparison
from grpo_gain_decomp.stats.significance import holm_correction


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
    preliminary: bool = Field(description=f"True below {MIN_SEEDS} seeds.")

    def headline(self) -> str:
        """The atomic claim: how many control rows survive family-wise correction."""
        tag = f"  [PRELIMINARY <{MIN_SEEDS} seeds]" if self.preliminary else ""
        sig = sum(row.significant for row in self.rows)
        return (
            f"{self.task}: {len(self.rows)} control rows over {self.n_seeds} seeds, "
            f"{sig}/{len(self.rows)} significant after Holm (FWER){tag}"
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
    ci_kind = (
        f"seed-level t, df={n_seeds - 1}"
        if n_seeds >= 2
        else "single-seed eval bootstrap (no seed variance)"
    )

    built: list[dict] = []
    raw_p: list[float] = []
    for control, probes, comparisons in rows:
        if len(comparisons) != n_seeds:
            raise ValueError(
                f"control {control!r}: {len(comparisons)} comparisons but {n_seeds} seeds"
            )
        deltas = np.array([c.delta for c in comparisons], dtype=float)
        mean = float(deltas.mean())
        if n_seeds >= 2:
            half = float(t.ppf(0.975, n_seeds - 1)) * float(deltas.std(ddof=1) / np.sqrt(n_seeds))
            ci_low, ci_high = mean - half, mean + half
            p = float(ttest_1samp(deltas, 0.0).pvalue)
        else:
            ci_low, ci_high = comparisons[0].ci_low, comparisons[0].ci_high
            p = comparisons[0].p_value
        raw_p.append(p)
        built.append(
            {
                "control": control,
                "probes": probes,
                "per_seed_delta": tuple(float(d) for d in deltas),
                "per_seed_correct_acc": tuple(float(c.accuracy_b) for c in comparisons),
                "base_acc": float(comparisons[0].accuracy_a),
                "mean_delta": mean,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_value": p,
            }
        )

    holm = holm_correction(raw_p)
    control_rows = tuple(
        ControlRow(**b, n_seeds=n_seeds, p_value_holm=float(ph), significant=ph < 0.05)
        for b, ph in zip(built, holm, strict=True)
    )
    return ControlDecomposition(
        task=task,
        n_seeds=n_seeds,
        family_size=len(control_rows),
        ci_kind=ci_kind,
        rows=control_rows,
        preliminary=n_seeds < MIN_SEEDS,
    )
