"""One trained arm against the seed-independent base, paired per problem across seeds.

The placebo comparison (`report/seeds.py`) asks whether correct beats random. This asks a
narrower question of a single arm: did training move it off the base model at all? Each
seed's greedy pass@1 is paired to the base on every problem, and the interval resamples
seeds and problems together. The per-problem grades are stored in the artifact, so the
statistics can be recomputed without the CompletionSets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field, StrictBool, model_validator

from grpo_decomp.report.evidence import AnalysisProvenance, seed_labels
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.report.validation import (
    Effect,
    PositiveCount,
    Rate,
    interval,
    mean,
    require_close,
    seed_axis,
)
from grpo_decomp.schemas import Record
from grpo_decomp.stats.bootstrap import seed_problem_bootstrap_ci


class ArmVsBase(Record):
    """Greedy pass@1 of one arm over seeds, minus the base, with per-problem grades."""

    task: str
    arm: str = Field(description="The trained arm compared to base, e.g. 'random'.")
    n_seeds: PositiveCount
    seeds: tuple[str, ...] = Field(description="Per-seed labels, in input order.")
    problem_ids: tuple[str, ...] = Field(description="The shared problem axis, in order.")
    base_correct: tuple[StrictBool, ...] = Field(description="Base greedy grade per problem.")
    per_seed_arm_correct: tuple[tuple[StrictBool, ...], ...] = Field(
        description="Arm greedy grade per problem, one row per seed."
    )
    base_acc: Rate
    per_seed_arm_acc: tuple[Rate, ...]
    per_seed_delta: tuple[Effect, ...] = Field(description="arm(seed) - base, per seed.")
    mean_delta: Effect
    ci_low: float = Field(description="Seed x problem bootstrap CI low on the mean delta.")
    ci_high: float
    ci_kind: str
    preliminary: bool = Field(description=f"True below {MIN_HEADLINE_SEEDS} seeds.")

    analysis: AnalysisProvenance | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ArmVsBase:
        seed_axis(
            self.n_seeds,
            self.seeds,
            self.preliminary,
            self.per_seed_arm_correct,
            self.per_seed_arm_acc,
            self.per_seed_delta,
        )
        n = len(self.problem_ids)
        if not n or len(set(self.problem_ids)) != n or len(self.base_correct) != n:
            raise ValueError("problem_ids must be unique, nonempty, and align with base_correct")
        require_close("base_acc", self.base_acc, sum(self.base_correct) / n)
        for row, acc, delta in zip(
            self.per_seed_arm_correct, self.per_seed_arm_acc, self.per_seed_delta, strict=True
        ):
            if len(row) != n:
                raise ValueError("per_seed_arm_correct must align with problem_ids")
            require_close("per_seed_arm_acc", acc, sum(row) / n)
            require_close("per_seed_delta", delta, acc - self.base_acc)
        require_close("mean_delta", self.mean_delta, mean(self.per_seed_delta))
        interval("arm vs base", self.ci_low, self.ci_high)
        return self

    def headline(self) -> str:
        """The atomic claim: the arm's mean gain over base, with its paired interval."""
        tag = f"  [PRELIMINARY <{MIN_HEADLINE_SEEDS} seeds]" if self.preliminary else ""
        return (
            f"over {self.n_seeds} seed(s) on {self.task}, {self.arm} - base = "
            f"{self.mean_delta * 100:+.1f} pp (95% CI [{self.ci_low * 100:+.1f}, "
            f"{self.ci_high * 100:+.1f}]; {self.ci_kind}){tag}"
        )


def aggregate_arm_vs_base(
    base_grade: Mapping[str, bool],
    arm_grades: Sequence[tuple[object, Mapping[str, bool]]],
    *,
    task: str,
    arm: str,
    analysis: AnalysisProvenance | None = None,
) -> ArmVsBase:
    """Pair each seed's per-problem grades to the base and aggregate over seeds and problems."""
    if not arm_grades:
        raise ValueError("no per-seed arm grades to aggregate")
    labels = seed_labels([seed for seed, _ in arm_grades])
    problem_ids = tuple(base_grade)
    for seed, grades in arm_grades:
        if set(grades) != set(problem_ids):
            raise ValueError(f"seed {seed}: problem ids do not match the base")

    base_row = tuple(bool(base_grade[pid]) for pid in problem_ids)
    arm_rows = tuple(tuple(bool(grades[pid]) for pid in problem_ids) for _, grades in arm_grades)
    deltas = [[float(a) - float(b) for a, b in zip(row, base_row, strict=True)] for row in arm_rows]
    mean, ci_low, ci_high = seed_problem_bootstrap_ci(deltas)
    n_problems = len(problem_ids)
    base_acc = sum(base_row) / n_problems
    arm_accs = tuple(sum(row) / n_problems for row in arm_rows)
    return ArmVsBase(
        task=task,
        arm=arm,
        n_seeds=len(arm_rows),
        seeds=labels,
        analysis=analysis,
        problem_ids=problem_ids,
        base_correct=base_row,
        per_seed_arm_correct=arm_rows,
        base_acc=base_acc,
        per_seed_arm_acc=arm_accs,
        per_seed_delta=tuple(acc - base_acc for acc in arm_accs),
        mean_delta=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_kind=f"seed x problem bootstrap ({len(arm_rows)} seeds x {n_problems} problems)",
        preliminary=len(arm_rows) < MIN_HEADLINE_SEEDS,
    )
