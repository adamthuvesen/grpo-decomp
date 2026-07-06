"""Assemble the headline decomposition from paired comparisons.

The table re-measures the raw GSM8K gain under each control **independently**.
Rows are overlapping lenses on one gain, never an additive partition. The placebo
(correct - random) delta is the confirmatory comparison; the pass@k / CoT result is a
directional verdict in its own panel, not a percentage-point row. A result over
fewer than three seeds is flagged preliminary, with CIs reflecting eval-sampling
noise only, so it is not a headline claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from grpo_decomp.eval.battery import BatteryResult, run_battery
from grpo_decomp.eval.completions import CompletionSet
from grpo_decomp.registries import PROBES
from grpo_decomp.report.inputs import greedy_pass1
from grpo_decomp.report.status import (
    artifact_scope_for,
    is_preliminary_seed_count,
    preliminary_caveat_for,
)
from grpo_decomp.schemas import Record
from grpo_decomp.stats.compare import Comparison, compare


class DecompositionRow(Record):
    """One control re-measured against the raw gain, with its controlled interpretation."""

    control: str = Field(
        description="Row label, e.g. 'raw gain' or 'contamination (GSM-Symbolic)'."
    )
    probes: str = Field(description="What this control actually measures.")
    comparison: Comparison


class Decomposition(Record):
    """The headline decomposition table plus its confirmatory comparison and caveats."""

    base_model: str
    task: str
    seeds: int
    preliminary: bool
    artifact_scope: str = Field(
        description="Reader-facing warning about whether this is a seed-level claim or diagnostic."
    )
    confirmatory_comparison: DecompositionRow = Field(
        description="The placebo delta: the confirmatory contrast."
    )
    rows: tuple[DecompositionRow, ...] = Field(
        description="Independent re-measurements (non-additive)."
    )
    elicitation_note: str = Field(
        description="The pass@k / CoT verdict. This is a separate panel, not a row."
    )
    caveats: tuple[str, ...]


def build_decomposition(
    *,
    base_model: str,
    task: str,
    seeds: int,
    raw_gain: DecompositionRow,
    control_rows: Sequence[DecompositionRow],
    format_row: DecompositionRow,
    placebo: DecompositionRow,
    elicitation_note: str,
) -> Decomposition:
    """Assemble the decomposition, flagging it preliminary below `MIN_HEADLINE_SEEDS` seeds."""
    preliminary = is_preliminary_seed_count(seeds)
    artifact_scope = artifact_scope_for(seeds)
    caveats = [
        "Rows re-measure the raw gain under each control. They overlap, so do not "
        "sum them into an additive partition.",
        "The placebo (correct - random) delta is a within-Qwen lower bound on "
        "non-correctness-driven gain, not a cross-family verdict "
        "(needs the v2 Llama arm).",
        "Only the placebo comparison is the pre-registered confirmatory test. Every other "
        "row is descriptive, and its 95% CI is marginal (per-row), not family-wise "
        "corrected. Do not read a single row's p<0.05 as confirmed.",
    ]
    preliminary_caveat = preliminary_caveat_for(seeds)
    if preliminary_caveat is not None:
        caveats.append(preliminary_caveat)

    return Decomposition(
        base_model=base_model,
        task=task,
        seeds=seeds,
        preliminary=preliminary,
        artifact_scope=artifact_scope,
        confirmatory_comparison=placebo,
        rows=(raw_gain, *control_rows, format_row),
        elicitation_note=elicitation_note,
        caveats=tuple(caveats),
    )


def build_single_seed_decomposition(
    grouped: dict[str, dict[str, CompletionSet]],
    task: str,
    *,
    base_model: str | None = None,
) -> Decomposition:
    """Build the single-seed decomposition from loaded completion artifacts."""
    base = grouped[task]["base"]
    correct = grouped[task]["correct"]
    random_arm = grouped[task]["random"]

    base_lenient = greedy_pass1(base, "lenient")
    correct_lenient = greedy_pass1(correct, "lenient")

    raw_gain = DecompositionRow(
        control="raw gain",
        probes=f"correct vs base on {task}",
        comparison=compare("base", base_lenient, "correct", correct_lenient),
    )
    placebo = DecompositionRow(
        control="placebo (correct - random)",
        probes="non-correctness-driven gain",
        comparison=compare(
            "random", greedy_pass1(random_arm, "lenient"), "correct", correct_lenient
        ),
    )
    format_row = DecompositionRow(
        control="format sensitivity",
        probes="lenient vs strict (same completions)",
        comparison=compare(
            "correct/strict", greedy_pass1(correct, "strict"), "correct/lenient", correct_lenient
        ),
    )
    control_rows = [
        control_row(slug, arms)
        for slug, arms in sorted(grouped.items())
        if slug != task and "base" in arms and "correct" in arms
    ]

    return build_decomposition(
        base_model=base_model or base.provenance.model,
        task=task,
        seeds=1,
        raw_gain=raw_gain,
        control_rows=control_rows,
        format_row=format_row,
        placebo=placebo,
        elicitation_note=elicitation_note(base, correct),
    )


def control_row(slug: str, arms: dict[str, CompletionSet]) -> DecompositionRow:
    """Build one base-vs-correct control row from loaded completion artifacts."""
    comparison = compare(
        "base",
        greedy_pass1(arms["base"], "lenient"),
        "correct",
        greedy_pass1(arms["correct"], "lenient"),
    )
    return DecompositionRow(
        control=f"control ({slug})", probes=PROBES.get(slug, slug), comparison=comparison
    )


def battery_at(completion_set: CompletionSet, k_values: list[int]) -> BatteryResult:
    """Run the eval battery at exactly the requested k values."""
    return run_battery(
        completion_set.problem_set(), completion_set.completions_by_id(), k_values=k_values
    )


def vanilla_at(battery: BatteryResult, k: int) -> float:
    """The vanilla pass@k at exactly `k` from a battery result."""
    for entry in battery.pass_at_k:
        if entry.k == k:
            return entry.vanilla
    raise ValueError(f"pass@{k} was not computed")


def elicitation_note(base: CompletionSet, correct: CompletionSet) -> str:
    """The elicitation / capability-expansion panel line."""
    n_base = base.provenance.sampling.n
    n_correct = correct.provenance.sampling.n
    if n_base > 1 and n_correct > 1:
        k = min(n_base, n_correct)
        base_battery = battery_at(base, sorted({1, k}))
        correct_battery = battery_at(correct, sorted({1, k}))
        base_k = vanilla_at(base_battery, k)
        correct_k = vanilla_at(correct_battery, k)
        return (
            f"pass@k curve: base pass@{k}={base_k:.2f} vs correct pass@{k}={correct_k:.2f} "
            f"(Δ={correct_k - base_k:+.2f}); pass@1 base={base_battery.lenient_accuracy:.2f}, "
            f"correct={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    base_battery = battery_at(base, sorted({1, n_base}))
    correct_battery = battery_at(correct, [1])
    if n_base > 1:
        return (
            f"base pass@{n_base}={base_battery.pass_at_k[-1].vanilla:.2f} vs "
            f"correct pass@1={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    return (
        f"pass@1: base={base_battery.lenient_accuracy:.2f}, "
        f"correct={correct_battery.lenient_accuracy:.2f}; "
        "pass@k coverage is reported in the separate multi-seed panel "
        "(pass8-multiseed.json via grpo-decomp report-passk-seeds)"
    )
