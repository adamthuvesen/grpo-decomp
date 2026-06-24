"""Assemble the headline decomposition from paired comparisons.

The table re-measures the raw GSM8K gain under each control **independently** —
rows are overlapping lenses on one gain, never an additive partition. The placebo
(correct - random) delta is the confirmatory comparison; the pass@k / CoT result is a
directional verdict in its own panel, not a percentage-point row. A result over
fewer than three seeds is flagged preliminary, with CIs reflecting eval-sampling
noise only — not a headline claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from llm_grpo_gains.report.status import (
    MIN_HEADLINE_SEEDS,
    artifact_scope_for,
    is_preliminary_seed_count,
    preliminary_caveat_for,
)
from llm_grpo_gains.schemas import Record
from llm_grpo_gains.stats.compare import Comparison

#: Minimum seeds before a result is a headline claim rather than preliminary.
MIN_SEEDS = MIN_HEADLINE_SEEDS


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
        description="The pass@k / CoT verdict — a separate panel, not a row."
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
    """Assemble the decomposition, flagging it preliminary below `MIN_SEEDS` seeds."""
    preliminary = is_preliminary_seed_count(seeds)
    artifact_scope = artifact_scope_for(seeds)
    caveats = [
        "Rows are independent re-measurements of the raw gain under each control; "
        "they overlap and MUST NOT be summed into an additive partition.",
        "The placebo (correct - random) delta is a within-Qwen lower bound on "
        "non-correctness-driven gain, not a cross-family artifact verdict "
        "(needs the v2 Llama arm).",
        "Only the placebo comparison is the pre-registered confirmatory test; every table "
        "row is descriptive/exploratory and its 95% CI is marginal (per-row), NOT "
        "family-wise corrected — do not read a single row's p<0.05 as confirmed.",
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
