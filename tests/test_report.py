"""Unit tests for the decomposition report (no network)."""

from __future__ import annotations

import pytest

from grpo_decomp.report.decomposition import DecompositionRow, build_decomposition
from grpo_decomp.report.render import render_table, to_summary_json, write_summary
from grpo_decomp.report.status import (
    MIN_HEADLINE_SEEDS,
    artifact_scope_for,
    is_preliminary_seed_count,
    preliminary_suffix,
)
from grpo_decomp.stats.compare import Comparison


def _row(
    control: str, probes: str, delta: float, *, p: float = 0.01, n: int = 100
) -> DecompositionRow:
    comparison = Comparison(
        label_a="base",
        label_b="rl",
        n=n,
        accuracy_a=0.5,
        accuracy_b=0.5 + delta,
        delta=delta,
        ci_low=delta - 0.02,
        ci_high=delta + 0.02,
        p_value=p,
        n_discordant=20,
        test="exact-binomial",
    )
    return DecompositionRow(control=control, probes=probes, comparison=comparison)


def _build(seeds: int):
    return build_decomposition(
        base_model="Qwen2.5-Math-1.5B",
        task="GSM8K",
        seeds=seeds,
        raw_gain=_row("raw gain", "RL correct vs base on GSM8K", 0.10),
        control_rows=[
            _row("contamination (GSM-Symbolic)", "memorization", 0.06),
            _row("label noise (GSM8K-Platinum)", "label noise", 0.09),
            _row("robustness (GSM-Plus)", "perturbation robustness", 0.04),
        ],
        format_row=_row("format sensitivity", "lenient vs strict", 0.03),
        placebo=_row("placebo (correct - random)", "non-correctness-driven gain", 0.05),
        elicitation_note="base pass@256 matches RL pass@1 (elicitation, not new reasoning)",
    )


def test_preliminary_flag_tracks_min_seeds() -> None:
    assert _build(1).preliminary is True
    assert _build(MIN_HEADLINE_SEEDS - 1).preliminary is True
    assert _build(MIN_HEADLINE_SEEDS).preliminary is False


def test_claim_status_helpers_are_the_shared_policy() -> None:
    assert is_preliminary_seed_count(1) is True
    assert is_preliminary_seed_count(MIN_HEADLINE_SEEDS) is False
    assert "Single-seed descriptive decomposition" in artifact_scope_for(1)
    assert "Seed-aggregated decomposition" in artifact_scope_for(MIN_HEADLINE_SEEDS)
    assert preliminary_suffix(True) == " [PRELIMINARY]"
    assert preliminary_suffix(False) == ""


def test_caveats_cover_preliminary_nonadditivity_and_within_qwen() -> None:
    caveats = _build(1).caveats
    assert any("PRELIMINARY" in c for c in caveats)
    assert any("do not sum" in c for c in caveats)
    assert any("within-Qwen lower bound" in c for c in caveats)
    # The multiple-comparisons guard: rows are descriptive with marginal CIs.
    assert any("descriptive" in c and "marginal" in c for c in caveats)


def test_artifact_scope_marks_single_seed_reports_as_diagnostic() -> None:
    scope = _build(1).artifact_scope
    assert "Single-seed descriptive decomposition" in scope
    assert "per-seed diagnostic" in scope
    assert "seed-placebo-comparison.json" in scope


def test_build_rejects_non_positive_seeds() -> None:
    with pytest.raises(ValueError, match="seeds must be"):
        _build(0)


def test_summary_json_is_byte_identical_for_equal_inputs() -> None:
    assert to_summary_json(_build(3)) == to_summary_json(_build(3))


def test_render_table_is_deterministic_and_has_panels() -> None:
    first = render_table(_build(1))
    assert first == render_table(_build(1))
    assert "[PRELIMINARY]" in first
    assert "Artifact scope: Single-seed descriptive decomposition" in first
    assert "Confirmatory comparison: rl beats base by" in first
    assert "Elicitation (separate panel):" in first
    assert "| Control | Probes |" in first


def test_render_omits_preliminary_with_enough_seeds() -> None:
    assert "[PRELIMINARY]" not in render_table(_build(MIN_HEADLINE_SEEDS))


def test_write_summary_writes_the_serialized_json(tmp_path) -> None:
    decomposition = _build(3)
    out = tmp_path / "summary.json"
    write_summary(decomposition, out)
    assert out.read_text(encoding="utf-8") == to_summary_json(decomposition)
