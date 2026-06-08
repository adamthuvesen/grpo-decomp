"""Unit tests for the multi-seed control aggregation with Holm correction (no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.report.control_seeds import aggregate_control_rows
from grpo_gain_decomp.stats.compare import Comparison


def _cmp(base_acc: float, correct_acc: float) -> Comparison:
    """A base-vs-correct Comparison with a fixed delta (the aggregator reads delta/acc only)."""
    delta = correct_acc - base_acc
    return Comparison(
        label_a="base",
        label_b="correct",
        n=100,
        accuracy_a=base_acc,
        accuracy_b=correct_acc,
        delta=delta,
        ci_low=delta - 0.05,
        ci_high=delta + 0.05,
        p_value=0.01,
        n_discordant=10,
        test="chi2",
    )


def test_aggregate_control_rows_seed_level_ci_and_holm() -> None:
    # base is the same seed-0 anchor; correct varies by seed. Two control sets, two seeds.
    rows = [
        ("gsm-symbolic", "memorization", [_cmp(0.50, 0.60), _cmp(0.50, 0.70)]),  # deltas .10/.20
        ("gsm-plus", "robustness", [_cmp(0.50, 0.51), _cmp(0.50, 0.52)]),  # deltas .01/.02
    ]
    decomp = aggregate_control_rows(rows, seeds=[0, 1], task="gsm8k-test")

    assert decomp.n_seeds == 2 and decomp.family_size == 2
    assert "df=1" in decomp.ci_kind
    sym, _plus = decomp.rows
    assert sym.control == "gsm-symbolic"
    assert sym.base_acc == pytest.approx(0.50)  # seed-independent base
    assert sym.per_seed_delta == pytest.approx((0.10, 0.20))
    assert sym.mean_delta == pytest.approx(0.15)
    assert sym.ci_low < sym.mean_delta < sym.ci_high  # seed-level t interval (df=1)
    # Holm never shrinks a p-value, and significance keys off the corrected p.
    for row in decomp.rows:
        assert row.p_value_holm >= row.p_value
        assert row.significant == (row.p_value_holm < 0.05)


def test_aggregate_control_rows_validates() -> None:
    with pytest.raises(ValueError, match="no control rows"):
        aggregate_control_rows([], seeds=[0], task="gsm8k-test")
    with pytest.raises(ValueError, match="comparisons but"):
        # one comparison but two seed labels declared
        aggregate_control_rows(
            [("gsm-symbolic", "memorization", [_cmp(0.5, 0.6)])], seeds=[0, 1], task="gsm8k-test"
        )
