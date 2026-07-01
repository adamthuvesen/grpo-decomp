"""Unit tests for seed-level placebo-comparison aggregation (CPU; uses the eval extra)."""

from __future__ import annotations

import pytest

from grpo_decomp.report.seeds import aggregate_placebo_comparison
from grpo_decomp.stats.compare import compare


def _placebo_comparison(n: int, k_correct: int):
    """A placebo comparison where random gets 0/n and correct gets k/n."""
    ids = [f"p{i}" for i in range(n)]
    random_correct = dict.fromkeys(ids, False)
    correct_correct = {pid: (idx < k_correct) for idx, pid in enumerate(ids)}
    return compare("random", random_correct, "correct", correct_correct)


def test_aggregate_three_seeds_uses_seed_level_t() -> None:
    comps = [
        _placebo_comparison(100, 10),
        _placebo_comparison(100, 20),
        _placebo_comparison(100, 30),
    ]
    result = aggregate_placebo_comparison(comps, [0, 1, 2], task="gsm8k-test")
    assert result.n_seeds == 3
    assert result.preliminary is False  # 3 >= MIN_SEEDS
    assert result.per_seed_delta == pytest.approx((0.10, 0.20, 0.30), abs=1e-9)
    assert result.mean_delta == pytest.approx(0.20, abs=1e-9)
    assert result.sem == pytest.approx(0.1 / 3**0.5, abs=1e-9)
    assert result.ci_low < 0.20 < result.ci_high  # interval brackets the mean
    assert "seed-level t" in result.ci_kind
    assert result.per_seed_random_acc == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_single_seed_is_preliminary_with_eval_ci() -> None:
    comps = [_placebo_comparison(100, 20)]
    result = aggregate_placebo_comparison(comps, [0], task="gsm8k-test")
    assert result.n_seeds == 1
    assert result.preliminary is True
    assert result.sem is None
    assert result.ci_low == comps[0].ci_low  # falls back to the eval-bootstrap CI
    assert result.ci_high == comps[0].ci_high
    assert "single-seed" in result.ci_kind


def test_mismatched_seed_count_is_explicit_error() -> None:
    comps = [_placebo_comparison(50, 10), _placebo_comparison(50, 20)]
    with pytest.raises(ValueError, match="comparisons but"):
        aggregate_placebo_comparison(comps, [0], task="gsm8k-test")


def test_empty_is_explicit_error() -> None:
    with pytest.raises(ValueError, match="no per-seed"):
        aggregate_placebo_comparison([], [], task="gsm8k-test")
