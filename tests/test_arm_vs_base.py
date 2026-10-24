"""Unit tests for the one-arm-vs-base report (paired per problem across seeds)."""

from __future__ import annotations

import pytest

from grpo_decomp.report.arm_vs_base import aggregate_arm_vs_base

_BASE = {"p0": True, "p1": False, "p2": False, "p3": True}


def test_arm_vs_base_pairs_seeds_to_base_and_keeps_per_problem_grades() -> None:
    seeds = [
        (0, {"p0": True, "p1": True, "p2": False, "p3": True}),
        (1, {"p0": True, "p1": True, "p2": True, "p3": True}),
        (2, {"p0": True, "p1": False, "p2": False, "p3": True}),
    ]
    result = aggregate_arm_vs_base(_BASE, seeds, task="gsm8k-test", arm="random")
    assert result.base_acc == pytest.approx(0.5)
    assert result.per_seed_delta == pytest.approx((0.25, 0.5, 0.0))
    assert result.mean_delta == pytest.approx(0.25)
    assert result.ci_low <= result.mean_delta <= result.ci_high
    assert result.base_correct == (True, False, False, True)
    assert result.per_seed_arm_correct[1] == (True, True, True, True)
    assert result.preliminary is False
    assert "random - base = +25.0 pp" in result.headline()


def test_arm_vs_base_rejects_a_seed_on_a_different_problem_set() -> None:
    with pytest.raises(ValueError, match="seed 0"):
        aggregate_arm_vs_base(_BASE, [(0, {"p0": True})], task="gsm8k-test", arm="random")
