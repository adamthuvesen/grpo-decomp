"""Unit tests for the unbiased pass@k estimator (no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.eval.passk import estimate_pass_at_k, pass_at_k


def test_pass_at_1_equals_empirical_rate() -> None:
    # pass@1 reduces to c/n (telescoping product).
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(200, 100, 1) == pytest.approx(0.5)


def test_pass_at_k_boundaries() -> None:
    assert pass_at_k(5, 0, 1) == 0.0  # no correct -> never
    assert pass_at_k(5, 5, 1) == 1.0  # all correct -> always
    assert pass_at_k(5, 1, 5) == 1.0  # k == n and c >= 1 -> certain


def test_pass_at_k_is_unbiased_not_the_biased_plugin() -> None:
    # Biased plug-in 1-(1-p)^k with p=c/n=0.2, k=2 gives 0.36; unbiased differs.
    value = pass_at_k(10, 2, 2)
    assert value == pytest.approx(1.0 - (8 * 7) / (10 * 9))  # 1 - C(8,2)/C(10,2)
    assert value != pytest.approx(0.36)


def test_pass_at_k_monotonic_in_k() -> None:
    values = [pass_at_k(20, 4, k) for k in range(1, 21)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_pass_at_k_validates_arguments() -> None:
    with pytest.raises(ValueError, match="c <= n"):
        pass_at_k(5, 6, 1)
    with pytest.raises(ValueError, match="k <= n"):
        pass_at_k(5, 1, 6)


def test_estimate_averages_over_problems() -> None:
    # n=5 samples each; correct counts [0, 5, 3] -> pass@1 = mean(0, 1.0, 0.6).
    assert estimate_pass_at_k([0, 5, 3], k=1, n=5) == pytest.approx((0.0 + 1.0 + 0.6) / 3)


def test_estimate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        estimate_pass_at_k([], k=1, n=5)
