"""Unit tests for the paired statistics layer (uses eval-audit, no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.stats.bootstrap import bootstrap_mean_ci
from grpo_gain_decomp.stats.compare import compare
from grpo_gain_decomp.stats.significance import holm_correction, mcnemar


def _by_id(values: list[bool]) -> dict[str, bool]:
    """Key a correctness vector by synthetic problem ids (compare is id-paired)."""
    return {str(i): value for i, value in enumerate(values)}


def test_mcnemar_identical_models_are_not_significant() -> None:
    vec = [True, False, True, False, True]
    p, n_discordant, test = mcnemar(vec, vec)
    assert p == 1.0
    assert n_discordant == 0
    assert test == "exact-binomial"


def test_mcnemar_uses_exact_binomial_for_few_discordant() -> None:
    a = [False] * 10
    b = [True] * 10  # 10 discordant, all favoring B
    p, n_discordant, test = mcnemar(a, b)
    assert test == "exact-binomial"
    assert n_discordant == 10
    assert p < 0.01  # 2 * 0.5**10


def test_mcnemar_uses_chi2_for_many_discordant() -> None:
    # 5 pairs favor A, 25 favor B (30 discordant >= 25), plus 10 concordant.
    a = [True] * 5 + [False] * 25 + [True] * 10
    b = [False] * 5 + [True] * 25 + [True] * 10
    p, n_discordant, test = mcnemar(a, b)
    assert test == "chi2"
    assert n_discordant == 30
    assert 0.0 < p < 0.05


def test_mcnemar_validates_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        mcnemar([True, False], [True])
    with pytest.raises(ValueError, match="empty"):
        mcnemar([], [])


def test_compare_assembles_delta_ci_and_significance() -> None:
    base = _by_id([False] * 10 + [True] * 10)  # acc 0.5
    rl = _by_id([True] * 15 + [False] * 5)  # acc 0.75
    result = compare("base", base, "rl", rl)

    assert result.n == 20
    assert result.accuracy_a == pytest.approx(0.5)
    assert result.accuracy_b == pytest.approx(0.75)
    # delta is acc_b - acc_a, and it falls within its own bootstrap CI.
    assert result.delta == pytest.approx(result.accuracy_b - result.accuracy_a)
    assert result.ci_low <= result.delta <= result.ci_high
    assert result.test in {"exact-binomial", "chi2"}


def test_compare_is_deterministic() -> None:
    a = _by_id([False, True, False, True, True, False, True, False])
    b = _by_id([True, True, False, True, True, True, True, False])
    assert compare("a", a, "b", b).ci_low == compare("a", a, "b", b).ci_low


def test_compare_rejects_mismatched_problem_ids() -> None:
    with pytest.raises(ValueError, match="same problem ids"):
        compare("a", {"p0": True, "p1": False}, "b", {"p0": True, "p2": False})


def test_headline_reports_delta_ci_p_and_n() -> None:
    base = _by_id([False] * 10 + [True] * 10)
    rl = _by_id([True] * 15 + [False] * 5)
    headline = compare("base", base, "rl", rl).headline()
    assert headline.startswith("rl beats base by")
    assert "95% CI [" in headline
    assert "McNemar p=" in headline
    assert "n=20" in headline


def test_headline_handles_a_negative_delta() -> None:
    strong = _by_id([True] * 18 + [False] * 2)
    weak = _by_id([False] * 10 + [True] * 10)
    # B (weak) is worse than A (strong) -> "trails".
    assert compare("strong", strong, "weak", weak).headline().startswith("weak trails strong by")


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.1, 0.2, 0.9, 0.4, 0.5]
    mean, ci_low, ci_high = bootstrap_mean_ci(values)
    assert (mean, ci_low, ci_high) == bootstrap_mean_ci(values)  # fixed seed -> identical
    assert mean == pytest.approx(sum(values) / len(values))
    assert ci_low <= mean <= ci_high


def test_bootstrap_mean_ci_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([])


def test_holm_correction_hand_example_and_order() -> None:
    # Holm of [0.01, 0.04, 0.03]: sorted [0.01,0.03,0.04] x [3,2,1] = [0.03,0.06,0.04],
    # enforce monotone non-decreasing -> [0.03,0.06,0.06], mapped back to input order.
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx((0.03, 0.06, 0.06))
    # Order is preserved (smaller raw p -> its adjusted value, in place).
    assert holm_correction([0.04, 0.01]) == pytest.approx((0.04, 0.02))


def test_holm_correction_caps_and_single() -> None:
    assert holm_correction([0.03]) == pytest.approx((0.03,))  # family of one: unchanged
    assert holm_correction([0.5, 0.5, 0.5]) == pytest.approx((1.0, 1.0, 1.0))  # capped at 1
    # Holm is never smaller than Bonferroni's m*p for the smallest p, never larger than 1.
    adj = holm_correction([0.2, 0.001, 0.9])
    assert adj[1] == pytest.approx(0.003) and all(0.0 <= a <= 1.0 for a in adj)


def test_holm_correction_validates() -> None:
    with pytest.raises(ValueError, match="no p-values"):
        holm_correction([])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_correction([0.5, 1.2])
