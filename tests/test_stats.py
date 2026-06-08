"""Unit tests for the paired statistics layer (uses eval-audit, no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.stats.compare import compare
from grpo_gain_decomp.stats.significance import mcnemar


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
