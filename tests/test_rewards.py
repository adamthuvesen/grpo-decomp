"""Unit tests for the reward functions (no network, no GPU)."""

from __future__ import annotations

import pytest

from llm_grpo_gains.rewards import (
    PLACEBO_REWARD,
    SELECTABLE,
    correct,
    countdown,
    get_reward,
    make_random_reward,
)


def test_correct_exact_match_scores_one() -> None:
    assert correct(["Let me think... the answer is 72"], ["72"]) == [1.0]


def test_correct_wrong_scores_zero() -> None:
    assert correct(["the total is 73"], ["72"]) == [0.0]


def test_correct_normalizes_commas_and_fractions() -> None:
    # math-verify handles thousands separators and fraction/decimal equivalence.
    assert correct(["she has 1,000 left"], ["1000"]) == [1.0]
    assert correct(["the result is 0.75"], ["3/4"]) == [1.0]


def test_correct_unparseable_scores_zero_in_training() -> None:
    assert correct(["no numeric answer at all"], ["5"]) == [0.0]


def test_correct_scores_a_batch_pointwise() -> None:
    rewards = correct(["it is 1", "it is 5", "it is 9"], ["1", "4", "9"])
    assert rewards == [1.0, 0.0, 1.0]


def test_correct_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValueError, match="zip"):
        correct(["a", "b"], ["1"])  # strict zip: a length mismatch is an explicit error


def test_placebo_values_in_unit_interval() -> None:
    values = make_random_reward(0)(["a", "b", "c", "d"])
    assert len(values) == 4
    assert all(0.0 <= v < 1.0 for v in values)


def test_placebo_ignores_correctness() -> None:
    # Same RNG seed, different completions and golds -> identical rewards (both are unused).
    with_gold_a = make_random_reward(0)(["x", "y"], gold_answer=["1", "2"])
    with_gold_b = make_random_reward(0)(["different", "text"], gold_answer=["999", "888"])
    assert with_gold_a == with_gold_b


def test_placebo_rng_is_stateful_within_a_run() -> None:
    reward = make_random_reward(0)
    first = reward(["a"])
    second = reward(["a"])
    assert first != second
    assert first == make_random_reward(0)(["a"])


def test_placebo_is_reproducible_per_seed() -> None:
    assert make_random_reward(0)(["a", "b", "c"]) == make_random_reward(0)(["a", "b", "c"])


def test_placebo_differs_by_seed() -> None:
    assert make_random_reward(0)(["a", "b", "c", "d"]) != make_random_reward(1)(
        ["a", "b", "c", "d"]
    )


def test_countdown_valid_boxed_expression_scores_one() -> None:
    key = "target=26;numbers=4,5,6,7"
    assert countdown([r"first I reason, then \boxed{4 * 5 + 6}"], [key]) == [1.0]


def test_countdown_wrong_expression_scores_zero() -> None:
    assert countdown([r"\boxed{4 * 5 + 7}"], ["target=26;numbers=4,5,6,7"]) == [0.0]


def test_countdown_unboxed_completion_scores_zero_in_training() -> None:
    # No box -> no expression to verify -> 0.0 (downward pressure under beta=0.0).
    assert countdown(["the answer is 4 * 5 + 6"], ["target=26;numbers=4,5,6,7"]) == [0.0]


def test_countdown_reusing_a_number_scores_zero() -> None:
    assert countdown([r"\boxed{5 * 5}"], ["target=25;numbers=5,6"]) == [0.0]


def test_get_reward_selects_by_name() -> None:
    assert get_reward("correct") is correct
    assert get_reward("countdown") is countdown
    assert "countdown" in SELECTABLE
    assert PLACEBO_REWARD == "random"
    assert PLACEBO_REWARD in SELECTABLE
    # 'random' yields a fresh seeded fn equivalent to constructing it directly.
    assert get_reward(PLACEBO_REWARD, seed=0)(["a"]) == make_random_reward(0)(["a"])


def test_get_reward_rejects_format_and_unknown() -> None:
    assert "format" not in SELECTABLE
    with pytest.raises(ValueError, match="format"):
        get_reward("format")
    with pytest.raises(ValueError, match="unknown reward"):
        get_reward("bogus")
