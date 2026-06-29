"""Unit tests for the reward functions (no network, no GPU)."""

from __future__ import annotations

import pytest

from grpo_decomp.eval.answers import extract_strict, is_correct
from grpo_decomp.registries import REWARDS
from grpo_decomp.rewards import PLACEBO_REWARD, get_reward, make_random_reward
from llm_grpo_gains.rewards import correct, countdown


def test_correct_boxed_exact_match_scores_one() -> None:
    assert correct([r"Let me think... \boxed{72}"], ["72"]) == [1.0]


def test_correct_unboxed_answer_scores_zero() -> None:
    assert correct(["Let me think... the answer is 72"], ["72"]) == [0.0]


def test_correct_wrong_boxed_scores_zero() -> None:
    assert correct([r"\boxed{73}"], ["72"]) == [0.0]


def test_correct_normalizes_commas_and_fractions() -> None:
    assert correct([r"\boxed{1,000}"], ["1000"]) == [1.0]
    assert correct([r"\boxed{0.75}"], ["3/4"]) == [1.0]


def test_correct_unparseable_scores_zero_in_training() -> None:
    assert correct(["no numeric answer at all"], ["5"]) == [0.0]


def test_correct_scores_a_batch_pointwise() -> None:
    rewards = correct([r"\boxed{1}", r"\boxed{5}", r"\boxed{9}"], ["1", "4", "9"])
    assert rewards == [1.0, 0.0, 1.0]


def test_correct_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValueError, match="zip"):
        correct(["a", "b"], ["1"])


def test_correct_matches_strict_eval_grading() -> None:
    """Training reward and headline strict accuracy use the same extraction path."""
    pairs = [
        (r"reasoning \boxed{42}", "42"),
        (r"\boxed{73}", "72"),
        ("plain 42", "42"),
        (r"\boxed{\frac{3}{4}}", "3/4"),
    ]
    for completion, gold in pairs:
        reward = correct([completion], [gold])[0]
        strict = is_correct(extract_strict(completion), gold)
        assert reward == (1.0 if strict else 0.0)


def test_placebo_values_in_unit_interval() -> None:
    values = make_random_reward(0)(["a", "b", "c", "d"])
    assert len(values) == 4
    assert all(0.0 <= v < 1.0 for v in values)


def test_placebo_ignores_correctness() -> None:
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
    assert countdown(["the answer is 4 * 5 + 6"], ["target=26;numbers=4,5,6,7"]) == [0.0]


def test_countdown_reusing_a_number_scores_zero() -> None:
    assert countdown([r"\boxed{5 * 5}"], ["target=25;numbers=5,6"]) == [0.0]


def test_get_reward_selects_by_name() -> None:
    assert get_reward("correct") is correct
    assert get_reward("countdown") is countdown
    assert "countdown" in REWARDS
    assert PLACEBO_REWARD == "random"
    assert PLACEBO_REWARD in REWARDS
    assert get_reward(PLACEBO_REWARD, seed=0)(["a"]) == make_random_reward(0)(["a"])


def test_get_reward_rejects_format_and_unknown() -> None:
    assert "format" not in REWARDS
    with pytest.raises(ValueError, match="format"):
        get_reward("format")
    with pytest.raises(ValueError, match="unknown reward"):
        get_reward("bogus")
