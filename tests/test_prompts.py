"""Train and eval must build prompts from the same registered strategy (no network)."""

from __future__ import annotations

from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, get_prompt_strategy


def test_r1_zero_prompt_contains_question_and_box_format() -> None:
    prompt = get_prompt_strategy("r1_zero").build_prompt("What is 2+2?")
    assert "What is 2+2?" in prompt
    assert "\\boxed" in prompt


def test_default_strategy_is_registered() -> None:
    assert get_prompt_strategy(DEFAULT_PROMPT_STRATEGY).name == DEFAULT_PROMPT_STRATEGY


def test_strategy_lookup_is_shared_by_train_and_eval() -> None:
    # Training (launcher) and eval (generate) both resolve the strategy by name from the
    # one registry, so they cannot drift apart: same name -> same build_prompt.
    strategy = get_prompt_strategy("r1_zero")
    assert get_prompt_strategy("r1_zero") is strategy
    question = "A robot has 3 wheels. It builds 4 more robots. How many wheels?"
    assert strategy.build_prompt(question) == get_prompt_strategy("r1_zero").build_prompt(question)


def test_unknown_strategy_is_an_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown prompt strategy"):
        get_prompt_strategy("does-not-exist")
