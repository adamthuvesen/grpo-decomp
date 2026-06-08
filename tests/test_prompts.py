"""Train and eval must build prompts from the one shared builder (no network)."""

from __future__ import annotations

from grpo_gain_decomp.prompts import build_prompt
from grpo_gain_decomp.train.launcher import build_prompt as train_build_prompt


def test_prompt_contains_question_and_box_format() -> None:
    prompt = build_prompt("What is 2+2?")
    assert "What is 2+2?" in prompt
    assert "\\boxed" in prompt


def test_train_and_eval_share_one_prompt_builder() -> None:
    # The launcher re-exports the shared builder, so eval generation (which imports
    # grpo_gain_decomp.prompts) and training cannot drift apart.
    assert train_build_prompt is build_prompt


def test_prompt_is_byte_identical_across_callers() -> None:
    question = "A robot has 3 wheels. It builds 4 more robots. How many wheels?"
    assert build_prompt(question) == train_build_prompt(question)
