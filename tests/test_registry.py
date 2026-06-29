"""Contract tests for the eval registry."""

from __future__ import annotations

from llm_grpo_gains.eval.registry import ARMS, CONTROL_SETS, PROBES, SETS


def test_sets_include_task_and_controls() -> None:
    assert "gsm8k-test" in SETS
    assert "countdown-test" in SETS
    for slug in CONTROL_SETS:
        assert slug in SETS
        assert slug in PROBES


def test_arms_are_the_decomposition_triplet() -> None:
    assert ARMS == ("base", "correct", "random")
