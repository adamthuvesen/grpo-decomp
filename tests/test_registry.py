"""Contract tests for the harness registries once the study has registered into them."""

from __future__ import annotations

from grpo_decomp.registries import ARMS, CONTROL_SETS, EVAL_SETS, PROBES, REWARDS, TASKS


def test_sets_include_task_and_controls() -> None:
    assert "gsm8k-test" in EVAL_SETS
    assert "countdown-test" in EVAL_SETS
    for slug in CONTROL_SETS:
        assert slug in EVAL_SETS
        assert slug in PROBES


def test_study_registered_rewards_and_tasks() -> None:
    assert {"correct", "countdown", "random"} <= set(REWARDS)
    assert {"gsm8k", "countdown"} <= set(TASKS)


def test_arms_are_the_decomposition_triplet() -> None:
    assert ARMS == ("base", "correct", "random")
