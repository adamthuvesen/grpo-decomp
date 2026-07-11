"""Contract tests for the harness registries once the study has registered into them."""

from __future__ import annotations

from grpo_decomp.registries import EVAL_SETS, PROBES, REWARDS, TASKS


def test_sets_include_task_and_controls() -> None:
    assert "gsm8k-test" in EVAL_SETS
    assert "countdown-test" in EVAL_SETS
    for slug in PROBES:
        assert slug in EVAL_SETS


def test_study_registered_rewards_and_tasks() -> None:
    assert {"correct", "countdown", "random"} <= set(REWARDS)
    assert {"gsm8k", "countdown"} <= set(TASKS)
