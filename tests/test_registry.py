"""Contract tests for the harness registries once the study has registered into them."""

from __future__ import annotations

from grpo_decomp.registries import EVAL_SETS, PROBES, REWARDS, TASKS


def test_task_profiles_and_controls_name_registered_sets() -> None:
    assert "gsm8k-test" in EVAL_SETS
    assert "countdown-test" in EVAL_SETS
    for slug in PROBES:
        assert slug in EVAL_SETS
    for profile in TASKS.values():
        assert {profile.task_set, *profile.control_sets} <= set(EVAL_SETS)


def test_study_registered_rewards_and_tasks() -> None:
    assert {"correct", "countdown", "random"} <= set(REWARDS)
    assert {"gsm8k", "countdown"} <= set(TASKS)


def test_explicit_unknown_verifier_fails_and_legacy_boxed_math_keeps_working() -> None:
    import pytest

    from grpo_decomp.grading import is_correct
    from grpo_decomp.registries import verifier_for, verifier_identity
    from grpo_decomp.schemas import DatasetRef
    from llm_grpo_gains.data.countdown import countdown_is_correct

    source = DatasetRef(name="countdown", config=None, split="test", revision="fixture")
    assert verifier_for(source) is countdown_is_correct
    assert verifier_identity(source).startswith("countdown:")
    assert verifier_for(source, name="boxed-math") is is_correct
    assert verifier_identity(source, name="boxed-math").endswith("grpo_decomp.grading.is_correct")
    with pytest.raises(ValueError, match="unknown verifier 'missing'"):
        verifier_for(source, name="missing")
    legacy = source.model_copy(update={"name": "legacy-math"})
    assert verifier_for(legacy) is is_correct


def test_defaults_are_study_owned_and_optional_without_a_study(monkeypatch) -> None:
    from grpo_decomp import registries

    assert registries.default_task_set() == "gsm8k-test"
    assert registries.default_train_dataset() == "gsm8k"
    monkeypatch.setattr(registries, "DEFAULTS", {})
    assert registries.default_task_set() is None
    assert registries.default_train_dataset() is None
