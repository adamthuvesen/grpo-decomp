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


def test_registration_rejects_other_owners_and_requires_explicit_idempotence() -> None:
    import pytest

    from grpo_decomp.registries import (
        register_eval_set,
        registration_owner,
    )
    from llm_grpo_gains.registration import register

    original = EVAL_SETS["gsm8k-test"]
    register()
    assert EVAL_SETS["gsm8k-test"] is original
    with (
        registration_owner("another-study", idempotent=True),
        pytest.raises(ValueError, match=r"already belongs to 'llm_grpo_gains\.registration'"),
    ):
        register_eval_set("gsm8k-test", original)
    with (
        registration_owner("llm_grpo_gains.registration"),
        pytest.raises(ValueError, match="already belongs"),
    ):
        register_eval_set("gsm8k-test", original)


def test_placebo_cannot_be_replaced_even_before_its_first_registration(monkeypatch) -> None:
    import pytest

    from grpo_decomp import registries

    monkeypatch.setattr(registries, "REWARDS", {})
    with (
        registries.registration_owner("study", idempotent=True),
        pytest.raises(ValueError, match=r"reserved for grpo_decomp\.rewards"),
    ):
        registries.register_reward("random", lambda seed: lambda **kwargs: [])
    assert registries.REWARDS == {}


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
    monkeypatch.setattr(registries, "_DEFAULTS", {})
    assert registries.default_task_set() is None
    assert registries.default_train_dataset() is None
