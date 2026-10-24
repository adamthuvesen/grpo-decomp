"""Unit tests for per-arm training configs (no network, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from grpo_decomp.train.config import ArmConfig, GRPOSettings, load_arm_config

_CONFIGS = Path(__file__).parents[1] / "configs"


def test_grpo_defaults_match_verified_trl_v1() -> None:
    grpo = GRPOSettings()
    assert grpo.loss_type == "dapo"
    assert grpo.scale_rewards == "group"
    assert grpo.beta == 0.0
    assert grpo.use_vllm is True  # colocate is inert otherwise
    assert grpo.vllm_mode == "colocate"
    assert grpo.mask_truncated_completions is True
    # Periodic checkpoints + logs feed the held-out curve and entropy/length curves.
    assert grpo.save_strategy == "steps" and grpo.save_steps > 0
    assert grpo.logging_steps > 0


@pytest.mark.parametrize(
    ("file", "reward"), [("correct.yaml", "correct"), ("random.yaml", "random")]
)
def test_load_arm_configs(file: str, reward: str) -> None:
    arm = load_arm_config(_CONFIGS / file)
    assert arm.reward == reward
    assert arm.base_model == "Qwen/Qwen2.5-Math-1.5B"
    assert arm.grpo.max_completion_length == 1024


def test_dataset_defaults_to_gsm8k() -> None:
    # Existing GSM8K configs omit `dataset` and must keep loading GSM8K.
    arm = ArmConfig(name="x", base_model="m", reward="correct", seed=0)
    assert arm.dataset == "gsm8k"


@pytest.mark.parametrize(
    ("file", "reward"),
    [("countdown-correct.yaml", "countdown"), ("countdown-random.yaml", "random")],
)
def test_load_countdown_arm_configs(file: str, reward: str) -> None:
    arm = load_arm_config(_CONFIGS / file)
    assert arm.dataset == "countdown"
    assert arm.reward == reward
    assert arm.base_model == "Qwen/Qwen2.5-1.5B"
    assert arm.grpo.max_completion_length == 1024


def test_reward_and_dataset_are_free_registry_keys() -> None:
    # `reward`/`dataset` are registry keys, not a fixed enum: ArmConfig accepts any string
    # (an unknown key is caught at launch time, by get_reward / the train-dataset registry).
    arm = ArmConfig(name="x", base_model="m", reward="my_reward", dataset="my_task", seed=0)
    assert arm.reward == "my_reward"
    assert arm.dataset == "my_task"


def test_prompt_strategy_defaults_to_r1_zero() -> None:
    arm = ArmConfig(name="x", base_model="m", reward="correct", seed=0)
    assert arm.prompt_strategy == "r1_zero"


def test_arm_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ArmConfig(name="x", base_model="m", reward="correct", seed=0, surprise=1)


def test_on_policy_arms_keep_trl_default_steps_per_generation() -> None:
    assert load_arm_config(_CONFIGS / "random.yaml").grpo.steps_per_generation is None


@pytest.mark.parametrize(
    ("new", "reference", "changed"),
    [
        ("offpolicy-random.yaml", "random.yaml", {"name", "grpo.steps_per_generation"}),
        (
            "countdown-math-correct.yaml",
            "countdown-correct.yaml",
            {"name", "base_model", "base_model_revision"},
        ),
    ],
)
def test_new_arms_change_one_variable_from_their_reference(
    new: str, reference: str, changed: set[str]
) -> None:
    def flat(file: str) -> dict[str, object]:
        dumped = load_arm_config(_CONFIGS / file).model_dump()
        grpo = dumped.pop("grpo")
        return dumped | {f"grpo.{key}": value for key, value in grpo.items()}

    a, b = flat(new), flat(reference)
    assert {key for key in a if a[key] != b[key]} == changed
