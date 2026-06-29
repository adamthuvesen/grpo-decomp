"""Unit tests for per-arm training configs (no network, no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_grpo_gains.train.config import ArmConfig, GRPOSettings, load_arm_config

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


def test_as_grpo_kwargs_exposes_trl_parameter_names() -> None:
    kwargs = GRPOSettings().as_grpo_kwargs()
    assert kwargs["loss_type"] == "dapo"
    assert kwargs["beta"] == 0.0
    assert kwargs["use_vllm"] is True


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


def test_arm_rejects_non_selectable_reward() -> None:
    # `format` is intentionally disabled and must not be selectable.
    with pytest.raises(ValidationError):
        ArmConfig(name="fmt", base_model="m", reward="format", seed=0)


def test_arm_rejects_unknown_dataset() -> None:
    with pytest.raises(ValidationError):
        ArmConfig(name="x", base_model="m", reward="correct", seed=0, dataset="mnist")


def test_arm_rejects_non_train_split() -> None:
    with pytest.raises(ValidationError):
        ArmConfig(name="x", base_model="m", reward="correct", seed=0, train_split="test")


def test_arm_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ArmConfig(name="x", base_model="m", reward="correct", seed=0, surprise=1)
