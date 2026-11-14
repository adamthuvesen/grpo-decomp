"""Per-arm GRPO training configuration (one YAML file per arm).

An *arm* is a single GRPO run: base model + reward + dataset + seed + GRPO
hyperparameters. Arms differ only by config — switching ``correct`` to ``random``
is a one-field change — so the study has no per-arm code. The GRPO defaults mirror
the verified TRL v1.0.0 values (re-confirm on any version bump); ``use_vllm`` must
be True or colocated vLLM is inert, and the field names mirror ``trl.GRPOConfig``
parameters so the settings map straight onto it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import ConfigDict, Field, model_validator

from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, default_train_dataset
from grpo_decomp.schemas import Record


class GRPOSettings(Record):
    """GRPO hyperparameters; field names mirror ``trl.GRPOConfig`` parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    loss_type: Literal["grpo", "dapo", "bnpo", "dr_grpo", "cispo", "sapo", "luspo", "vespo"] = (
        "dapo"
    )
    scale_rewards: Literal["group", "batch", "none"] = "group"
    beta: float = Field(default=0.0, ge=0)  # no KL / reference model (R1-Zero)
    learning_rate: float = Field(default=1e-6, gt=0)
    # With batch == num_generations == 8, no grad accumulation, and the default
    # steps_per_generation, each optimizer step sees one unique prompt's group.
    per_device_train_batch_size: int = Field(default=8, gt=0)
    num_generations: int = Field(default=8, ge=2)
    max_completion_length: int = Field(
        default=1024, gt=0
    )  # CoT headroom; 512 clipped too many rollouts.
    max_steps: int = Field(default=500, gt=0)
    # Periodic checkpoints and frequent logs are retained so held-out accuracy,
    # entropy, and completion length can be inspected across the run.
    logging_steps: int = Field(default=10, gt=0)
    save_strategy: Literal["no", "steps", "epoch", "best"] = "steps"
    save_steps: int = Field(default=100, gt=0)  # ~5 checkpoints over 500 steps.
    use_vllm: bool = True  # MUST be True or vllm_mode is inert
    vllm_mode: Literal["colocate", "server"] = "colocate"
    vllm_gpu_memory_utilization: float = Field(default=0.3, gt=0, le=1)
    mask_truncated_completions: bool = True  # guards length-hacking under beta=0.0
    # None keeps TRL's default (= gradient_accumulation_steps = 1): one optimizer step per
    # rollout batch, so the policy ratio stays 1 and the PPO clip never binds. With N > 1,
    # TRL generates N batches at once, shuffles them, and spends one slice per step: later
    # steps are off-policy, and each step mixes completions from N prompts.
    steps_per_generation: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_generation_batch(self) -> Self:
        # TRL 1.0.0 GRPOConfig.__post_init__, single GPU and gradient accumulation 1.
        batch = self.per_device_train_batch_size * (self.steps_per_generation or 1)
        if batch % self.num_generations:
            raise ValueError("generation batch size must be divisible by num_generations")
        if self.loss_type == "vespo" and self.use_vllm:
            raise ValueError("vespo with vLLM requires token-level correction settings")
        if self.save_strategy == "best":
            raise ValueError("save_strategy=best requires evaluation, which runs post-training")
        return self


def _default_dataset() -> str:
    dataset = default_train_dataset()
    if dataset is None:
        raise ValueError("dataset is required when no study registers a default training dataset")
    return dataset


class ArmConfig(Record):
    """One training arm: base model + reward + dataset + prompt strategy + seed + GRPO config.

    `reward`, `dataset`, and `prompt_strategy` are registry keys (resolved at launch time
    from :mod:`grpo_decomp.registries`), not a fixed enum — a new task plugs in its own
    without editing the harness. An unknown key is an explicit error when the run starts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    base_model: str = Field(min_length=1)
    reward: str = Field(min_length=1)
    seed: int = Field(ge=0)
    # The training task: a key into the train-dataset registry (e.g. `gsm8k`, `countdown`).
    dataset: str = Field(default_factory=_default_dataset)
    # The prompt strategy: a key into the prompt-strategy registry (defaults to the harness
    # `r1_zero`). Training and eval MUST use the same strategy.
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY
    base_model_revision: str | None = None
    grpo: GRPOSettings = Field(default_factory=GRPOSettings)


def load_arm_config(path: Path) -> ArmConfig:
    """Load and validate an arm config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ArmConfig.model_validate(data)
