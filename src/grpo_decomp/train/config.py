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

import yaml
from pydantic import ConfigDict, Field

from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, default_train_dataset
from grpo_decomp.schemas import GRPOSettings, Record


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
