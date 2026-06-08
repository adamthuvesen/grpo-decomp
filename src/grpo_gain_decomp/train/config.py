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
from typing import Literal

import yaml
from pydantic import Field

from grpo_gain_decomp.schemas import Record


class GRPOSettings(Record):
    """GRPO hyperparameters; field names mirror ``trl.GRPOConfig`` parameters."""

    loss_type: str = "dapo"
    scale_rewards: str = "group"
    beta: float = 0.0  # no KL / reference model (R1-Zero)
    learning_rate: float = 1e-6
    # Day-1 tunable: with batch == num_generations == 8 and no grad accumulation,
    # each optimizer step sees one unique prompt; raise the effective batch (or add
    # gradient accumulation) so a step spans several prompts (lower-variance advantage).
    per_device_train_batch_size: int = 8
    num_generations: int = 8
    max_completion_length: int = 1024  # CoT headroom; the 5-step dry run clipped 0-62% at 512
    max_steps: int = 500  # placeholder; anchor on the day-1 run
    # Periodic checkpoints + frequent logs so the held-out accuracy curve
    # (`grpo-decomp heldout`), entropy, and completion-length have resolution across
    # the run — the curves the GSM8K-vs-MATH decision and checkpoint selection read.
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100  # ~5 checkpoints over 500 steps; tune with max_steps on day 1
    use_vllm: bool = True  # MUST be True or vllm_mode is inert
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    mask_truncated_completions: bool = True  # guards length-hacking under beta=0.0

    def as_grpo_kwargs(self) -> dict[str, object]:
        """Kwargs for ``trl.GRPOConfig`` (field names mirror its parameters)."""
        return self.model_dump()


class ArmConfig(Record):
    """One training arm. `dataset` and `reward` are constrained to the selectable sets."""

    name: str
    base_model: str
    reward: Literal["correct", "random", "countdown"]
    seed: int
    # The training task. `gsm8k` (default) carves a validation split from train; `countdown`
    # uses its own generated, seed-independent validation split. Default keeps existing
    # GSM8K arm configs unchanged.
    dataset: Literal["gsm8k", "countdown"] = "gsm8k"
    base_model_revision: str | None = None
    train_split: Literal["train"] = "train"
    # Pre-registered (anti-peeking) rule for which checkpoint feeds the decomposition.
    checkpoint_selection: Literal["final", "best_on_validation"] = "final"
    grpo: GRPOSettings = Field(default_factory=GRPOSettings)


def load_arm_config(path: Path) -> ArmConfig:
    """Load and validate an arm config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ArmConfig.model_validate(data)
