"""Canonical, strict data schemas shared across grpo_decomp.

Every dataset grpo_decomp touches — GSM8K and its perturbation/clean-label controls —
is reduced to the same `Problem` shape and served as a `ProblemSet` that carries
its own pinned `DatasetRef`. One eval path then runs over every set without
special-casing the source.

The records are frozen and reject unknown fields: a result artifact should be
immutable once built and should fail with a clear error if the upstream schema drifts, rather
than silently absorbing an unexpected column.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Literal, Self, overload

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    """Base for all grpo_decomp records: immutable, and unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def record_json(record: Record) -> str:
    """Deterministic pretty JSON for committed result/provenance artifacts."""
    return json.dumps(record.model_dump(), sort_keys=True, indent=2, allow_nan=False) + "\n"


class DatasetRef(Record):
    """A reproducible pointer to exactly one dataset split.

    `revision` is an immutable commit SHA (never a branch name) so a result can
    be traced back to the precise data snapshot it was produced from.
    """

    name: str = Field(description="HuggingFace repo id, e.g. 'openai/gsm8k'.")
    config: str | None = Field(description="HF config/subset, e.g. 'main'; None if absent.")
    split: str = Field(description="Split name, e.g. 'train' or 'test'.")
    revision: str = Field(description="Immutable dataset commit SHA.")


class Problem(Record):
    """One math problem in grpo_decomp's canonical schema, identical across all datasets.

    `gold_answer` is the normalized final answer in string form. String — not
    float — because the family spans integers, decimals, and fractions, and
    exact-match comparison must not be at the mercy of float representation.
    """

    id: str = Field(description="Stable id, synthesized when absent, e.g. 'gsm8k/main/test/42'.")
    question: str
    gold_answer: str


class ProblemSet(Record):
    """An ordered, immutable collection of `Problem`s plus the `DatasetRef` they
    were loaded from. Iterable and sized so callers can treat it like a sequence.
    """

    source: DatasetRef
    problems: tuple[Problem, ...]

    @model_validator(mode="after")
    def _check_unique_ids(self) -> ProblemSet:
        ids = [problem.id for problem in self.problems]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate problem ids in problem set")
        return self

    def __len__(self) -> int:
        return len(self.problems)

    # Yields Problems, not Pydantic field pairs. This shadows
    # BaseModel.__iter__, so use .model_dump() (not dict(...)) for the field map.
    def __iter__(self) -> Iterator[Problem]:  # pyright: ignore[reportIncompatibleMethodOverride]  # deliberately yields Problems, not field pairs
        return iter(self.problems)

    @overload
    def __getitem__(self, index: int) -> Problem: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Problem, ...]: ...

    def __getitem__(self, index: int | slice) -> Problem | tuple[Problem, ...]:
        return self.problems[index]


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
