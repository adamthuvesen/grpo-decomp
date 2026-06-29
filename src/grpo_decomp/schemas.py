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

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    """Base for all grpo_decomp records: immutable, and unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


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

    def __len__(self) -> int:
        return len(self.problems)

    # Yields Problems, not Pydantic field pairs: this deliberately shadows
    # BaseModel.__iter__, so use .model_dump() (not dict(...)) for the field map.
    def __iter__(self) -> Iterator[Problem]:  # type: ignore[override]
        return iter(self.problems)

    def __getitem__(self, index: int | slice) -> Problem | tuple[Problem, ...]:
        return self.problems[index]
