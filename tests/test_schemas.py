"""Schema invariants: strict (no unknown fields), frozen, sequence-like sets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grpo_gain_decomp.schemas import DatasetRef, Problem, ProblemSet


def _ref() -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split="test", revision="abc123")


def test_problem_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Problem(id="x", question="q", gold_answer="1", surprise="nope")  # type: ignore[call-arg]


def test_problem_is_frozen() -> None:
    problem = Problem(id="x", question="q", gold_answer="1")
    with pytest.raises(ValidationError):
        problem.gold_answer = "2"  # type: ignore[misc]


def test_dataset_ref_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(name="openai/gsm8k", split="test", revision="abc123")  # type: ignore[call-arg]


def test_dataset_ref_allows_null_config() -> None:
    ref = DatasetRef(name="d", config=None, split="test", revision="abc123")
    assert ref.config is None


def test_problem_set_is_sequence_like() -> None:
    problems = tuple(Problem(id=str(i), question=f"q{i}", gold_answer=str(i)) for i in range(3))
    problem_set = ProblemSet(source=_ref(), problems=problems)

    assert len(problem_set) == 3
    assert problem_set[1].id == "1"
    assert [p.id for p in problem_set] == ["0", "1", "2"]
