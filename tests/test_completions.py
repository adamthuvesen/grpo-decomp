"""Unit tests for the completion artifact: schema, validation, IO (no model, no network)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    load_completion_set,
    write_completion_set,
)
from grpo_decomp.schemas import DatasetRef, Problem


def _ref() -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")


def _provenance(*, n: int, n_problems: int) -> GenerationProvenance:
    return GenerationProvenance(
        model="Qwen/Qwen2.5-Math-1.5B",
        model_revision=None,
        backend="transformers",
        sampling=SamplingConfig(temperature=0.0, n=n),
        dataset=_ref(),
        n_problems=n_problems,
        commit="c" * 40,
        python_version="3.11.0",
        package_versions={"grpo-decomp": "0.1.0"},
    )


def _completion_set(*, n: int = 2, ids: tuple[str, ...] = ("a", "b")) -> CompletionSet:
    items = tuple(
        ProblemCompletions(
            problem=Problem(id=pid, question=f"q{pid}", gold_answer="4"),
            samples=tuple(f"{pid}-s{j}" for j in range(n)),
        )
        for pid in ids
    )
    return CompletionSet(provenance=_provenance(n=n, n_problems=len(ids)), items=items)


def test_round_trips_through_disk(tmp_path) -> None:
    original = _completion_set()
    write_completion_set(original, tmp_path / "cs")
    assert load_completion_set(tmp_path / "cs") == original


def test_round_trip_preserves_problem_order_not_lexicographic_id_order(tmp_path) -> None:
    original = _completion_set(ids=("p1", "p2", "p10"))
    write_completion_set(original, tmp_path / "cs")
    loaded = load_completion_set(tmp_path / "cs")
    assert tuple(item.problem.id for item in loaded.items) == ("p1", "p2", "p10")


def test_write_is_byte_identical_for_equal_inputs(tmp_path) -> None:
    write_completion_set(_completion_set(), tmp_path / "one")
    write_completion_set(_completion_set(), tmp_path / "two")
    for name in ("provenance.json", "completions.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_non_uniform_sample_counts_are_loud() -> None:
    items = (
        ProblemCompletions(
            problem=Problem(id="a", question="q", gold_answer="4"), samples=("s0", "s1")
        ),
        ProblemCompletions(problem=Problem(id="b", question="q", gold_answer="4"), samples=("s0",)),
    )
    with pytest.raises(ValidationError, match="non-uniform"):
        CompletionSet(provenance=_provenance(n=2, n_problems=2), items=items)


def test_n_problems_mismatch_is_loud() -> None:
    with pytest.raises(ValidationError, match="n_problems"):
        CompletionSet(
            provenance=_provenance(n=1, n_problems=5),
            items=(
                ProblemCompletions(
                    problem=Problem(id="a", question="q", gold_answer="4"), samples=("s0",)
                ),
            ),
        )


def test_empty_completion_set_is_loud() -> None:
    with pytest.raises(ValidationError, match="empty"):
        CompletionSet(provenance=_provenance(n=1, n_problems=0), items=())


def test_duplicate_ids_are_loud() -> None:
    item = ProblemCompletions(
        problem=Problem(id="dup", question="q", gold_answer="4"), samples=("s0",)
    )
    with pytest.raises(ValidationError, match="duplicate"):
        CompletionSet(provenance=_provenance(n=1, n_problems=2), items=(item, item))


def test_problem_set_and_completions_by_id_round_trip() -> None:
    completion_set = _completion_set(n=2, ids=("x", "y"))
    assert {p.id for p in completion_set.problem_set()} == {"x", "y"}
    by_id = completion_set.completions_by_id()
    assert by_id["x"] == ("x-s0", "x-s1")


@pytest.mark.parametrize(
    "kwargs", [{"n": 0}, {"temperature": -0.1}, {"top_p": 1.5}, {"top_p": 0.0}]
)
def test_sampling_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(**kwargs)
