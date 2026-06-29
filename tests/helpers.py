"""Shared helpers for constructing CompletionSet fixtures in unit tests."""

from __future__ import annotations

from pathlib import Path

from llm_grpo_gains.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    write_completion_set,
)
from llm_grpo_gains.schemas import DatasetRef, Problem, ProblemSet


def dataset_ref(*, split: str = "test", revision: str = "rev") -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split=split, revision=revision)


def problem_set(
    *, ids: tuple[str, ...] = ("p1", "p2", "p3"), ref: DatasetRef | None = None
) -> ProblemSet:
    return ProblemSet(
        source=ref or dataset_ref(),
        problems=tuple(Problem(id=pid, question="q", gold_answer="4") for pid in ids),
    )


def write_completion_set_dir(
    path: Path,
    *,
    model: str,
    boxed: str,
    ids: tuple[str, ...] = ("p1", "p2", "p3"),
    n: int = 1,
    temperature: float = 0.0,
    ref: DatasetRef | None = None,
) -> None:
    items = tuple(
        ProblemCompletions(
            problem=Problem(id=pid, question="q", gold_answer="4"),
            samples=tuple(f"reasoning \\boxed{{{boxed}}}" for _ in range(n)),
        )
        for pid in ids
    )
    provenance = GenerationProvenance(
        model=model,
        model_revision=None,
        backend="transformers",
        sampling=SamplingConfig(temperature=temperature, n=n),
        dataset=ref or dataset_ref(),
        n_problems=len(ids),
        commit="c" * 40,
        python_version="3.11.0",
        package_versions={},
    )
    write_completion_set(CompletionSet(provenance=provenance, items=items), path)
