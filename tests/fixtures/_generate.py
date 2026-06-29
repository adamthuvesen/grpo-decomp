"""Generate the tiny frozen pass@k fixture (run once; the output under `mini/` is committed).

    uv run python tests/fixtures/_generate.py

A 12-problem, n=4 base + two correct seeds (~30 KB) so the multi-seed pass@k aggregator and the
mechanism report are testable end-to-end *from disk* — proving the load -> aggregate -> schema
path on committed data, without pulling ~14 MB CompletionSets from the Modal volume. Completion
text is fixed per (problem, sample); no randomness, so the fixture is byte-stable.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    write_completion_set,
)
from grpo_decomp.schemas import DatasetRef, Problem

_REF = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="frozen-fixture")
_FIX = Path(__file__).parent / "mini"
_N = 4
_N_PROBLEMS = 12


def _chain(ans: int) -> str:
    return f"<<{ans - 1}+1={ans}>> so \\boxed{{{ans}}}"  # correct AND a valid calculator chain


def _boxed(ans: int) -> str:
    return f"\\boxed{{{ans}}}"  # correct, no chain (CoT-gating drops it)


def _wrong(ans: int) -> str:
    return f"\\boxed{{{ans + 999}}}"  # wrong answer


def _write(name: str, samples_for: Callable[[int], list[str]]) -> None:
    problems = tuple(
        Problem(id=f"q{i:02d}", question="q", gold_answer=str(i)) for i in range(_N_PROBLEMS)
    )
    provenance = GenerationProvenance(
        model=name,
        model_revision=None,
        backend="vllm",
        sampling=SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=64, n=_N, seed=0),
        dataset=_REF,
        n_problems=_N_PROBLEMS,
        commit="frozen",
        dirty=False,
        python_version="3.11",
        package_versions={},
    )
    items = tuple(
        ProblemCompletions(problem=p, samples=tuple(samples_for(i))) for i, p in enumerate(problems)
    )
    write_completion_set(CompletionSet(provenance=provenance, items=items), _FIX / name)


def _base(i: int) -> list[str]:
    if i < 4:  # base solves first-try reliably (3/4), one with a chain
        return [_chain(i), _boxed(i), _boxed(i), _wrong(i)]
    if i < 8:  # within reach but unreliable (1/4): migration candidates once RL lands
        return [_boxed(i), _wrong(i), _wrong(i), _wrong(i)]
    return [_wrong(i)] * 4  # outside the base's reach


def _correct(seed: int) -> Callable[[int], list[str]]:
    def samples(i: int) -> list[str]:
        if i < 8:  # RL makes the first 8 reliable (the migration)
            return [_chain(i), _boxed(i), _boxed(i), _wrong(i)]
        if i < 10 and seed == 1:  # seed 1 reaches two the base could not (a little expansion)
            return [_boxed(i), _boxed(i), _wrong(i), _wrong(i)]
        return [_wrong(i)] * 4

    return samples


if __name__ == "__main__":
    _write("base__mini", _base)
    _write("correct-seed0__mini", _correct(0))
    _write("correct-seed1__mini", _correct(1))
    print(f"wrote frozen fixture to {_FIX}")
