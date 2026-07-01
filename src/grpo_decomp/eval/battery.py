"""Run the full eval battery over a checkpoint's completions and report results.

Operates on already-generated completions (problem id -> its `n` samples); the
generation backend (vLLM / transformers) is the GPU-side concern. pass@k
coverage uses *lenient* extraction so capability is not
undercounted on formatting — which makes the k=1 vanilla pass@k equal the lenient
accuracy by construction. CoT-gated pass@k additionally requires a valid chain, so
it is always <= the vanilla value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from pydantic import Field

from grpo_decomp.eval.code_reasoning import is_code_reasoning
from grpo_decomp.eval.cot import chain_is_valid, has_verifiable_chain
from grpo_decomp.eval.passk import estimate_pass_at_k
from grpo_decomp.grading import extract_lenient, extract_strict
from grpo_decomp.registries import verifier_for
from grpo_decomp.schemas import ProblemSet, Record


class PassK(Record):
    """pass@k at one k, both vanilla and CoT-gated."""

    k: int
    vanilla: float = Field(description="Unbiased pass@k under lenient extraction.")
    cot_gated: float = Field(description="pass@k requiring a correct answer AND a valid chain.")


class BatteryResult(Record):
    """The eval battery's measurements for one checkpoint over one problem set."""

    n_problems: int
    n_samples: int = Field(description="Completions sampled per problem (uniform).")
    strict_accuracy: float = Field(description="pass@1 under strict (boxed-only) extraction.")
    lenient_accuracy: float = Field(description="pass@1 under lenient extraction.")
    pass_at_k: tuple[PassK, ...]
    code_reasoning_frequency: float
    chain_coverage: float = Field(description="Fraction of completions with >=1 verifiable step.")


class _BatteryScores(NamedTuple):
    n: int
    strict_correct: int
    lenient_correct: int
    code_count: int
    chain_coverage_count: int
    lenient_counts: list[int]
    cot_counts: list[int]


def _uniform_sample_count(
    problems: ProblemSet, completions_by_id: Mapping[str, Sequence[str]]
) -> int:
    sizes = set()
    for problem in problems:
        if problem.id not in completions_by_id:
            raise ValueError(f"no completions for problem {problem.id!r}")
        sizes.add(len(completions_by_id[problem.id]))
    if len(sizes) != 1:
        raise ValueError(f"completions must be uniform per problem, got sizes {sorted(sizes)}")
    n = sizes.pop()
    if n == 0:
        raise ValueError("each problem needs >=1 completion")
    return n


def _score_completions(
    problems: ProblemSet, completions_by_id: Mapping[str, Sequence[str]], *, n: int | None = None
) -> _BatteryScores:
    """Score completions once and expose both aggregate and per-problem counts."""
    if n is None:
        n = _uniform_sample_count(problems, completions_by_id)
    check = verifier_for(problems.source)
    strict_correct = 0
    lenient_correct = 0
    code_count = 0
    chain_coverage_count = 0
    lenient_counts: list[int] = []
    cot_counts: list[int] = []

    for problem in problems:
        gold = problem.gold_answer
        per_problem_lenient = 0
        per_problem_cot = 0
        for completion in completions_by_id[problem.id]:
            strict_correct += check(extract_strict(completion), gold)
            lenient_ok = check(extract_lenient(completion), gold)
            lenient_correct += lenient_ok
            if lenient_ok:
                per_problem_lenient += 1
                if chain_is_valid(completion):
                    per_problem_cot += 1
            code_count += is_code_reasoning(completion)
            chain_coverage_count += has_verifiable_chain(completion)
        lenient_counts.append(per_problem_lenient)
        cot_counts.append(per_problem_cot)

    return _BatteryScores(
        n=n,
        strict_correct=strict_correct,
        lenient_correct=lenient_correct,
        code_count=code_count,
        chain_coverage_count=chain_coverage_count,
        lenient_counts=lenient_counts,
        cot_counts=cot_counts,
    )


def run_battery(
    problems: ProblemSet,
    completions_by_id: Mapping[str, Sequence[str]],
    *,
    k_values: Sequence[int],
) -> BatteryResult:
    """Score every problem's completions and return the battery's `BatteryResult`.

    Each problem must have the same number of completions `n`, and every `k` must
    satisfy ``1 <= k <= n``.
    """
    if not k_values:
        raise ValueError("k_values is empty")
    if len(problems) == 0:
        raise ValueError("problems is empty")

    n = _uniform_sample_count(problems, completions_by_id)
    for k in k_values:
        if not 1 <= k <= n:
            raise ValueError(f"each k must satisfy 1 <= k <= n={n}, got {k}")

    scores = _score_completions(problems, completions_by_id, n=n)
    total = len(problems) * n
    pass_at_k = tuple(
        PassK(
            k=k,
            vanilla=estimate_pass_at_k(scores.lenient_counts, k, n=n),
            cot_gated=estimate_pass_at_k(scores.cot_counts, k, n=n),
        )
        for k in k_values
    )
    return BatteryResult(
        n_problems=len(problems),
        n_samples=n,
        strict_accuracy=scores.strict_correct / total,
        lenient_accuracy=scores.lenient_correct / total,
        pass_at_k=pass_at_k,
        code_reasoning_frequency=scores.code_count / total,
        chain_coverage=scores.chain_coverage_count / total,
    )


def lenient_counts_by_problem(
    problems: ProblemSet, completions_by_id: Mapping[str, Sequence[str]]
) -> tuple[list[int], int]:
    """Per-problem count of lenient-correct completions (problem order), plus the uniform n.

    The building block `run_battery` scores pass@k from, exposed for sampled (n>1)
    multi-seed pass@k aggregation: ``estimate_pass_at_k(counts, k, n=n)`` for the panel and
    the per-problem ``pass_at_k(n, c, k)`` the base-anchor bootstrap resamples. Reuses the
    same lenient extraction + task verifier the battery uses, so counts match it exactly.
    """
    scores = _score_completions(problems, completions_by_id)
    return scores.lenient_counts, scores.n


def cot_counts_by_problem(
    problems: ProblemSet, completions_by_id: Mapping[str, Sequence[str]]
) -> tuple[list[int], int]:
    """Per-problem count of CoT-gated-correct completions (problem order), plus the uniform n.

    The CoT twin of `lenient_counts_by_problem`: a completion counts only when its lenient
    answer is correct AND its chain is valid (>=1 ``<<a op b=c>>`` step, all steps compute) —
    the stricter bar the CoT-Pass@K critique argues for. Always <= the lenient count per
    problem, so CoT-gated pass@k <= vanilla. Reuses the same extraction + verifier + chain
    check as `run_battery`, so these per-problem counts match its `cot_gated` pass@k exactly.
    """
    scores = _score_completions(problems, completions_by_id)
    return scores.cot_counts, scores.n


_EXTRACTORS = {"strict": extract_strict, "lenient": extract_lenient}


def grade(
    problems: ProblemSet,
    completion_by_id: Mapping[str, str],
    *,
    policy: str = "lenient",
) -> dict[str, bool]:
    """Grade one completion per problem (pass@1) into ``id -> correct``.

    This is the bridge from the eval layer to the stats layer: feed two arms'
    `grade(...)` outputs (same problem ids) to `compare()` for the paired delta.
    `policy` selects strict or lenient extraction. Every problem must have a
    completion, or it is an explicit error.
    """
    if policy not in _EXTRACTORS:
        raise ValueError(f"policy must be one of {tuple(_EXTRACTORS)}, got {policy!r}")
    extract = _EXTRACTORS[policy]
    check = verifier_for(problems.source)
    missing = [problem.id for problem in problems if problem.id not in completion_by_id]
    if missing:
        raise ValueError(f"no completion for {len(missing)} problem(s), e.g. {missing[:3]}")
    return {
        problem.id: check(extract(completion_by_id[problem.id]), problem.gold_answer)
        for problem in problems
    }
