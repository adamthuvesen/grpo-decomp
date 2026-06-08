"""Run the full eval battery over a checkpoint's completions and report results.

Operates on already-generated completions (problem id -> its `n` samples); the
generation backend (vLLM / transformers) is the GPU-side concern, deferred to
Phase 1/2. pass@k coverage uses *lenient* extraction so capability is not
undercounted on formatting — which makes the k=1 vanilla pass@k equal the lenient
accuracy by construction. CoT-gated pass@k additionally requires a valid chain, so
it is always <= the vanilla value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from grpo_gain_decomp.data.countdown import countdown_is_correct
from grpo_gain_decomp.eval.answers import extract_lenient, extract_strict, is_correct
from grpo_gain_decomp.eval.code_reasoning import is_code_reasoning
from grpo_gain_decomp.eval.cot import chain_is_valid, has_verifiable_chain
from grpo_gain_decomp.eval.passk import estimate_pass_at_k
from grpo_gain_decomp.schemas import DatasetRef, ProblemSet, Record

#: Signature shared by the math (`is_correct`) and Countdown (`countdown_is_correct`)
#: graders: an extracted answer + the gold key -> correct?
Verifier = Callable[[str | None, str], bool]


def verifier_for(source: DatasetRef) -> Verifier:
    """Select the grading verifier by task: the Countdown checker for generated Countdown
    sets, math-verify for the GSM8K family. Both consume the same extracted boxed answer,
    so strict/lenient extraction still applies uniformly.
    """
    return countdown_is_correct if source.name == "countdown" else is_correct


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

    total = len(problems) * n
    pass_at_k = tuple(
        PassK(
            k=k,
            vanilla=estimate_pass_at_k(lenient_counts, k, n=n),
            cot_gated=estimate_pass_at_k(cot_counts, k, n=n),
        )
        for k in k_values
    )
    return BatteryResult(
        n_problems=len(problems),
        n_samples=n,
        strict_accuracy=strict_correct / total,
        lenient_accuracy=lenient_correct / total,
        pass_at_k=pass_at_k,
        code_reasoning_frequency=code_count / total,
        chain_coverage=chain_coverage_count / total,
    )


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
