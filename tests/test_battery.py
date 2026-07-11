"""Unit tests for the eval battery orchestration (no network)."""

from __future__ import annotations

import pytest

from grpo_decomp.eval.battery import counts_by_problem, run_battery
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet

_REF = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")


def _problems(*golds: str) -> ProblemSet:
    problems = tuple(Problem(id=f"p{i}", question="q", gold_answer=g) for i, g in enumerate(golds))
    return ProblemSet(source=_REF, problems=problems)


def _battery():
    problems = _problems("4", "12")
    completions = {
        "p0": [r"reasoning... \boxed{4}", "I think 5"],
        "p1": [r"<<3*4=12>> therefore \boxed{12}", "in python: print(12)"],
    }
    return run_battery(problems, completions, k_values=[1, 2])


def test_accuracies_under_both_policies() -> None:
    result = _battery()
    assert result.n_problems == 2
    assert result.n_samples == 2
    # strict: only the two boxed answers; lenient also recovers the unboxed "12".
    assert result.strict_accuracy == pytest.approx(0.5)
    assert result.lenient_accuracy == pytest.approx(0.75)


def test_pass_at_k_values_and_invariants() -> None:
    result = _battery()
    by_k = {p.k: p for p in result.pass_at_k}

    # k=1 vanilla pass@k equals the lenient accuracy by construction.
    assert by_k[1].vanilla == pytest.approx(result.lenient_accuracy)
    assert by_k[1].cot_gated == pytest.approx(0.25)
    assert by_k[2].vanilla == pytest.approx(1.0)
    assert by_k[2].cot_gated == pytest.approx(0.5)

    # CoT gating never exceeds vanilla.
    for p in result.pass_at_k:
        assert p.cot_gated <= p.vanilla


def test_cot_counts_are_a_subset_of_lenient() -> None:
    problems = _problems("4", "12")
    completions = {
        "p0": [r"reasoning... \boxed{4}", "I think 5"],
        "p1": [r"<<3*4=12>> therefore \boxed{12}", "in python: print(12)"],
    }
    lenient_counts, cot_counts, n = counts_by_problem(problems, completions)
    assert n == 2
    # p0: \boxed{4} is correct but has no calculator chain -> 0 gated. p1: only the
    # "<<3*4=12>>" sample is correct AND valid-chain ("print(12)" is unverifiable) -> 1.
    assert cot_counts == [0, 1]
    assert lenient_counts == [1, 2]
    # CoT-gated counts are a per-problem subset of lenient-correct -> the pass@k invariant.
    assert all(c <= lo for c, lo in zip(cot_counts, lenient_counts, strict=True))


def test_diagnostics() -> None:
    result = _battery()
    assert result.chain_coverage == pytest.approx(0.25)  # one <<...>> completion


def test_rejects_missing_completions() -> None:
    with pytest.raises(ValueError, match="no completions"):
        run_battery(_problems("4"), {}, k_values=[1])


def test_rejects_nonuniform_sample_counts() -> None:
    problems = _problems("4", "12")
    completions = {"p0": ["a", "b"], "p1": ["c"]}
    with pytest.raises(ValueError, match="uniform"):
        run_battery(problems, completions, k_values=[1])


def test_rejects_k_greater_than_n() -> None:
    with pytest.raises(ValueError, match="1 <= k <= n"):
        run_battery(_problems("4"), {"p0": ["a", "b"]}, k_values=[3])


def test_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="k_values is empty"):
        run_battery(_problems("4"), {"p0": ["a"]}, k_values=[])
    with pytest.raises(ValueError, match="problems is empty"):
        run_battery(ProblemSet(source=_REF, problems=()), {}, k_values=[1])
