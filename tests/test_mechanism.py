"""Unit tests for the mechanism report (per-problem migration + length shift; no network)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from llm_grpo_gains.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
)
from llm_grpo_gains.report.mechanism import build_mechanism
from llm_grpo_gains.schemas import DatasetRef, Problem

_REF = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")
_C, _W = r"\boxed{7}", r"\boxed{0}"  # correct vs wrong, gold is "7"


def _cset(golds: Sequence[str], samples_by_id: dict[str, list[str]], n: int) -> CompletionSet:
    problems = tuple(Problem(id=f"p{i}", question="q", gold_answer=g) for i, g in enumerate(golds))
    provenance = GenerationProvenance(
        model="m",
        model_revision=None,
        backend="vllm",
        sampling=SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=64, n=n, seed=0),
        dataset=_REF,
        n_problems=len(problems),
        commit="c",
        dirty=False,
        python_version="3.11",
        package_versions={},
    )
    items = tuple(
        ProblemCompletions(problem=p, samples=tuple(samples_by_id[p.id])) for p in problems
    )
    return CompletionSet(provenance=provenance, items=items)


def test_build_mechanism_categorizes_each_migration_class() -> None:
    golds = ["7", "7", "7", "7"]
    # k=2, n_base=4, tau=0.5 -> base pass@2(4,1,2)=0.5 (>=tau), base pass@2(4,0,2)=0 (<tau).
    base = _cset(
        golds,
        {
            "p0": [_C, _C, _C, _W],  # base pass@1 .75 -> already reliable
            "p1": [_C, _W, _W, _W],  # base pass@1 .25 < tau <= base pass@2 .5 -> migration-eligible
            "p2": [_W, _W, _W, _W],  # base pass@2 0 -> new-capability-eligible
            "p3": [_W, _W, _W, _W],  # base 0; correct stays hard -> still hard
        },
        n=4,
    )
    seed_a = _cset(golds, {"p0": [_C, _C], "p1": [_C, _C], "p2": [_C, _C], "p3": [_W, _W]}, n=2)
    seed_b = _cset(golds, {"p0": [_C, _C], "p1": [_C, _W], "p2": [_C, _C], "p3": [_C, _W]}, n=2)

    rep = build_mechanism(base, [seed_a, seed_b], task="gsm8k-test", k=2, tau=0.5)

    assert rep.n_problems == 4
    assert rep.n_base == 4 and rep.n_correct_pooled == 4
    assert rep.frac_base_already_reliable == pytest.approx(0.25)  # p0
    assert rep.frac_migrated_to_reliable == pytest.approx(0.25)  # p1: within base pass@2 reach
    assert rep.frac_new_capability == pytest.approx(0.25)  # p2: outside base pass@2 envelope
    assert rep.frac_still_hard == pytest.approx(0.25)  # p3: correct pass@1 .25 < tau
    partition = (
        rep.frac_base_already_reliable
        + rep.frac_migrated_to_reliable
        + rep.frac_new_capability
        + rep.frac_still_hard
    )
    assert partition == pytest.approx(1.0)
    # migrated 1 / (migrated 1 + new 1) -> half the added reliability is within-envelope.
    assert rep.migration_share_of_gain == pytest.approx(0.5)
    assert rep.base_mean_chars > 0 and rep.correct_mean_chars > 0
    assert rep.base_mean_words > 0 and rep.correct_mean_words > 0


def test_build_mechanism_rejects_empty_and_mismatched() -> None:
    base = _cset(["7", "7"], {"p0": [_C], "p1": [_W]}, n=1)
    with pytest.raises(ValueError, match="no correct seeds"):
        build_mechanism(base, [], task="gsm8k-test", k=1)

    different = _cset(["7", "7", "7"], {"p0": [_C], "p1": [_C], "p2": [_C]}, n=1)
    with pytest.raises(ValueError, match="same problems"):
        build_mechanism(base, [different], task="gsm8k-test", k=1)
