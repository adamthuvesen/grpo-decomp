"""Unit tests for multi-seed pass@k coverage aggregation (CPU; uses the eval extra)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import t

from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
)
from grpo_decomp.report.passk_seeds import PassKMultiSeed, aggregate_passk_seeds
from grpo_decomp.schemas import DatasetRef, Problem

_REF = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")
_C7, _C9, _WRONG = r"\boxed{7}", r"\boxed{9}", r"\boxed{0}"
# Correct AND carrying a valid <<a op b=c>> chain -> counts toward CoT-gated pass@k too.
_CHAIN7, _CHAIN9 = r"<<3+4=7>> so \boxed{7}", r"<<4+5=9>> so \boxed{9}"


def _cset(samples_by_id: dict[str, list[str]], n: int) -> CompletionSet:
    """A sampled (temp>0) CompletionSet over two problems with golds 7 and 9."""
    problems = (
        Problem(id="p0", question="q", gold_answer="7"),
        Problem(id="p1", question="q", gold_answer="9"),
    )
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


# pass@1 per arm = mean over problems of (correct / n). At n=2:
_BASE = _cset({"p0": [_C7, _WRONG], "p1": [_WRONG, _WRONG]}, n=2)  # (0.5 + 0)/2 = 0.25
_SEED_A = _cset({"p0": [_C7, _C7], "p1": [_WRONG, _WRONG]}, n=2)  # (1 + 0)/2  = 0.50
_SEED_B = _cset({"p0": [_C7, _C7], "p1": [_C9, _WRONG]}, n=2)  # (1 + 0.5)/2 = 0.75
_SEED_C = _cset({"p0": [_C7, _C7], "p1": [_C9, _C9]}, n=2)  # (1 + 1)/2   = 1.00


def test_aggregate_passk_seeds_t_ci_and_delta() -> None:
    panel = aggregate_passk_seeds(
        _BASE, [(0, _SEED_A), (1, _SEED_B), (2, _SEED_C)], task="gsm8k-test", k=1
    )
    assert panel.seeds == ("0", "1", "2")
    assert panel.n_base == 2 and panel.n_correct == 2
    assert panel.per_seed_correct_passk == pytest.approx((0.5, 0.75, 1.0))
    assert panel.mean_correct_passk == pytest.approx(0.75)
    assert panel.base_passk == pytest.approx(0.25)
    assert panel.delta == pytest.approx(0.5)
    assert panel.preliminary is False  # 3 >= MIN_SEEDS
    assert "seed-level t, df=2" in panel.ci_kind

    # The correct/Δ half-width matches a hand scipy t-interval over the per-seed values.
    half = float(t.ppf(0.975, 2)) * float(np.std([0.5, 0.75, 1.0], ddof=1) / np.sqrt(3))
    assert panel.correct_passk_ci_high - panel.mean_correct_passk == pytest.approx(half)
    assert panel.delta_ci_low == pytest.approx(0.5 - half)
    assert panel.delta_ci_high == pytest.approx(0.5 + half)
    # The base anchor is not noiseless: its own bootstrap CI brackets the point estimate.
    assert panel.base_passk_ci_low <= panel.base_passk <= panel.base_passk_ci_high
    # The propagated Δ interval folds in the anchor's bootstrap half-width -> strictly wider
    # than the seed-level interval (base p@1 over [0.5, 0.0] has real problem-sampling spread).
    assert panel.delta_propagated_ci_low < panel.delta_ci_low
    assert panel.delta_propagated_ci_high > panel.delta_ci_high
    # These fixtures carry no <<...>> chains, so every CoT-gated metric is 0 (unverifiable),
    # and the cot <= vanilla invariant holds trivially.
    assert panel.base_cot_passk == 0.0
    assert panel.mean_correct_cot_passk == 0.0
    assert panel.cot_delta == 0.0
    assert panel.base_chain_coverage == 0.0  # zero parseable <<a op b=c>> steps


def test_aggregate_passk_seeds_cot_gated_is_a_subset_of_vanilla() -> None:
    # p0 mixes a chained solve with a bare-boxed solve; CoT-gating drops the latter, so the
    # CoT-gated level sits strictly below vanilla while obeying the same CI structure.
    base = _cset({"p0": [_CHAIN7, _C7], "p1": [_WRONG, _WRONG]}, n=2)
    seed_a = _cset({"p0": [_CHAIN7, _C7], "p1": [_WRONG, _WRONG]}, n=2)
    seed_b = _cset({"p0": [_CHAIN7, _CHAIN7], "p1": [_CHAIN9, _WRONG]}, n=2)
    panel = aggregate_passk_seeds(base, [(0, seed_a), (1, seed_b)], task="gsm8k-test", k=1)

    # base p0 lenient=2 (both boxed 7) but cot=1 (only the <<3+4=7>> sample) -> 0.25 < 0.50.
    assert panel.base_passk == pytest.approx(0.5)
    assert panel.base_cot_passk == pytest.approx(0.25)
    assert panel.base_cot_passk < panel.base_passk
    assert panel.mean_correct_cot_passk <= panel.mean_correct_passk
    assert panel.base_chain_coverage == pytest.approx(0.25)  # 1 of 4 base completions has a chain
    # The CoT twin carries the same propagated-widening structure as the vanilla interval.
    assert panel.cot_delta_propagated_ci_low <= panel.cot_delta_ci_low
    assert panel.cot_delta_propagated_ci_high >= panel.cot_delta_ci_high


def test_below_min_seeds_is_preliminary() -> None:
    panel = aggregate_passk_seeds(_BASE, [(0, _SEED_A), (1, _SEED_B)], task="gsm8k-test", k=1)
    assert panel.n_seeds == 2
    assert panel.preliminary is True


def test_empty_correct_arms_is_explicit_error() -> None:
    with pytest.raises(ValueError, match="no per-seed correct arms"):
        aggregate_passk_seeds(_BASE, [], task="gsm8k-test", k=1)


def test_correct_arms_must_share_n() -> None:
    seed_n4 = _cset({"p0": [_C7] * 4, "p1": [_WRONG] * 4}, n=4)
    with pytest.raises(ValueError, match="share n"):
        aggregate_passk_seeds(_BASE, [(0, _SEED_A), (1, seed_n4)], task="gsm8k-test", k=1)


def test_k_greater_than_n_is_explicit_error() -> None:
    with pytest.raises(ValueError, match="1<=k<=n"):
        aggregate_passk_seeds(_BASE, [(0, _SEED_A)], task="gsm8k-test", k=8)


def test_schema_rejects_unknown_fields() -> None:
    panel = aggregate_passk_seeds(_BASE, [(0, _SEED_A), (1, _SEED_B)], task="gsm8k-test", k=1)
    with pytest.raises(ValidationError):
        PassKMultiSeed(**{**panel.model_dump(), "bogus": 1})
