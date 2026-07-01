"""Aggregate the pass@k coverage panel (base anchor vs correct) across seed replicates.

The decomposition's *interpretive* axis: did RL expand pass@k coverage (new capability)
or merely elicit it (reliability on what the base could already do)? The single-seed panel
left that verdict resting on one seed. This aggregates the *correct* arm over training
seeds (a seed-level t CI capturing run-to-run variance) against the single, seed-independent
base anchor (its own problem-bootstrap CI), and reports delta = mean correct pass@k - base
pass@k. The base anchor is not treated as noiseless. Mirrors `report/seeds.py`, which does
the same seed-level aggregation for the pass@1 placebo comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import Field

from grpo_decomp.eval.battery import counts_by_problem
from grpo_decomp.eval.code_reasoning import code_reasoning_frequency
from grpo_decomp.eval.completions import CompletionSet
from grpo_decomp.eval.cot import has_verifiable_chain
from grpo_decomp.eval.passk import estimate_pass_at_k, pass_at_k
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.schemas import Record
from grpo_decomp.stats.bootstrap import bootstrap_mean_ci
from grpo_decomp.stats.seed_aggregate import seed_level_t_half_width

#: Propagated pass@k Δ above this (pp) with CI excluding zero reads as expansion, not elicitation.
EXPANSION_DELTA_THRESHOLD = 0.10


class PassKMultiSeed(Record):
    """Multi-seed pass@k coverage panel: base anchor vs per-seed correct, with CIs.

    Field names are k-generic (`base_passk`, `per_seed_correct_passk`) with the level in
    `k`; the committed artifact records k=8 (the coverage level the elicitation/expansion
    verdict rests on).
    """

    task: str
    k: int = Field(description="The pass@k coverage level the verdict rests on (e.g. 8).")
    n_seeds: int
    seeds: tuple[str, ...] = Field(description="Per-seed labels (training seeds), input order.")
    n_base: int = Field(description="Samples per problem for the base anchor.")
    n_correct: int = Field(description="Samples per problem for each correct seed.")

    base_pass1: float
    base_passk: float = Field(description="Base pass@k anchor (seed-independent).")
    base_passk_ci_low: float = Field(description="Problem-bootstrap CI low on the base anchor.")
    base_passk_ci_high: float
    base_code_reasoning_freq: float

    per_seed_correct_pass1: tuple[float, ...]
    per_seed_correct_passk: tuple[float, ...]
    per_seed_code_reasoning_freq: tuple[float, ...]
    mean_correct_pass1: float
    mean_correct_passk: float
    correct_passk_ci_low: float = Field(description="Seed-level t CI low on correct pass@k.")
    correct_passk_ci_high: float

    delta: float = Field(description="mean correct pass@k - base pass@k.")
    delta_ci_low: float = Field(
        description="Seed-level t CI on Δ — between-seed variance only, base treated as a fixed "
        "anchor. Understates Δ uncertainty when the anchor's own sampling SE is large."
    )
    delta_ci_high: float
    delta_propagated_ci_low: float = Field(
        description="Δ CI folding the base anchor's problem-bootstrap half-width into the "
        "seed-level half-width (quadrature). The honest headline interval; conservative "
        "(base and correct share the problem set, so pairing would only tighten it)."
    )
    delta_propagated_ci_high: float

    # CoT-gated (valid-chain) twins: pass@k counting a sample only when its answer is correct
    # AND its <<a op b=c>> chain verifies. The CoT-Pass@K critique's stricter yardstick — does
    # RL move *valid-chain* coverage, or just right-answer coverage? Always <= the vanilla twin.
    base_cot_pass1: float
    base_cot_passk: float = Field(description="Base CoT-gated pass@k anchor (seed-independent).")
    base_cot_passk_ci_low: float = Field(description="Problem-bootstrap CI low on the CoT anchor.")
    base_cot_passk_ci_high: float
    base_chain_coverage: float = Field(
        description="Fraction of base completions with >=1 parseable <<a op b=c>> step."
    )
    per_seed_correct_cot_passk: tuple[float, ...]
    mean_correct_cot_pass1: float
    mean_correct_cot_passk: float
    mean_correct_chain_coverage: float = Field(
        description="Seed-mean fraction of correct completions with a verifiable chain."
    )
    correct_cot_passk_ci_low: float = Field(
        description="Seed-level t CI low on correct CoT pass@k."
    )
    correct_cot_passk_ci_high: float
    cot_delta: float = Field(description="mean correct CoT pass@k - base CoT pass@k.")
    cot_delta_ci_low: float = Field(description="Seed-level t CI on the CoT Δ (anchor fixed).")
    cot_delta_ci_high: float
    cot_delta_propagated_ci_low: float = Field(
        description="CoT Δ CI folding the CoT anchor's problem-bootstrap half-width into the "
        "seed-level half-width (quadrature) — the honest CoT-gated headline interval."
    )
    cot_delta_propagated_ci_high: float

    ci_kind: str = Field(description="How the correct/seed-level Δ interval was formed.")
    preliminary: bool = Field(description=f"True below {MIN_HEADLINE_SEEDS} seeds.")

    def headline(self) -> str:
        """The atomic verdict line: bounded-small (elicitation) vs expansion.

        The verdict keys off the *propagated* interval (the base anchor's sampling SE folded
        in), so a movement that only clears zero with the base held fixed does not read as
        expansion.
        """
        expanded = self.delta_propagated_ci_low > 0.0 and self.delta >= EXPANSION_DELTA_THRESHOLD
        verdict = "expansion" if expanded else "bounded-small (elicitation)"
        tag = f"  [PRELIMINARY <{MIN_HEADLINE_SEEDS} seeds]" if self.preliminary else ""
        return (
            f"over {self.n_seeds} seed(s) on {self.task}, correct pass@{self.k} "
            f"{self.mean_correct_passk * 100:.1f}% vs base {self.base_passk * 100:.1f}%: "
            f"Δ {self.delta * 100:+.1f} pp (with anchor [{self.delta_propagated_ci_low * 100:.1f}, "
            f"{self.delta_propagated_ci_high * 100:.1f}]; seed-level "
            f"[{self.delta_ci_low * 100:.1f}, {self.delta_ci_high * 100:.1f}]) → {verdict}{tag}"
        )

    def cot_headline(self) -> str:
        """The CoT-gated (valid-chain) line — the CoT-Pass@K yardstick beside the vanilla one.

        Counts a solve only with a verified ``<<a op b=c>>`` chain. If such chains are rare the
        level reads low (a coverage limit of the non-neural check, not necessarily a reasoning
        gap), but the base-vs-correct contrast still answers whether RL moved *valid-chain*
        coverage rather than only right-answer coverage.
        """
        return (
            f"CoT-gated pass@{self.k}: correct {self.mean_correct_cot_passk * 100:.1f}% vs "
            f"base {self.base_cot_passk * 100:.1f}%: Δ {self.cot_delta * 100:+.1f} pp (with anchor "
            f"[{self.cot_delta_propagated_ci_low * 100:.1f}, "
            f"{self.cot_delta_propagated_ci_high * 100:.1f}])"
        )


class _ArmMetrics(NamedTuple):
    """One sampled arm's pass@1/pass@k (vanilla + CoT-gated), code-reasoning frequency, and the
    per-problem lenient/CoT counts that both the panel estimate and the base bootstrap resample.
    """

    pass1: float
    passk: float
    cot_pass1: float
    cot_passk: float
    crf: float
    chain_coverage: float
    counts: list[int]
    cot_counts: list[int]
    n: int


class _CorrectMetrics(NamedTuple):
    seeds: tuple[str, ...]
    n: int
    pass1s: tuple[float, ...]
    passks: tuple[float, ...]
    cot_pass1s: tuple[float, ...]
    cot_passks: tuple[float, ...]
    crfs: tuple[float, ...]
    chain_coverages: tuple[float, ...]


class _CoverageStats(NamedTuple):
    mean: float
    ci: tuple[float, float]
    delta: float
    delta_ci: tuple[float, float]
    delta_propagated_ci: tuple[float, float]
    ci_kind: str


def _problem_axis(cs: CompletionSet) -> tuple[object, ...]:
    return tuple(item.problem for item in cs.items)


def _validate_same_axis(
    base: CompletionSet, correct_by_seed: Sequence[tuple[object, CompletionSet]]
) -> None:
    base_axis = _problem_axis(base)
    for label, cs in correct_by_seed:
        if cs.provenance.dataset != base.provenance.dataset:
            raise ValueError(
                f"correct seed {label}: dataset metadata does not match base "
                f"{base.provenance.dataset.model_dump()}"
            )
        if cs.provenance.prompt_strategy != base.provenance.prompt_strategy:
            raise ValueError(
                f"correct seed {label}: prompt strategy {cs.provenance.prompt_strategy!r} "
                f"does not match base {base.provenance.prompt_strategy!r}"
            )
        if _problem_axis(cs) != base_axis:
            raise ValueError(f"correct seed {label}: problem records do not match base")


def _interval(center: float, half_width: float) -> tuple[float, float]:
    return center - half_width, center + half_width


def _bootstrap_passk_ci(counts: Sequence[int], *, n: int, k: int) -> tuple[float, float]:
    """Problem-bootstrap CI for one seed-independent pass@k anchor."""
    _, ci_low, ci_high = bootstrap_mean_ci([pass_at_k(n, count, k) for count in counts])
    return ci_low, ci_high


def _delta_intervals(
    mean: float, anchor: float, seed_half_width: float, anchor_ci: tuple[float, float]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """Δ plus its seed-level and anchor-propagated intervals."""
    delta = mean - anchor
    anchor_half_width = (anchor_ci[1] - anchor_ci[0]) / 2.0
    propagated_half_width = float(np.hypot(seed_half_width, anchor_half_width))
    return delta, _interval(delta, seed_half_width), _interval(delta, propagated_half_width)


def _coverage_stats(
    values: Sequence[float], *, anchor: float, anchor_ci: tuple[float, float]
) -> _CoverageStats:
    values_array = np.array(values, dtype=float)
    mean = float(values_array.mean())
    half_width, ci_kind = seed_level_t_half_width(values_array)
    delta, delta_ci, delta_propagated_ci = _delta_intervals(mean, anchor, half_width, anchor_ci)
    return _CoverageStats(
        mean=mean,
        ci=_interval(mean, half_width),
        delta=delta,
        delta_ci=delta_ci,
        delta_propagated_ci=delta_propagated_ci,
        ci_kind=ci_kind,
    )


def _arm_metrics(cs: CompletionSet, k: int) -> _ArmMetrics:
    """All pass@k metrics for one sampled arm — vanilla and CoT-gated share the problem set."""
    counts, cot_counts, n = counts_by_problem(cs.problem_set(), cs.completions_by_id())
    if not 1 <= k <= n:
        raise ValueError(f"pass@{k} needs 1<=k<=n; arm has n={n}")
    samples = [sample for item in cs.items for sample in item.samples]
    # Chain coverage is the companion to the CoT-gated metric: the fraction of completions with
    # any parseable <<a op b=c>> step. When it is ~0 the model isn't emitting that format at all,
    # so a low CoT-gated pass@k is a proxy-coverage limit, not an invalid-reasoning verdict.
    chain_cov = sum(has_verifiable_chain(sample) for sample in samples) / len(samples)
    return _ArmMetrics(
        pass1=estimate_pass_at_k(counts, 1, n=n),
        passk=estimate_pass_at_k(counts, k, n=n),
        cot_pass1=estimate_pass_at_k(cot_counts, 1, n=n),
        cot_passk=estimate_pass_at_k(cot_counts, k, n=n),
        crf=code_reasoning_frequency(samples),
        chain_coverage=chain_cov,
        counts=counts,
        cot_counts=cot_counts,
        n=n,
    )


def _collect_correct_metrics(
    correct_by_seed: Sequence[tuple[object, CompletionSet]], k: int
) -> _CorrectMetrics:
    seeds: list[str] = []
    pass1s: list[float] = []
    passks: list[float] = []
    cot_pass1s: list[float] = []
    cot_passks: list[float] = []
    crfs: list[float] = []
    chain_coverages: list[float] = []
    n_corrects: set[int] = set()
    for label, cs in correct_by_seed:
        metrics = _arm_metrics(cs, k)
        seeds.append(str(label))
        pass1s.append(metrics.pass1)
        passks.append(metrics.passk)
        cot_pass1s.append(metrics.cot_pass1)
        cot_passks.append(metrics.cot_passk)
        crfs.append(metrics.crf)
        chain_coverages.append(metrics.chain_coverage)
        n_corrects.add(metrics.n)
    if len(n_corrects) != 1:
        raise ValueError(f"correct arms must share n; got {sorted(n_corrects)}")
    return _CorrectMetrics(
        seeds=tuple(seeds),
        n=n_corrects.pop(),
        pass1s=tuple(pass1s),
        passks=tuple(passks),
        cot_pass1s=tuple(cot_pass1s),
        cot_passks=tuple(cot_passks),
        crfs=tuple(crfs),
        chain_coverages=tuple(chain_coverages),
    )


def aggregate_passk_seeds(
    base: CompletionSet,
    correct_by_seed: Sequence[tuple[object, CompletionSet]],
    *,
    task: str,
    k: int = 8,
) -> PassKMultiSeed:
    """Aggregate the base anchor + per-seed correct pass@k into a `PassKMultiSeed` panel.

    `correct_by_seed` is ``(seed_label, CompletionSet)`` per training seed. The correct
    pass@k gets a seed-level t CI (>=2 seeds) capturing run-to-run variance; the base
    anchor gets a problem-bootstrap CI. delta = mean correct pass@k - base pass@k, with the
    seed-level interval shifted by the fixed anchor (whose own bootstrap CI is reported
    separately). All correct arms must share one `n`; the base anchor may differ.
    """
    if not correct_by_seed:
        raise ValueError("no per-seed correct arms to aggregate")
    _validate_same_axis(base, correct_by_seed)

    base_m = _arm_metrics(base, k)
    n_base = base_m.n
    base_ci = _bootstrap_passk_ci(base_m.counts, n=n_base, k=k)
    base_cot_ci = _bootstrap_passk_ci(base_m.cot_counts, n=n_base, k=k)

    correct = _collect_correct_metrics(correct_by_seed, k)
    vanilla = _coverage_stats(correct.passks, anchor=base_m.passk, anchor_ci=base_ci)
    cot = _coverage_stats(correct.cot_passks, anchor=base_m.cot_passk, anchor_ci=base_cot_ci)
    return PassKMultiSeed(
        task=task,
        k=k,
        n_seeds=len(correct.passks),
        seeds=correct.seeds,
        n_base=n_base,
        n_correct=correct.n,
        base_pass1=base_m.pass1,
        base_passk=base_m.passk,
        base_passk_ci_low=base_ci[0],
        base_passk_ci_high=base_ci[1],
        base_code_reasoning_freq=base_m.crf,
        per_seed_correct_pass1=correct.pass1s,
        per_seed_correct_passk=correct.passks,
        per_seed_code_reasoning_freq=correct.crfs,
        mean_correct_pass1=float(np.mean(correct.pass1s)),
        mean_correct_passk=vanilla.mean,
        correct_passk_ci_low=vanilla.ci[0],
        correct_passk_ci_high=vanilla.ci[1],
        delta=vanilla.delta,
        delta_ci_low=vanilla.delta_ci[0],
        delta_ci_high=vanilla.delta_ci[1],
        delta_propagated_ci_low=vanilla.delta_propagated_ci[0],
        delta_propagated_ci_high=vanilla.delta_propagated_ci[1],
        base_cot_pass1=base_m.cot_pass1,
        base_cot_passk=base_m.cot_passk,
        base_cot_passk_ci_low=base_cot_ci[0],
        base_cot_passk_ci_high=base_cot_ci[1],
        base_chain_coverage=base_m.chain_coverage,
        per_seed_correct_cot_passk=correct.cot_passks,
        mean_correct_cot_pass1=float(np.mean(correct.cot_pass1s)),
        mean_correct_cot_passk=cot.mean,
        mean_correct_chain_coverage=float(np.mean(correct.chain_coverages)),
        correct_cot_passk_ci_low=cot.ci[0],
        correct_cot_passk_ci_high=cot.ci[1],
        cot_delta=cot.delta,
        cot_delta_ci_low=cot.delta_ci[0],
        cot_delta_ci_high=cot.delta_ci[1],
        cot_delta_propagated_ci_low=cot.delta_propagated_ci[0],
        cot_delta_propagated_ci_high=cot.delta_propagated_ci[1],
        ci_kind=vanilla.ci_kind,
        preliminary=len(correct.passks) < MIN_HEADLINE_SEEDS,
    )
