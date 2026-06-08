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
from scipy.stats import t

from grpo_gain_decomp.eval.battery import cot_counts_by_problem, lenient_counts_by_problem
from grpo_gain_decomp.eval.code_reasoning import code_reasoning_frequency
from grpo_gain_decomp.eval.completions import CompletionSet
from grpo_gain_decomp.eval.cot import has_verifiable_chain
from grpo_gain_decomp.eval.passk import estimate_pass_at_k, pass_at_k
from grpo_gain_decomp.report.decomposition import MIN_SEEDS
from grpo_gain_decomp.schemas import Record


class Pass8MultiSeed(Record):
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
    preliminary: bool = Field(description=f"True below {MIN_SEEDS} seeds.")

    def headline(self) -> str:
        """The atomic verdict line: bounded-small (elicitation) vs expansion.

        The verdict keys off the *propagated* interval (the base anchor's sampling SE folded
        in), so a movement that only clears zero with the base held fixed does not read as
        expansion.
        """
        expanded = self.delta_propagated_ci_low > 0.0 and self.delta >= 0.10
        verdict = "expansion" if expanded else "bounded-small (elicitation)"
        tag = f"  [PRELIMINARY <{MIN_SEEDS} seeds]" if self.preliminary else ""
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


def _arm_metrics(cs: CompletionSet, k: int) -> _ArmMetrics:
    """All pass@k metrics for one sampled arm — vanilla and CoT-gated share the problem set."""
    counts, n = lenient_counts_by_problem(cs.problem_set(), cs.completions_by_id())
    cot_counts, _ = cot_counts_by_problem(cs.problem_set(), cs.completions_by_id())
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


def aggregate_passk_seeds(
    base: CompletionSet,
    correct_by_seed: Sequence[tuple[object, CompletionSet]],
    *,
    task: str,
    k: int = 8,
) -> Pass8MultiSeed:
    """Aggregate the base anchor + per-seed correct pass@k into a `Pass8MultiSeed` panel.

    `correct_by_seed` is ``(seed_label, CompletionSet)`` per training seed. The correct
    pass@k gets a seed-level t CI (>=2 seeds) capturing run-to-run variance; the base
    anchor gets a problem-bootstrap CI. delta = mean correct pass@k - base pass@k, with the
    seed-level interval shifted by the fixed anchor (whose own bootstrap CI is reported
    separately). All correct arms must share one `n`; the base anchor may differ.
    """
    # Lazy import: the one-sample anchor bootstrap needs numpy only, but keeping the import
    # local mirrors stats.bootstrap's separation of the (eval-extra) paired path.
    from grpo_gain_decomp.stats.bootstrap import bootstrap_mean_ci

    if not correct_by_seed:
        raise ValueError("no per-seed correct arms to aggregate")

    base_m = _arm_metrics(base, k)
    n_base = base_m.n
    _, base_ci_low, base_ci_high = bootstrap_mean_ci(
        [pass_at_k(n_base, c, k) for c in base_m.counts]
    )
    _, base_cot_ci_low, base_cot_ci_high = bootstrap_mean_ci(
        [pass_at_k(n_base, c, k) for c in base_m.cot_counts]
    )

    seeds = [str(label) for label, _ in correct_by_seed]
    pass1s: list[float] = []
    passks: list[float] = []
    cot_pass1s: list[float] = []
    cot_passks: list[float] = []
    crfs: list[float] = []
    chain_covs: list[float] = []
    n_corrects: set[int] = set()
    for _label, cs in correct_by_seed:
        m = _arm_metrics(cs, k)
        pass1s.append(m.pass1)
        passks.append(m.passk)
        cot_pass1s.append(m.cot_pass1)
        cot_passks.append(m.cot_passk)
        crfs.append(m.crf)
        chain_covs.append(m.chain_coverage)
        n_corrects.add(m.n)
    if len(n_corrects) != 1:
        raise ValueError(f"correct arms must share n; got {sorted(n_corrects)}")
    n_correct = n_corrects.pop()

    passk_arr = np.array(passks, dtype=float)
    cot_passk_arr = np.array(cot_passks, dtype=float)
    n_seeds = len(passk_arr)
    mean_passk = float(passk_arr.mean())
    mean_cot_passk = float(cot_passk_arr.mean())
    # Seed-level t half-widths over the per-seed pass@k (identical form to report/seeds.py):
    # the same critical value applied to the vanilla and CoT-gated between-seed spreads.
    if n_seeds >= 2:
        t_crit = float(t.ppf(0.975, n_seeds - 1))
        half = t_crit * float(passk_arr.std(ddof=1) / np.sqrt(n_seeds))
        cot_half = t_crit * float(cot_passk_arr.std(ddof=1) / np.sqrt(n_seeds))
        ci_kind = f"seed-level t, df={n_seeds - 1}"
    else:
        half = cot_half = 0.0
        ci_kind = "single-seed (no seed variance)"

    # Propagated Δ intervals fold the base anchor's problem-bootstrap half-width into the
    # seed-level half-width in quadrature (see the field docs): the anchor's finite-problem SE
    # is typically the dominant term, so the seed-level interval alone treats it as noiseless.
    delta = mean_passk - base_m.passk
    prop_half = float(np.hypot(half, (base_ci_high - base_ci_low) / 2.0))
    cot_delta = mean_cot_passk - base_m.cot_passk
    cot_prop_half = float(np.hypot(cot_half, (base_cot_ci_high - base_cot_ci_low) / 2.0))
    return Pass8MultiSeed(
        task=task,
        k=k,
        n_seeds=n_seeds,
        seeds=tuple(seeds),
        n_base=n_base,
        n_correct=n_correct,
        base_pass1=base_m.pass1,
        base_passk=base_m.passk,
        base_passk_ci_low=base_ci_low,
        base_passk_ci_high=base_ci_high,
        base_code_reasoning_freq=base_m.crf,
        per_seed_correct_pass1=tuple(pass1s),
        per_seed_correct_passk=tuple(passks),
        per_seed_code_reasoning_freq=tuple(crfs),
        mean_correct_pass1=float(np.mean(pass1s)),
        mean_correct_passk=mean_passk,
        correct_passk_ci_low=mean_passk - half,
        correct_passk_ci_high=mean_passk + half,
        delta=delta,
        delta_ci_low=delta - half,
        delta_ci_high=delta + half,
        delta_propagated_ci_low=delta - prop_half,
        delta_propagated_ci_high=delta + prop_half,
        base_cot_pass1=base_m.cot_pass1,
        base_cot_passk=base_m.cot_passk,
        base_cot_passk_ci_low=base_cot_ci_low,
        base_cot_passk_ci_high=base_cot_ci_high,
        base_chain_coverage=base_m.chain_coverage,
        per_seed_correct_cot_passk=tuple(cot_passks),
        mean_correct_cot_pass1=float(np.mean(cot_pass1s)),
        mean_correct_cot_passk=mean_cot_passk,
        mean_correct_chain_coverage=float(np.mean(chain_covs)),
        correct_cot_passk_ci_low=mean_cot_passk - cot_half,
        correct_cot_passk_ci_high=mean_cot_passk + cot_half,
        cot_delta=cot_delta,
        cot_delta_ci_low=cot_delta - cot_half,
        cot_delta_ci_high=cot_delta + cot_half,
        cot_delta_propagated_ci_low=cot_delta - cot_prop_half,
        cot_delta_propagated_ci_high=cot_delta + cot_prop_half,
        ci_kind=ci_kind,
        preliminary=n_seeds < MIN_SEEDS,
    )
