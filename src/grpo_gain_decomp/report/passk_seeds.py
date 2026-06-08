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

import numpy as np
from pydantic import Field
from scipy.stats import t

from grpo_gain_decomp.eval.battery import lenient_counts_by_problem
from grpo_gain_decomp.eval.code_reasoning import code_reasoning_frequency
from grpo_gain_decomp.eval.completions import CompletionSet
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


def _arm_metrics(cs: CompletionSet, k: int) -> tuple[float, float, float, list[int], int]:
    """``(pass1, passk, code_reasoning_freq, per_problem_counts, n)`` for one sampled arm."""
    counts, n = lenient_counts_by_problem(cs.problem_set(), cs.completions_by_id())
    if not 1 <= k <= n:
        raise ValueError(f"pass@{k} needs 1<=k<=n; arm has n={n}")
    pass1 = estimate_pass_at_k(counts, 1, n=n)
    passk = estimate_pass_at_k(counts, k, n=n)
    crf = code_reasoning_frequency([sample for item in cs.items for sample in item.samples])
    return pass1, passk, crf, counts, n


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

    base_pass1, base_passk, base_crf, base_counts, n_base = _arm_metrics(base, k)
    base_passk_values = [pass_at_k(n_base, c, k) for c in base_counts]
    _, base_ci_low, base_ci_high = bootstrap_mean_ci(base_passk_values)

    seeds = [str(label) for label, _ in correct_by_seed]
    pass1s: list[float] = []
    passks: list[float] = []
    crfs: list[float] = []
    n_corrects: set[int] = set()
    for _label, cs in correct_by_seed:
        pass1, passk, crf, _counts, n_c = _arm_metrics(cs, k)
        pass1s.append(pass1)
        passks.append(passk)
        crfs.append(crf)
        n_corrects.add(n_c)
    if len(n_corrects) != 1:
        raise ValueError(f"correct arms must share n; got {sorted(n_corrects)}")
    n_correct = n_corrects.pop()

    passk_arr = np.array(passks, dtype=float)
    n_seeds = len(passk_arr)
    mean_passk = float(passk_arr.mean())
    # Seed-level t interval over per-seed correct pass@k (identical form to report/seeds.py).
    if n_seeds >= 2:
        half = float(t.ppf(0.975, n_seeds - 1)) * float(passk_arr.std(ddof=1) / np.sqrt(n_seeds))
        ci_kind = f"seed-level t, df={n_seeds - 1}"
    else:
        half = 0.0
        ci_kind = "single-seed (no seed variance)"

    delta = mean_passk - base_passk
    # Propagated Δ interval: combine the seed-level half-width with the base anchor's
    # problem-bootstrap half-width in quadrature. The anchor carries no training-seed variance,
    # but its pass@k is estimated over finite problems and that SE is typically the dominant
    # term — so the seed-level interval alone treats the anchor as noiseless and understates Δ.
    # Conservative: base and correct share the problem set (positively correlated), so a paired
    # estimate would only tighten this. Guaranteed to contain the seed-level interval.
    base_half = (base_ci_high - base_ci_low) / 2.0
    prop_half = float(np.hypot(half, base_half))
    return Pass8MultiSeed(
        task=task,
        k=k,
        n_seeds=n_seeds,
        seeds=tuple(seeds),
        n_base=n_base,
        n_correct=n_correct,
        base_pass1=base_pass1,
        base_passk=base_passk,
        base_passk_ci_low=base_ci_low,
        base_passk_ci_high=base_ci_high,
        base_code_reasoning_freq=base_crf,
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
        ci_kind=ci_kind,
        preliminary=n_seeds < MIN_SEEDS,
    )
