"""Mechanism of the GSM8K gain: reliability, not new capability.

The multi-seed pass@8 panel shows *that* the gain is elicitation (coverage barely moves); this
shows *why*, per problem, over the same completions. Two readouts:

- migration: at a reliability threshold ``tau``, classify each problem by whether the base
  already solves it first-try (pass@1 >= tau), whether the trained model now does (correct
  pass@1 >= tau), and — when the base did not — whether the base could still reach it within
  pass@k. The elicitation signature is problems that migrate from "base needs several tries"
  (base pass@1 < tau <= base pass@k) to "trained solves first try", with almost none genuinely
  new (outside the base's pass@k envelope). Expansion (Countdown) is the opposite.
- length: the mean completion-length shift (characters + whitespace words), base vs trained —
  the style change that accompanies the reliability gain.

Reads the same base + per-seed correct CompletionSets as the pass@8 panel (correct pooled over
seeds for a seed-averaged per-problem pass@1). Deterministic; CPU-only.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from llm_grpo_gains.eval.battery import lenient_counts_by_problem
from llm_grpo_gains.eval.completions import CompletionSet
from llm_grpo_gains.eval.passk import pass_at_k
from llm_grpo_gains.schemas import Record

#: Default pass-rate cutoff for classifying a problem as first-try reliable.
DEFAULT_RELIABILITY_TAU = 0.5


class MechanismReport(Record):
    """Per-problem base->trained migration + the completion-length shift behind the verdict."""

    task: str
    n_problems: int
    n_base: int = Field(description="Base samples per problem.")
    n_correct_pooled: int = Field(description="Correct samples per problem, pooled over seeds.")
    k: int = Field(description="pass@k envelope level for 'the base could still reach it'.")
    reliability_threshold: float = Field(description="pass-rate cutoff tau for 'reliably solves'.")

    base_mean_chars: float
    correct_mean_chars: float
    base_mean_words: float
    correct_mean_words: float

    # Mutually exclusive, exhaustive problem categories (the four fractions sum to 1).
    frac_base_already_reliable: float = Field(description="base pass@1 >= tau (RL adds nothing).")
    frac_migrated_to_reliable: float = Field(
        description="base pass@1 < tau <= base pass@k and correct pass@1 >= tau: the elicitation "
        "signature — a within-reach problem made first-try reliable."
    )
    frac_new_capability: float = Field(
        description="base pass@k < tau <= correct pass@1: outside the base's pass@k envelope."
    )
    frac_still_hard: float = Field(description="correct pass@1 < tau (not reliably solved).")
    migration_share_of_gain: float = Field(
        description="migrated / (migrated + new): of the problems the trained model newly solves "
        "first-try reliably, the share already within the base's pass@k reach. ~1 is pure "
        "elicitation; lower means real expansion."
    )

    def headline(self) -> str:
        """One-line mechanism verdict: how much of the added reliability is within-envelope."""
        share = self.migration_share_of_gain * 100
        return (
            f"{self.task}: of the first-try reliability the trained model adds, {share:.0f}% is "
            f"migration within the base pass@{self.k} envelope (elicitation), {100 - share:.0f}% "
            f"new capability; mean completion {self.base_mean_words:.0f}->"
            f"{self.correct_mean_words:.0f} words"
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def build_mechanism(
    base: CompletionSet,
    correct_by_seed: Sequence[CompletionSet],
    *,
    task: str,
    k: int = 8,
    tau: float = DEFAULT_RELIABILITY_TAU,
) -> MechanismReport:
    """Classify each problem's base->trained migration and measure the completion-length shift.

    `correct_by_seed` are the per-seed correct CompletionSets (pooled here for a seed-averaged
    per-problem pass@1). All arms must cover the same problems in the same order.
    """
    if not correct_by_seed:
        raise ValueError("no correct seeds to aggregate")

    base_counts, n_base = lenient_counts_by_problem(base.problem_set(), base.completions_by_id())
    if not 1 <= k <= n_base:
        raise ValueError(f"pass@{k} envelope needs 1<=k<=n_base; base has n={n_base}")
    base_ids = tuple(problem.id for problem in base.problem_set())

    n_correct_each: set[int] = set()
    pooled = [0] * len(base_counts)
    for cs in correct_by_seed:
        if tuple(problem.id for problem in cs.problem_set()) != base_ids:
            raise ValueError("base and correct arms must cover the same problems in the same order")
        counts, n_c = lenient_counts_by_problem(cs.problem_set(), cs.completions_by_id())
        n_correct_each.add(n_c)
        for i, c in enumerate(counts):
            pooled[i] += c
    if len(n_correct_each) != 1:
        raise ValueError(f"correct arms must share n; got {sorted(n_correct_each)}")
    n_pool = n_correct_each.pop() * len(correct_by_seed)

    base_already_reliable = migrated = new_capability = still_hard = 0
    for c_base, c_correct in zip(base_counts, pooled, strict=True):
        base_p1 = c_base / n_base
        correct_p1 = c_correct / n_pool
        if correct_p1 < tau:
            still_hard += 1
        elif base_p1 >= tau:
            base_already_reliable += 1
        elif pass_at_k(n_base, c_base, k) >= tau:
            migrated += 1
        else:
            new_capability += 1
    n = len(base_counts)
    gain = migrated + new_capability

    base_samples = [sample for item in base.items for sample in item.samples]
    correct_samples = [
        sample for cs in correct_by_seed for item in cs.items for sample in item.samples
    ]
    return MechanismReport(
        task=task,
        n_problems=n,
        n_base=n_base,
        n_correct_pooled=n_pool,
        k=k,
        reliability_threshold=tau,
        base_mean_chars=_mean([len(sample) for sample in base_samples]),
        correct_mean_chars=_mean([len(sample) for sample in correct_samples]),
        base_mean_words=_mean([len(sample.split()) for sample in base_samples]),
        correct_mean_words=_mean([len(sample.split()) for sample in correct_samples]),
        frac_base_already_reliable=base_already_reliable / n,
        frac_migrated_to_reliable=migrated / n,
        frac_new_capability=new_capability / n,
        frac_still_hard=still_hard / n,
        migration_share_of_gain=(migrated / gain) if gain else 0.0,
    )
