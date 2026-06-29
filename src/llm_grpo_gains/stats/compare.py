"""Compare two models on a shared test set: delta, CI, McNemar p, and headline.

The headline string is the report's atomic claim — every accuracy delta carries a
bootstrap CI and a paired significance test, never a bare point estimate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pydantic import Field

from llm_grpo_gains.schemas import Record
from llm_grpo_gains.stats.bootstrap import PAIRED_BOOTSTRAP_SEED, paired_bootstrap_ci
from llm_grpo_gains.stats.significance import mcnemar


class Comparison(Record):
    """A paired comparison of model B against baseline A on a shared test set."""

    label_a: str
    label_b: str
    n: int
    accuracy_a: float
    accuracy_b: float
    delta: float = Field(description="acc_b - acc_a (risk difference / effect size).")
    ci_low: float
    ci_high: float
    p_value: float
    n_discordant: int
    test: str = Field(description="'exact-binomial' or 'chi2'.")

    def headline(self) -> str:
        """The report's atomic claim, e.g. 'rl beats base by 5.2% (95% CI [...]; ...)'."""
        verb = "beats" if self.delta >= 0 else "trails"
        return (
            f"{self.label_b} {verb} {self.label_a} by {abs(self.delta) * 100:.1f}% "
            f"(95% CI [{self.ci_low * 100:.1f}, {self.ci_high * 100:.1f}]; "
            f"McNemar p={self.p_value:.3g}; n={self.n})"
        )


def compare(
    label_a: str,
    correct_a: Mapping[str, bool],
    label_b: str,
    correct_b: Mapping[str, bool],
    *,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> Comparison:
    """Compare B against baseline A on per-problem paired correctness.

    Both arms are keyed by problem id and MUST cover the same ids — the pairing
    is what makes the placebo delta and McNemar valid, so a mismatch is an
    explicit error, never a silent positional misalignment.
    """
    if set(correct_a) != set(correct_b):
        raise ValueError("both arms must cover the same problem ids")
    if not correct_a:
        raise ValueError("empty correctness")

    ids = sorted(correct_a)
    a = [bool(correct_a[i]) for i in ids]
    b = [bool(correct_b[i]) for i in ids]

    delta, ci_low, ci_high = paired_bootstrap_ci(a, b, seed=seed)
    p_value, n_discordant, test = mcnemar(a, b)
    return Comparison(
        label_a=label_a,
        label_b=label_b,
        n=len(ids),
        accuracy_a=float(np.mean(a)),
        accuracy_b=float(np.mean(b)),
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_discordant=n_discordant,
        test=test,
    )
