"""McNemar's paired significance test for two models on a shared test set.

eval-audit provides the bootstrap CI but not a paired *significance* test, so this
is local: the exact two-sided binomial on the discordant pairs when they are few
(< 25, where the chi-square approximation is unreliable), and the chi-square test
with continuity correction otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest, chi2

_EXACT_THRESHOLD = 25


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[float, int, str]:
    """Return ``(p_value, n_discordant, test)`` for paired correctness on a shared set.

    Pairs are aligned by position. `test` is ``"exact-binomial"`` when the
    discordant count is below 25, else ``"chi2"``.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors must align, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("empty correctness vectors")

    a_only = int(np.sum(a & ~b))  # A correct, B wrong
    b_only = int(np.sum(~a & b))  # A wrong, B correct
    discordant = a_only + b_only

    if discordant == 0:
        return 1.0, 0, "exact-binomial"
    if discordant < _EXACT_THRESHOLD:
        p = binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue
        return float(p), discordant, "exact-binomial"
    stat = (abs(a_only - b_only) - 1) ** 2 / discordant
    return float(chi2.sf(stat, df=1)), discordant, "chi2"
