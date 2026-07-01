"""Unbiased pass@k estimation (Chen et al., 2107.03374).

The biased ``1 - (1 - p)^k`` plug-in over-/under-states coverage; this uses the
unbiased estimator ``pass@k = 1 - C(n-c, k)/C(n, k)`` via the numerically stable
product form, which is the elicitation yardstick (base pass@k vs RL pass@1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased probability that >=1 of k samples is correct, given c correct of n.

    ``pass@k = 1 - C(n-c, k)/C(n, k)``, computed as ``1 - prod_{i}(1 - k/i)`` over
    ``i in [n-c+1, n]`` to avoid overflow in the binomial coefficients.
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def estimate_pass_at_k(num_correct: Sequence[int], k: int, *, n: int) -> float:
    """Mean pass@k over problems, each sampled `n` times with `num_correct[i]` correct."""
    if not num_correct:
        raise ValueError("num_correct is empty")
    return float(np.mean([pass_at_k(n, c, k) for c in num_correct]))
