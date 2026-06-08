"""Paired statistics for the decomposition: McNemar, bootstrap CIs, comparisons.

The bootstrap CI is delegated to eval-audit; McNemar (with an exact-binomial
fallback) and the comparison assembly are local.
"""

from grpo_gain_decomp.stats.bootstrap import paired_bootstrap_ci
from grpo_gain_decomp.stats.compare import Comparison, compare
from grpo_gain_decomp.stats.significance import mcnemar

__all__ = ["Comparison", "compare", "mcnemar", "paired_bootstrap_ci"]
