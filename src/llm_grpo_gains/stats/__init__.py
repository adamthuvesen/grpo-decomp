"""Paired statistics for the decomposition: McNemar, bootstrap CIs, comparisons.

The bootstrap CI is delegated to eval-audit; McNemar (with an exact-binomial
fallback) and the comparison assembly are local.
"""

from llm_grpo_gains.stats.bootstrap import paired_bootstrap_ci
from llm_grpo_gains.stats.compare import Comparison, compare
from llm_grpo_gains.stats.significance import mcnemar

__all__ = ["Comparison", "compare", "mcnemar", "paired_bootstrap_ci"]
