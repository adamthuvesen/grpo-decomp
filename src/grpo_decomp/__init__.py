"""grpo-decomp — a controls-first harness for decomposing GRPO gains.

Train a GRPO arm, sample a ``CompletionSet``, grade it, and decompose the benchmark gain
into real capability vs. elicitation / contamination / formatting / noise — for any model
and any task plugged in through :mod:`grpo_decomp.registries`. The reference study that
exercises this harness on GSM8K + Countdown lives in the separate ``llm_grpo_gains``
package.
"""

# Side-effect imports register the harness's built-ins on `import grpo_decomp`, without
# loading any study: the `r1_zero` prompt strategy and the `random` placebo reward.
import grpo_decomp.prompts
import grpo_decomp.rewards  # noqa: F401
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet

__version__ = "0.1.0"

__all__ = ["DatasetRef", "Problem", "ProblemSet", "__version__"]
