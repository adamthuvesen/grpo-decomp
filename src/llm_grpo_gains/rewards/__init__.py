"""The study's verifiable rewards (the real signals the placebo is compared against).

- ``correct``   — verifiable exact-match correctness on math (the GSM8K signal).
- ``countdown`` — verifiable search correctness for the Countdown positive control.

Both are registered into the harness reward registry by
:func:`llm_grpo_gains.registration.register`. The correctness-blind ``random`` placebo is
provided by the harness (:mod:`grpo_decomp.rewards`), not here.
"""

from llm_grpo_gains.rewards.correct import correct
from llm_grpo_gains.rewards.countdown import countdown

__all__ = ["correct", "countdown"]
