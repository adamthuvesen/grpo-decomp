"""Dataset loaders that reduce every source to the canonical `grpo_decomp` `ProblemSet`.

These are the GSM8K-family and Countdown loaders for the reference study. Generic
sub-selection helpers (``dev_slice``, ``validation_split``) live in
:mod:`grpo_decomp.splits`.
"""

from llm_grpo_gains.data._hf_problem_sets import GoldAnswerError
from llm_grpo_gains.data.countdown import countdown_is_correct, load_countdown
from llm_grpo_gains.data.gsm8k import load_gsm8k
from llm_grpo_gains.data.gsm8k_platinum import load_gsm8k_platinum
from llm_grpo_gains.data.gsm_plus import load_gsm_plus
from llm_grpo_gains.data.gsm_symbolic import load_gsm_symbolic

__all__ = [
    "GoldAnswerError",
    "countdown_is_correct",
    "load_countdown",
    "load_gsm8k",
    "load_gsm8k_platinum",
    "load_gsm_plus",
    "load_gsm_symbolic",
]
