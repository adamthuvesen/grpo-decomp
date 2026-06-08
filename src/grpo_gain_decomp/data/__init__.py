"""Dataset loaders that reduce every source to grpo_gain_decomp's canonical `ProblemSet`."""

from grpo_gain_decomp.data._common import GoldAnswerError
from grpo_gain_decomp.data.countdown import countdown_is_correct, load_countdown
from grpo_gain_decomp.data.gsm8k import load_gsm8k
from grpo_gain_decomp.data.gsm8k_platinum import load_gsm8k_platinum
from grpo_gain_decomp.data.gsm_plus import load_gsm_plus
from grpo_gain_decomp.data.gsm_symbolic import load_gsm_symbolic
from grpo_gain_decomp.data.splits import dev_slice, validation_split

__all__ = [
    "GoldAnswerError",
    "countdown_is_correct",
    "dev_slice",
    "load_countdown",
    "load_gsm8k",
    "load_gsm8k_platinum",
    "load_gsm_plus",
    "load_gsm_symbolic",
    "validation_split",
]
