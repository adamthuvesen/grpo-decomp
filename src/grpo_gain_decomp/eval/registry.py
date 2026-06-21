"""Named evaluation sets and report labels shared by CLI and Modal entrypoints."""

from __future__ import annotations

from collections.abc import Callable

from grpo_gain_decomp.data import (
    dev_slice,
    load_countdown,
    load_gsm8k,
    load_gsm8k_platinum,
    load_gsm_plus,
    load_gsm_symbolic,
)
from grpo_gain_decomp.schemas import ProblemSet

#: The named problem sets `generate --set` can target.
SETS: dict[str, Callable[[], ProblemSet]] = {
    "gsm8k-test": lambda: load_gsm8k("test"),
    "gsm8k-train": lambda: load_gsm8k("train"),
    "dev": lambda: dev_slice(load_gsm8k("test")),
    "gsm-symbolic": lambda: load_gsm_symbolic("main"),
    "gsm-plus": lambda: load_gsm_plus("test"),
    "gsm8k-platinum": load_gsm8k_platinum,
    "countdown-test": lambda: load_countdown("test"),
    "countdown-dev": lambda: load_countdown("dev"),
}

#: The arms a decomposition compares; `correct` is compared against `base` and `random`.
ARMS = ("base", "correct", "random")

#: What each control set actually probes (the report's controlled row labels).
PROBES = {
    "gsm-symbolic": "memorization (templated renumbering)",
    "gsm8k-platinum": "label noise (cleaned labels)",
    "gsm-plus": "robustness (adversarial perturbation)",
}
