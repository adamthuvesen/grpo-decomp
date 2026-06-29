"""Named evaluation sets and report labels shared by CLI and Modal entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from llm_grpo_gains.data import (
    dev_slice,
    load_countdown,
    load_gsm8k,
    load_gsm8k_platinum,
    load_gsm_plus,
    load_gsm_symbolic,
)
from llm_grpo_gains.schemas import ProblemSet

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

#: Perturbation / clean-label control sets for the GSM8K decomposition (not Countdown).
CONTROL_SETS: tuple[str, ...] = ("gsm-symbolic", "gsm-plus", "gsm8k-platinum")

#: The arms a decomposition compares; `correct` is compared against `base` and `random`.
ARMS = ("base", "correct", "random")

#: What each control set actually probes (the report's controlled row labels).
PROBES = {
    "gsm-symbolic": "memorization (templated renumbering)",
    "gsm8k-platinum": "label noise (cleaned labels)",
    "gsm-plus": "robustness (adversarial perturbation)",
}


@dataclass(frozen=True)
class EvalTaskProfile:
    """Modal/eval wiring for one study task."""

    base_config: str
    task_set: str
    control_sets: tuple[str, ...]
    run_prefix: str
    battery_root: str
    elicitation_root: str
    passk_multiseed_root: str


TASKS: dict[str, EvalTaskProfile] = {
    "gsm8k": EvalTaskProfile(
        base_config="configs/correct.yaml",
        task_set="gsm8k-test",
        control_sets=CONTROL_SETS,
        run_prefix="",
        battery_root="battery",
        elicitation_root="elicitation",
        passk_multiseed_root="passk-multiseed",
    ),
    "countdown": EvalTaskProfile(
        base_config="configs/countdown-correct.yaml",
        task_set="countdown-test",
        control_sets=(),
        run_prefix="countdown-",
        battery_root="battery-countdown",
        elicitation_root="elicitation-countdown",
        passk_multiseed_root="passk-multiseed-countdown",
    ),
}


def get_task_profile(task: str) -> EvalTaskProfile:
    """Return the eval wiring for a named study task."""
    try:
        return TASKS[task]
    except KeyError as exc:
        raise ValueError(f"eval task must be one of {tuple(sorted(TASKS))}, got {task!r}") from exc
