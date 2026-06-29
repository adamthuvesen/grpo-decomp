"""Verifiable GRPO reward functions sharing one signature, selected by name.

Every reward takes the generated `completions` plus forwarded dataset columns
(e.g. `gold_answer`) and returns one score per completion. Training arms differ only by
which reward `get_reward` returns:

- ``correct``   — verifiable exact-match correctness on math (the real signal).
- ``countdown`` — verifiable search correctness for the Countdown positive control.
- ``random``    — the seeded placebo control.

``format`` (a small answer-format bonus) is specified but deferred and deliberately not
selectable here: on the Qwen substrate a format reward is itself a confound.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_grpo_gains.rewards.correct import correct
from llm_grpo_gains.rewards.countdown import countdown
from llm_grpo_gains.rewards.placebo import make_random_reward

#: Config value for the correctness-blind placebo reward. Kept as "random" for
#: backward-compatible arm YAMLs; named here so callers do not treat it as a placeholder baseline.
PLACEBO_REWARD = "random"

#: Shared reward signature: completions + forwarded columns -> one score per completion.
RewardFn = Callable[..., list[float]]

#: The rewards a training arm may select. `format` is deferred (see module docstring).
SELECTABLE = ("correct", "countdown", PLACEBO_REWARD)


def get_reward(name: str, *, seed: int = 0) -> RewardFn:
    """Return the reward function for a training arm, selected by name.

    Switching arms is a one-word config change. `seed` parameterizes the
    `random` placebo (ignored by the verifiable rewards).

    Construct the reward **once per run** and reuse it: the `random` placebo holds
    a stateful RNG, so re-creating it each batch would restart the sequence and
    break reproducibility.
    """
    if name == "correct":
        return correct
    if name == "countdown":
        return countdown
    if name == PLACEBO_REWARD:
        return make_random_reward(seed)
    raise ValueError(f"unknown reward {name!r}; selectable rewards are {SELECTABLE}")


__all__ = [
    "PLACEBO_REWARD",
    "SELECTABLE",
    "RewardFn",
    "correct",
    "countdown",
    "get_reward",
    "make_random_reward",
]
