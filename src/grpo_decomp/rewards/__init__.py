"""Reward selection for a GRPO arm, resolved by name from the registry.

Rewards share one signature: they take the generated ``completions`` plus forwarded
dataset columns (e.g. ``gold_answer``) and return one score per completion. A study
registers its verifiable rewards (e.g. ``correct``) via
:func:`grpo_decomp.registries.register_reward`; switching arms is then a one-word config
change.

The harness itself provides exactly one reward, the **placebo** (``random``) — a
correctness-blind control, and the control the whole decomposition method leans on.
"""

from __future__ import annotations

from grpo_decomp.registries import PLACEBO_REWARD, REWARDS, RewardFn, register_reward
from grpo_decomp.rewards.placebo import make_random_reward


def get_reward(name: str, *, seed: int = 0) -> RewardFn:
    """Return the reward function for a training arm, selected by name.

    `seed` parameterizes stateful rewards (the ``random`` placebo holds a seeded RNG);
    verifiable rewards ignore it. Construct the reward **once per run** and reuse it, so
    the placebo's RNG sequence stays reproducible.
    """
    try:
        factory = REWARDS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown reward {name!r}; registered rewards are {tuple(sorted(REWARDS))}"
        ) from exc
    return factory(seed)


# The placebo is harness-provided (it is the study's confirmatory control).
register_reward(PLACEBO_REWARD, make_random_reward)


__all__ = ["PLACEBO_REWARD", "RewardFn", "get_reward", "make_random_reward", "register_reward"]
