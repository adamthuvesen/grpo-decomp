"""The `random` (placebo) reward: a correctness-blind control."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grpo_decomp.registries import RewardFn


def make_random_reward(seed: int) -> RewardFn:
    """Build the placebo reward: a uniform value in ``[0, 1)`` per completion,
    independent of correctness, from a seeded RNG so a run is reproducible.

    The project's signature result — *"random rewards still improve Qwen"* —
    depends on this being genuinely correctness-blind, so it reads neither the
    completion nor the gold. The RNG is seeded per run and the seed recorded in
    provenance, so the reward sequence reproduces exactly.
    """
    rng = random.Random(seed)

    def random_reward(completions: Sequence[str], **kwargs: object) -> list[float]:
        return [rng.random() for _ in completions]

    return random_reward
