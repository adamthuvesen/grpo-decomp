"""Deterministic sub-selections over a `ProblemSet`: the dev subset and the
validation split.

Both are seeded and reproducible: the same `(problem_set, n, seed)` always yields
the same problem ids. Selection is index-based and the result is returned in
original order, so ids are stable across runs and platforms.
"""

from __future__ import annotations

import random

from llm_grpo_gains.schemas import ProblemSet


def _select(problem_set: ProblemSet, n: int, seed: int) -> list[int]:
    """Return `n` distinct indices into `problem_set`, sorted (stable order)."""
    total = len(problem_set)
    if not 0 <= n <= total:
        raise ValueError(f"cannot select {n} of {total} problems")
    return sorted(random.Random(seed).sample(range(total), n))


def _subset(problem_set: ProblemSet, indices: list[int]) -> ProblemSet:
    chosen = tuple(problem_set[i] for i in indices)
    return ProblemSet(source=problem_set.source, problems=chosen)


def dev_slice(problem_set: ProblemSet, *, n: int = 50, seed: int = 0) -> ProblemSet:
    """A small fixed subset for local pipeline testing without a GPU.

    Deterministic: the same `n` and `seed` return the same problem ids every time.
    """
    return _subset(problem_set, _select(problem_set, n, seed))


def validation_split(
    train: ProblemSet, *, n: int = 256, seed: int = 0
) -> tuple[ProblemSet, ProblemSet]:
    """Partition a training `ProblemSet` into ``(train_remainder, validation)``.

    The validation set is held out for in-training held-out accuracy and
    checkpoint selection; the remainder is what GRPO trains on. The two are
    disjoint by construction, and — being drawn from the train split — share no
    ids with the test split. `n` defaults to a 256-problem held-out eval set.
    """
    if not 0 < n < len(train):
        raise ValueError(
            f"validation_split needs 0 < n < {len(train)} for a non-empty partition, got n={n}"
        )
    held_out = set(_select(train, n, seed))
    remainder_idx = [i for i in range(len(train)) if i not in held_out]
    return _subset(train, remainder_idx), _subset(train, sorted(held_out))
