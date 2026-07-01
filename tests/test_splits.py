"""Unit tests for deterministic dev subset and validation split (no network)."""

from __future__ import annotations

import pytest

from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from grpo_decomp.splits import dev_slice, validation_split


def _problem_set(n: int, split: str = "train") -> ProblemSet:
    ref = DatasetRef(name="openai/gsm8k", config="main", split=split, revision="rev")
    problems = tuple(
        Problem(id=f"gsm8k/main/{split}/{i}", question=f"q{i}", gold_answer=str(i))
        for i in range(n)
    )
    return ProblemSet(source=ref, problems=problems)


def test_dev_slice_size_and_membership() -> None:
    full = _problem_set(500)
    slice_ = dev_slice(full, n=50, seed=0)
    assert len(slice_) == 50
    full_ids = {p.id for p in full}
    assert {p.id for p in slice_} <= full_ids


def test_dev_slice_is_deterministic() -> None:
    full = _problem_set(500)
    assert [p.id for p in dev_slice(full, seed=0)] == [p.id for p in dev_slice(full, seed=0)]


def test_dev_slice_rejects_oversize_n() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        dev_slice(_problem_set(10), n=50)


def test_validation_split_partitions_disjointly() -> None:
    train = _problem_set(1000)
    remainder, validation = validation_split(train, n=256, seed=0)

    assert len(validation) == 256
    assert len(remainder) == 1000 - 256

    val_ids = {p.id for p in validation}
    rem_ids = {p.id for p in remainder}
    assert val_ids.isdisjoint(rem_ids)
    assert val_ids | rem_ids == {p.id for p in train}


def test_validation_is_drawn_from_train_and_disjoint_from_test() -> None:
    train = _problem_set(1000, split="train")
    test = _problem_set(500, split="test")
    _, validation = validation_split(train, n=128, seed=0)
    val_ids = {p.id for p in validation}
    # Substantive guard: validation is a real subset of train (would fail on a leak),
    # which makes disjointness-from-test follow rather than being true by id prefix.
    assert val_ids <= {p.id for p in train}
    assert val_ids.isdisjoint({p.id for p in test})


def test_validation_split_is_deterministic() -> None:
    train = _problem_set(1000)
    _, val_a = validation_split(train, n=128, seed=0)
    _, val_b = validation_split(train, n=128, seed=0)
    assert [p.id for p in val_a] == [p.id for p in val_b]


def test_selection_responds_to_seed() -> None:
    # Guards against a regression that ignores the seed param (e.g. hardcoded RNG):
    # different seeds must select different problems.
    full = _problem_set(500)
    assert [p.id for p in dev_slice(full, seed=0)] != [p.id for p in dev_slice(full, seed=1)]

    train = _problem_set(1000)
    _, val0 = validation_split(train, n=128, seed=0)
    _, val1 = validation_split(train, n=128, seed=1)
    assert [p.id for p in val0] != [p.id for p in val1]


def test_validation_split_rejects_degenerate_n() -> None:
    train = _problem_set(100)
    with pytest.raises(ValueError, match="non-empty partition"):
        validation_split(train, n=100)  # would leave an empty training remainder
    with pytest.raises(ValueError, match="non-empty partition"):
        validation_split(train, n=0)
