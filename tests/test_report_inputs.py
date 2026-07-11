"""Unit tests for report input discovery and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from completion_set_fixtures import problem_set, write_completion_set_dir

from grpo_decomp.eval.completions import load_completion_set
from grpo_decomp.registries import EVAL_SETS
from grpo_decomp.report.inputs import (
    base_and_correct_seeds,
    seed_label,
    validate_aligned_artifacts,
)


def test_validate_aligned_artifacts_accepts_matching_numeric_suffix_order(tmp_path) -> None:
    write_completion_set_dir(
        tmp_path / "base__mini", model="base", boxed="4", ids=("p1", "p2", "p10")
    )
    write_completion_set_dir(
        tmp_path / "correct__mini", model="correct", boxed="4", ids=("p1", "p2", "p10")
    )
    validate_aligned_artifacts(
        "mini",
        {
            "base": load_completion_set(tmp_path / "base__mini"),
            "correct": load_completion_set(tmp_path / "correct__mini"),
        },
    )


def test_base_and_correct_seeds_validates_registered_problem_records(tmp_path, monkeypatch) -> None:
    expected = problem_set(ids=("p1", "p2"))
    monkeypatch.setitem(EVAL_SETS, "mini", lambda: expected)
    write_completion_set_dir(tmp_path / "base__mini", model="base", boxed="4", ids=("p1", "p2"))
    write_completion_set_dir(
        tmp_path / "correct-seed0__mini", model="correct", boxed="4", ids=("p1", "p3")
    )

    with pytest.raises(ValueError, match="problem records"):
        base_and_correct_seeds(tmp_path, "mini")


def test_base_and_correct_seeds_accepts_limited_artifact_subset(tmp_path, monkeypatch) -> None:
    expected = problem_set(ids=("p1", "p2", "p3"))
    monkeypatch.setitem(EVAL_SETS, "mini", lambda: expected)
    write_completion_set_dir(tmp_path / "base__mini", model="base", boxed="4", ids=("p1", "p2"))
    write_completion_set_dir(
        tmp_path / "correct-seed0__mini", model="correct", boxed="4", ids=("p1", "p2")
    )

    base, correct_by_seed = base_and_correct_seeds(tmp_path, "mini")

    assert [item.problem.id for item in base.items] == ["p1", "p2"]
    assert [seed for seed, _completion_set in correct_by_seed] == [0]


def test_base_and_correct_seeds_rejects_mismatched_correct_sample_counts(
    tmp_path, monkeypatch
) -> None:
    expected = problem_set(ids=("p1", "p2"))
    monkeypatch.setitem(EVAL_SETS, "mini", lambda: expected)
    write_completion_set_dir(
        tmp_path / "base__mini", model="base", boxed="4", ids=("p1", "p2"), n=2, temperature=0.7
    )
    write_completion_set_dir(
        tmp_path / "correct-seed0__mini",
        model="correct",
        boxed="4",
        ids=("p1", "p2"),
        n=2,
        temperature=0.7,
    )
    write_completion_set_dir(
        tmp_path / "correct-seed1__mini",
        model="correct",
        boxed="4",
        ids=("p1", "p2"),
        n=3,
        temperature=0.7,
    )

    with pytest.raises(ValueError, match=r"share sampling\.n"):
        base_and_correct_seeds(tmp_path, "mini")


def test_seed_label_from_battery_dir() -> None:
    assert seed_label(Path("battery")) == 0
    assert seed_label(Path("battery-seed2")) == 2
