"""Unit tests for report input discovery and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import dataset_ref, write_completion_set_dir

from grpo_decomp.eval.report_inputs import (
    discover_completion_sets,
    seed_label,
    validate_report_artifacts,
)


def test_discover_completion_sets_groups_by_set_and_arm(tmp_path) -> None:
    ref = dataset_ref()
    write_completion_set_dir(tmp_path / "base__gsm8k-test", model="base", boxed="4", ref=ref)
    write_completion_set_dir(tmp_path / "correct__gsm8k-test", model="correct", boxed="4", ref=ref)
    write_completion_set_dir(tmp_path / "random__gsm8k-test", model="random", boxed="4", ref=ref)

    grouped = discover_completion_sets(tmp_path)
    assert set(grouped["gsm8k-test"]) == {"base", "correct", "random"}


def test_validate_report_artifacts_rejects_unknown_set(tmp_path, monkeypatch) -> None:
    ref = dataset_ref()
    write_completion_set_dir(tmp_path / "base__unknown-set", model="base", boxed="4", ref=ref)
    grouped = discover_completion_sets(tmp_path)
    with pytest.raises(ValueError, match="unknown report set"):
        validate_report_artifacts(grouped)


def test_validate_report_artifacts_rejects_mixed_prompt_strategies(tmp_path) -> None:
    ref = dataset_ref()
    write_completion_set_dir(
        tmp_path / "base__gsm8k-test", model="base", boxed="4", ref=ref, prompt_strategy="r1_zero"
    )
    write_completion_set_dir(
        tmp_path / "correct__gsm8k-test",
        model="correct",
        boxed="4",
        ref=ref,
        prompt_strategy="chat_template",
    )
    grouped = discover_completion_sets(tmp_path)
    with pytest.raises(ValueError, match="different prompt strategies"):
        validate_report_artifacts(grouped)


def test_seed_label_from_battery_dir() -> None:
    assert seed_label(Path("battery")) == 0
    assert seed_label(Path("battery-seed2")) == 2
