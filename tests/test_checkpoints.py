"""Unit tests for checkpoint-path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from grpo_decomp.train.checkpoints import (
    final_or_selected_checkpoint_path,
    require_selected_checkpoint_path,
)


def test_selected_checkpoint_path_requires_heldout() -> None:
    run_dir = Path("/runs/correct-seed0")

    with pytest.raises(ValueError, match="heldout"):
        require_selected_checkpoint_path(run_dir, None)

    assert (
        require_selected_checkpoint_path(run_dir, "final")
        == "/runs/correct-seed0/checkpoints/final"
    )


def test_final_or_selected_checkpoint_realizes_the_final_rule() -> None:
    run_dir = Path("/runs/correct-seed3")

    assert (
        final_or_selected_checkpoint_path(run_dir, "checkpoint-400", "best_on_validation")
        == "/runs/correct-seed3/checkpoints/checkpoint-400"
    )
    assert (
        final_or_selected_checkpoint_path(run_dir, None, "final")
        == "/runs/correct-seed3/checkpoints/final"
    )
    with pytest.raises(ValueError, match="heldout"):
        final_or_selected_checkpoint_path(run_dir, None, "best_on_validation")
