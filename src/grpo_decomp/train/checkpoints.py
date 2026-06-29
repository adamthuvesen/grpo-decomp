"""Resolve trained checkpoint paths from recorded selection policy."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class CheckpointResolutionPolicy(StrEnum):
    """How to handle a run whose held-out checkpoint selection is not recorded."""

    REQUIRE_SELECTED = "require_selected"
    ALLOW_FINAL_RULE = "allow_final_rule"


def resolve_checkpoint_path(
    run_dir: Path,
    selected_checkpoint: str | None,
    rule: str | None,
    *,
    policy: CheckpointResolutionPolicy,
) -> str:
    """Resolve the checkpoint path to evaluate for a finished training run."""
    run_dir = Path(run_dir)
    if selected_checkpoint is None:
        if policy == CheckpointResolutionPolicy.REQUIRE_SELECTED:
            raise ValueError(
                f"{run_dir} has no selected checkpoint; run "
                "`modal run modal_app.py --command heldout` for this arm before evaluation"
            )
        if rule != "final":
            raise ValueError(
                f"{run_dir} has no selected checkpoint and rule is {rule!r}; run "
                "`modal run modal_app.py --command heldout` for this arm first"
            )
        selected_checkpoint = "final"
    return str(run_dir / "checkpoints" / selected_checkpoint)


def require_selected_checkpoint_path(run_dir: Path, selected_checkpoint: str | None) -> str:
    """Return the pre-selected checkpoint path, or fail before evaluation can peek."""
    return resolve_checkpoint_path(
        run_dir,
        selected_checkpoint,
        None,
        policy=CheckpointResolutionPolicy.REQUIRE_SELECTED,
    )


def final_or_selected_checkpoint_path(
    run_dir: Path, selected_checkpoint: str | None, rule: str
) -> str:
    """Prefer a recorded selection, otherwise realize the deterministic final rule."""
    return resolve_checkpoint_path(
        run_dir,
        selected_checkpoint,
        rule,
        policy=CheckpointResolutionPolicy.ALLOW_FINAL_RULE,
    )
