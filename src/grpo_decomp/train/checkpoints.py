"""Resolve trained checkpoint paths from recorded selection policy."""

from __future__ import annotations

from pathlib import Path


def _checkpoint_path(run_dir: Path, selected_checkpoint: str) -> str:
    return str(Path(run_dir) / "checkpoints" / selected_checkpoint)


def require_selected_checkpoint_path(run_dir: Path, selected_checkpoint: str | None) -> str:
    """Return the pre-selected checkpoint path, or fail before evaluation can peek."""
    if selected_checkpoint is None:
        raise ValueError(
            f"{run_dir} has no selected checkpoint; run "
            "`modal run modal_app.py --command heldout` for this arm before evaluation"
        )
    return _checkpoint_path(run_dir, selected_checkpoint)


def final_or_selected_checkpoint_path(
    run_dir: Path, selected_checkpoint: str | None, rule: str
) -> str:
    """Prefer a recorded selection, otherwise realize the deterministic final rule."""
    if selected_checkpoint is None:
        if rule != "final":
            raise ValueError(
                f"{run_dir} has no selected checkpoint and rule is {rule!r}; run "
                "`modal run modal_app.py --command heldout` for this arm first"
            )
        selected_checkpoint = "final"
    return _checkpoint_path(run_dir, selected_checkpoint)
