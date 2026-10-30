"""Reproducible provenance recorded with every training run.

A run is reproducible only if the artifact records what produced it: base model +
revision, reward, dataset revision, seed, the full GRPO config, the code commit,
and the resolved dependency versions (TRL's GRPO behavior moves between versions,
so its pinned version is part of the record).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from grpo_decomp.provenance import (
    PROVENANCE_PACKAGES,
    content_digest,
    git_commit,
    git_is_dirty,
    package_versions,
)
from grpo_decomp.schemas import DatasetRef, Record, record_json
from grpo_decomp.train.config import ArmConfig, GRPOSettings


class RunProvenance(Record):
    """Everything needed to reproduce one GRPO run."""

    attempt_id: str | None = None
    config_digest: str | None = None
    validation_identity: str | None = None
    arm: str
    base_model: str
    base_model_revision: str | None
    reward: str
    prompt_strategy: str
    dataset: DatasetRef
    train_size: int
    validation_size: int
    seed: int
    grpo: GRPOSettings
    commit: str
    dirty: bool = Field(default=False, description="Worktree had uncommitted changes at capture.")
    python_version: str
    package_versions: dict[str, str]


_RETIRED_CHECKPOINT_FIELDS = (
    "checkpoint_selection",
    "selected_checkpoint",
    "selected_step",
)


def load_run_provenance(run_dir: Path) -> RunProvenance:
    """Load run metadata, accepting checkpoint fields written by older releases.

    Checkpoint selection no longer affects evaluation. Existing Modal runs still carry
    its three provenance fields, so they are removed before the remaining strict schema
    is checked. Other unknown fields remain errors.
    """
    path = resolve_run_dir(run_dir) / "provenance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for field in _RETIRED_CHECKPOINT_FIELDS:
            data.pop(field, None)
    return RunProvenance.model_validate(data)


def capture_provenance(
    arm: ArmConfig,
    dataset: DatasetRef,
    *,
    train_size: int,
    validation_size: int,
    commit: str | None = None,
    dirty: bool | None = None,
    packages: Sequence[str] = PROVENANCE_PACKAGES,
    attempt_id: str | None = None,
    validation_identity: str | None = None,
) -> RunProvenance:
    """Capture a run's provenance: model, data, sizes, seed, config, commit, deps.

    `train_size` / `validation_size` are the realized split sizes (so a smoke
    subset is distinguishable from a full run, which otherwise share a DatasetRef).
    """
    return RunProvenance(
        attempt_id=attempt_id,
        config_digest=content_digest(arm.model_dump()),
        validation_identity=validation_identity,
        arm=arm.name,
        base_model=arm.base_model,
        base_model_revision=arm.base_model_revision,
        reward=arm.reward,
        prompt_strategy=arm.prompt_strategy,
        dataset=dataset,
        train_size=train_size,
        validation_size=validation_size,
        seed=arm.seed,
        grpo=arm.grpo,
        # Overrides win when given: on Modal the image strips .git, so the container
        # can't read git — the local entrypoint computes these and passes them in.
        commit=commit if commit is not None else git_commit(),
        dirty=dirty if dirty is not None else git_is_dirty(),
        python_version=sys.version.split()[0],
        package_versions=package_versions(packages),
    )


class RunStatus(Record):
    """Execution status for one attempt, independent of checkpoint directory presence."""

    attempt_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    state: Literal["running", "completed", "failed"]
    error: str | None = None


def write_run_record(path: Path, record: Record) -> None:
    """Replace a small lifecycle record atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(record_json(record), encoding="utf-8")
    temporary.replace(path)


def resolve_run_dir(run_dir: Path, *, completed: bool = True) -> Path:
    """Select one visible attempt without a shared mutable alias file.

    Each attempt owns its status. Multiple completed attempts require explicit
    selection. Directory order and finish time do not define scientific identity.
    """
    run_dir = Path(run_dir)
    attempts = run_dir / "attempts"
    if not attempts.is_dir():
        return run_dir
    state = "completed" if completed else "running"
    candidates = []
    for status_path in sorted(attempts.glob("*/status.json")):
        candidate = status_path.parent
        if not candidate.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError(f"attempt path escapes run directory: {candidate}")
        status = RunStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
        if status.state != state:
            continue
        if completed:
            completed_checkpoint(candidate)
        candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in candidates)
        raise ValueError(
            f"multiple {state} attempts under {run_dir}; select one explicitly: {choices}"
        )
    if completed and (run_dir / "provenance.json").is_file():
        return run_dir
    raise ValueError(f"no {state} attempt under {run_dir}")


def completed_checkpoint(run_dir: Path) -> Path:
    """Return the final checkpoint of a completed attempt, or an unverified legacy run.

    Legacy runs lack lifecycle records. Their provenance keeps ``attempt_id=None``,
    so consumers can distinguish a legacy path from verified completed execution.
    """
    resolved = resolve_run_dir(run_dir)
    provenance = load_run_provenance(resolved)
    status_path = resolved / "status.json"
    if provenance.attempt_id is not None or status_path.exists():
        if not status_path.is_file():
            raise ValueError(f"missing run status under {resolved}")
        status = RunStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
        if status.state != "completed" or status.attempt_id != provenance.attempt_id:
            raise ValueError(f"run attempt is not completed: {resolved}")
        if resolved.name != status.attempt_id or provenance.config_digest is None:
            raise ValueError(f"run attempt identity does not match provenance: {resolved}")
    checkpoint = resolved / "checkpoints" / "final"
    if not checkpoint.is_dir():
        raise ValueError(f"no final checkpoint under {resolved}")
    if not checkpoint.resolve().is_relative_to(resolved.resolve()):
        raise ValueError(f"checkpoint path escapes run directory: {checkpoint}")
    return checkpoint


def finish_run(run_dir: Path, *, error: str | None = None) -> None:
    """Record the attempt result and advance the completed alias only after success."""
    provenance = load_run_provenance(run_dir)
    if provenance.attempt_id is None:
        raise ValueError("cannot finish a legacy run without an attempt identity")
    previous = RunStatus.model_validate_json((run_dir / "status.json").read_text(encoding="utf-8"))
    if previous.state != "running" or previous.attempt_id != provenance.attempt_id:
        raise ValueError("only a running attempt can finish")
    checkpoint = run_dir / "checkpoints" / "final"
    if error is None and (
        not checkpoint.is_dir() or not checkpoint.resolve().is_relative_to(run_dir.resolve())
    ):
        raise ValueError("cannot complete an attempt without its final checkpoint")
    status = RunStatus(
        attempt_id=provenance.attempt_id,
        state="completed" if error is None else "failed",
        error=error,
    )
    write_run_record(run_dir / "status.json", status)
    if error is None:
        completed_checkpoint(run_dir)
