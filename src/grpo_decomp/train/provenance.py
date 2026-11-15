"""Reproducible provenance recorded with every training run.

A run is reproducible only if the artifact records what produced it: base model +
revision, reward, dataset revision, seed, the full GRPO config, the code commit,
and the resolved dependency versions (TRL's GRPO behavior moves between versions,
so its pinned version is part of the record).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from grpo_decomp.provenance import (
    PROVENANCE_PACKAGES,
    RunProvenance,
    RunStatus,
    completed_checkpoint,
    content_digest,
    git_commit,
    git_is_dirty,
    load_run_provenance,
    package_versions,
)
from grpo_decomp.schemas import DatasetRef, Record, record_json
from grpo_decomp.train.config import ArmConfig


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


def write_run_record(path: Path, record: Record) -> None:
    """Replace a small lifecycle record atomically."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(record_json(record), encoding="utf-8")
    temporary.replace(path)


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
