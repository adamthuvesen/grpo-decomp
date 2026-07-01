"""Reproducible provenance recorded with every training run.

A run is reproducible only if the artifact records what produced it: base model +
revision, reward, dataset revision, seed, the full GRPO config, the code commit,
and the resolved dependency versions (TRL's GRPO behavior moves between versions,
so its pinned version is part of the record).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from pydantic import Field

from grpo_decomp.provenance import (
    PROVENANCE_PACKAGES,
    git_commit,
    git_is_dirty,
    package_versions,
)
from grpo_decomp.schemas import DatasetRef, Record
from grpo_decomp.train.config import ArmConfig, GRPOSettings


class RunProvenance(Record):
    """Everything needed to reproduce one GRPO run."""

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
    checkpoint_selection: str
    selected_step: int | None
    selected_checkpoint: str | None = Field(
        default=None, description="Realized checkpoint dir (e.g. 'final' or 'checkpoint-300')."
    )
    commit: str
    dirty: bool = Field(default=False, description="Worktree had uncommitted changes at capture.")
    python_version: str
    package_versions: dict[str, str]


def capture_provenance(
    arm: ArmConfig,
    dataset: DatasetRef,
    *,
    train_size: int,
    validation_size: int,
    selected_step: int | None = None,
    selected_checkpoint: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    packages: Sequence[str] = PROVENANCE_PACKAGES,
) -> RunProvenance:
    """Capture a run's provenance: model, data, sizes, seed, config, commit, deps.

    `train_size` / `validation_size` are the realized split sizes (so a smoke
    subset is distinguishable from a full run, which otherwise share a DatasetRef).
    `selected_step` is the checkpoint step chosen per `arm.checkpoint_selection`;
    it is filled in by the GPU run and left None at launch time.
    """
    return RunProvenance(
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
        checkpoint_selection=arm.checkpoint_selection,
        selected_step=selected_step,
        selected_checkpoint=selected_checkpoint,
        # Overrides win when given: on Modal the image strips .git, so the container
        # can't read git — the local entrypoint computes these and passes them in.
        commit=commit if commit is not None else git_commit(),
        dirty=dirty if dirty is not None else git_is_dirty(),
        python_version=sys.version.split()[0],
        package_versions=package_versions(packages),
    )
