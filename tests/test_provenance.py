"""Unit tests for run provenance capture (no network, no GPU)."""

from __future__ import annotations

from grpo_decomp.provenance import package_versions
from grpo_decomp.schemas import DatasetRef
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.provenance import capture_provenance


def _arm() -> ArmConfig:
    return ArmConfig(name="correct", base_model="Qwen/Qwen2.5-Math-1.5B", reward="correct", seed=7)


def _dataset() -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split="train", revision="abc123")


def test_package_versions_marks_missing_as_absent() -> None:
    versions = package_versions(["grpo-decomp", "definitely-not-a-real-package-xyz"])
    assert versions["grpo-decomp"]  # an installed version string
    assert versions["definitely-not-a-real-package-xyz"] == "absent"


def test_capture_records_run_identity() -> None:
    prov = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    assert prov.arm == "correct"
    assert prov.reward == "correct"
    assert prov.seed == 7
    assert prov.train_size == 7217
    assert prov.validation_size == 256
    assert prov.checkpoint_selection == "final"  # pre-registered default
    assert prov.selected_step is None  # filled in by the held-out selection
    assert prov.selected_checkpoint is None  # filled in by the held-out selection
    assert prov.dataset.revision == "abc123"
    assert prov.grpo.beta == 0.0


def test_capture_records_environment() -> None:
    prov = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    assert prov.python_version.count(".") == 2  # e.g. "3.12.7"
    assert prov.commit == "unknown" or len(prov.commit) == 40  # git SHA or sentinel
    assert prov.package_versions["grpo-decomp"]
    # TRL is not installed in the CPU dev env -> recorded as absent, not crashing.
    assert prov.package_versions["trl"] == "absent"
    assert isinstance(prov.dirty, bool)  # worktree cleanliness recorded


def test_capture_uses_explicit_commit_and_dirty_overrides() -> None:
    # On Modal the container has no .git, so the local entrypoint passes these in.
    prov = capture_provenance(
        _arm(), _dataset(), train_size=7217, validation_size=256, commit="deadbeef", dirty=True
    )
    assert prov.commit == "deadbeef"
    assert prov.dirty is True
