"""Shared provenance capture: the code commit and the dependency versions.

A result artifact — a training run or a generation pass — is reproducible only if
it records what produced it. The git commit and the resolved versions of the
behavior-pinning dependencies are common to both, so they live here.

The run record and its attempt lookup also live here, because training writes them
while evaluation and the registries read them. ``train.provenance`` captures and
finishes runs; the generation record shape stays in ``eval.completions``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Literal

from pydantic import Field

from grpo_decomp.schemas import DatasetRef, GRPOSettings, ProblemSet, Record

#: Dependencies whose versions pin a result's behavior (training or generation).
PROVENANCE_PACKAGES = (
    "grpo-decomp",
    "trl",
    "transformers",
    "torch",
    "vllm",
    "datasets",
    "math-verify",
    "eval-audit",
    "numpy",
    "scipy",
    "polars",
)


def content_digest(value: object) -> str:
    """SHA256 of canonical JSON, including ordered problem and prompt content."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def problem_set_digest(problems: ProblemSet) -> str:
    """Identify the dataset metadata and actual ordered problem records."""
    return content_digest(problems.model_dump())


def file_digest(path: Path) -> str:
    """Hash a file without loading model weights into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_commit() -> str:
    """The current HEAD SHA, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def git_is_dirty() -> bool:
    """True if the worktree has uncommitted changes.

    Modal ships the working tree (not HEAD) into the image, so a dirty tree means the
    recorded `git_commit()` did not actually produce the result — provenance records
    this so a dirty-tree artifact is never mistaken for a reproducible one.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(result.stdout.strip())


def package_versions(names: Sequence[str]) -> dict[str, str]:
    """Resolve installed versions for `names`, marking absent packages ``"absent"``."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


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


class RunStatus(Record):
    """Execution status for one attempt, independent of checkpoint directory presence."""

    attempt_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    state: Literal["running", "completed", "failed"]
    error: str | None = None


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
