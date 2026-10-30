"""Shared provenance capture: the code commit and the dependency versions.

A result artifact — a training run or a generation pass — is reproducible only if
it records what produced it. The git commit and the resolved versions of the
behavior-pinning dependencies are common to both, so they live here; the
record *shapes* (run vs. generation) are defined by their own modules.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from grpo_decomp.schemas import ProblemSet

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
