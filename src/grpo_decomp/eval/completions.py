"""The completion artifact: the boundary between generation (GPU) and analysis (CPU).

`generate` (on whatever box holds the model) writes a `CompletionSet`; CPU-only
report commands read it and feed the battery and stats. The artifact is
self-contained — it carries each `Problem` (so gold is
present for grading) plus its sampled completions and a provenance record — so the
analysis side needs no network and no dataset re-pinning.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, StrictBool, StrictFloat, StrictInt, ValidationError, model_validator

from grpo_decomp.prompts import EVAL_MAX_NEW_TOKENS
from grpo_decomp.provenance import (
    PROVENANCE_PACKAGES,
    file_digest,
    git_commit,
    git_is_dirty,
    package_versions,
    problem_set_digest,
)
from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet, Record, record_json

PROVENANCE_FILE = "provenance.json"
COMPLETIONS_FILE = "completions.jsonl"
MANIFEST_FILE = "manifest.json"


class SamplingConfig(Record):
    """How completions are drawn. Validated strictly: a bad knob is an explicit error."""

    temperature: StrictFloat = Field(default=0.0, ge=0.0, description="0 = greedy (pass@1).")
    top_p: StrictFloat = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: StrictInt = Field(default=EVAL_MAX_NEW_TOKENS, gt=0)
    n: StrictInt = Field(default=1, gt=0, description="Completions sampled per problem (uniform).")
    seed: StrictInt = 0


class GenerationProvenance(Record):
    """Everything needed to reproduce one generation pass."""

    model: str = Field(description="Model id or checkpoint path.")
    model_revision: str | None
    backend: str = Field(description="'transformers' or 'vllm'.")
    device: str | None = Field(default=None, description="Device the generator ran on.")
    resolved_model_revision: str | None = None
    training_run_id: str | None = None
    training_base_model: str | None = None
    training_base_model_revision: str | None = None
    training_reward: str | None = None
    training_protocol_digest: str | None = None
    training_seed: StrictInt | None = None
    training_config_digest: str | None = None
    checkpoint_digest: str | None = None
    prompt_digest: str | None = None
    dataset_digest: str | None = None
    grader_identity: str | None = None
    prompt_strategy: str = Field(
        default=DEFAULT_PROMPT_STRATEGY,
        description="Prompt strategy used to build prompts (must match training).",
    )
    sampling: SamplingConfig
    dataset: DatasetRef
    n_problems: StrictInt = Field(ge=0)
    commit: str
    dirty: StrictBool = Field(
        default=False, description="Worktree had uncommitted changes at capture."
    )
    python_version: str
    package_versions: dict[str, str]


class ProblemCompletions(Record):
    """One problem and the completions sampled for it."""

    problem: Problem
    samples: tuple[str, ...]


class CompletionSet(Record):
    """A generation pass: provenance plus per-problem completions, uniform in n."""

    provenance: GenerationProvenance
    items: tuple[ProblemCompletions, ...]

    @model_validator(mode="after")
    def _check_uniform_and_consistent(self) -> CompletionSet:
        n = self.provenance.sampling.n
        if not self.items:
            raise ValueError("completion set is empty (no problems); generation produced nothing")
        if len(self.items) != self.provenance.n_problems:
            raise ValueError(
                f"n_problems={self.provenance.n_problems} but found {len(self.items)} items"
            )
        ids = [item.problem.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate problem ids in completion set")
        bad = {item.problem.id: len(item.samples) for item in self.items if len(item.samples) != n}
        if bad:
            sample = dict(list(bad.items())[:3])
            raise ValueError(f"non-uniform completion counts (expected n={n}): {sample}")
        if (
            self.provenance.dataset_digest is not None
            and self.provenance.dataset_digest != problem_set_digest(self.problem_set())
        ):
            raise ValueError("dataset content digest mismatch")
        return self

    def problem_set(self) -> ProblemSet:
        """Reconstruct the `ProblemSet` (for `run_battery` / `grade`)."""
        return ProblemSet(
            source=self.provenance.dataset, problems=tuple(item.problem for item in self.items)
        )

    def completions_by_id(self) -> dict[str, tuple[str, ...]]:
        """`problem id -> its samples`, the battery's input shape."""
        return {item.problem.id: item.samples for item in self.items}


def capture_generation_provenance(
    *,
    model: str,
    dataset: DatasetRef,
    sampling: SamplingConfig,
    backend: str,
    n_problems: int,
    model_revision: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    commit: str | None = None,
    dirty: bool | None = None,
    packages: Sequence[str] = PROVENANCE_PACKAGES,
    device: str | None = None,
    resolved_model_revision: str | None = None,
    training_run_id: str | None = None,
    training_base_model: str | None = None,
    training_base_model_revision: str | None = None,
    training_reward: str | None = None,
    training_protocol_digest: str | None = None,
    training_seed: int | None = None,
    training_config_digest: str | None = None,
    checkpoint_digest: str | None = None,
    prompt_digest: str | None = None,
    dataset_digest: str | None = None,
    grader_identity: str | None = None,
) -> GenerationProvenance:
    """Capture a generation pass's provenance: model, data, sampling, commit, deps.

    `commit`/`dirty` override the git-derived values when given — on Modal the image
    strips `.git`, so the local entrypoint computes them and passes them in (the same
    path training uses), keeping every result artifact traceable to its code.
    """
    return GenerationProvenance(
        model=model,
        model_revision=model_revision,
        backend=backend,
        device=device,
        resolved_model_revision=resolved_model_revision,
        training_run_id=training_run_id,
        training_base_model=training_base_model,
        training_base_model_revision=training_base_model_revision,
        training_reward=training_reward,
        training_protocol_digest=training_protocol_digest,
        training_seed=training_seed,
        training_config_digest=training_config_digest,
        checkpoint_digest=checkpoint_digest,
        prompt_digest=prompt_digest,
        dataset_digest=dataset_digest,
        grader_identity=grader_identity,
        prompt_strategy=prompt_strategy,
        sampling=sampling,
        dataset=dataset,
        n_problems=n_problems,
        commit=commit if commit is not None else git_commit(),
        dirty=dirty if dirty is not None else git_is_dirty(),
        python_version=sys.version.split()[0],
        package_versions=package_versions(packages),
    )


class CompletionManifest(Record):
    """Version and integrity hashes for a newly published CompletionSet."""

    schema_version: StrictInt = 1
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_version(self) -> CompletionManifest:
        if self.schema_version != 1:
            raise ValueError(f"unsupported CompletionSet schema version {self.schema_version}")
        return self


def write_completion_set(completion_set: CompletionSet, out_dir: Path) -> Path:
    """Publish an immutable, versioned directory through one atomic rename.

    Equal retries preserve existing bytes, including legacy unversioned artifacts.
    Conflicting output and damaged existing artifacts fail without replacing either.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        _check_existing(completion_set, out_dir)
        return out_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        provenance_bytes = record_json(completion_set.provenance).encode("utf-8")
        lines = [
            json.dumps(item.model_dump(), sort_keys=True, allow_nan=False)
            for item in completion_set.items
        ]
        completions_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        (stage / PROVENANCE_FILE).write_bytes(provenance_bytes)
        (stage / COMPLETIONS_FILE).write_bytes(completions_bytes)
        manifest = CompletionManifest(
            provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
            completions_sha256=hashlib.sha256(completions_bytes).hexdigest(),
            dataset_digest=problem_set_digest(completion_set.problem_set()),
        )
        (stage / MANIFEST_FILE).write_text(record_json(manifest), encoding="utf-8")
        try:
            stage.rename(out_dir)
        except OSError:
            if not out_dir.exists():
                raise
            _check_existing(completion_set, out_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return out_dir


def _check_existing(completion_set: CompletionSet, out_dir: Path) -> None:
    if load_completion_set(out_dir) != completion_set:
        raise ValueError(f"{out_dir}: conflicting CompletionSet already exists")


def load_completion_set(in_dir: Path) -> CompletionSet:
    """Read versioned or legacy artifacts, with file and JSONL line diagnostics."""
    in_dir = Path(in_dir)
    manifest_path = in_dir / MANIFEST_FILE
    manifest = None
    if manifest_path.exists():
        try:
            manifest = CompletionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (ValidationError, UnicodeError) as exc:
            raise ValueError(f"{manifest_path}: {exc}") from exc
        for name, expected in (
            (PROVENANCE_FILE, manifest.provenance_sha256),
            (COMPLETIONS_FILE, manifest.completions_sha256),
        ):
            path = in_dir / name
            if file_digest(path) != expected:
                raise ValueError(f"{path}: SHA256 integrity check failed")
    provenance_path = in_dir / PROVENANCE_FILE
    try:
        provenance = GenerationProvenance.model_validate_json(
            provenance_path.read_text(encoding="utf-8")
        )
    except (ValidationError, UnicodeError) as exc:
        raise ValueError(f"{provenance_path}: {exc}") from exc
    completions_path = in_dir / COMPLETIONS_FILE
    items: list[ProblemCompletions] = []
    with completions_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                items.append(ProblemCompletions.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"{completions_path}:{line_number}: {exc}") from exc
    try:
        result = CompletionSet(provenance=provenance, items=tuple(items))
    except ValidationError as exc:
        raise ValueError(f"{completions_path}: {exc}") from exc
    digest = problem_set_digest(result.problem_set())
    if manifest is not None and manifest.dataset_digest != digest:
        raise ValueError(f"{manifest_path}: dataset content digest mismatch")
    return result
