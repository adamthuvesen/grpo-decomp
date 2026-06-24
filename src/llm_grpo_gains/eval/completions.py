"""The completion artifact: the seam between generation (GPU) and analysis (CPU).

`generate` (on whatever box holds the model) writes a `CompletionSet`; `report`
(on a cheap CPU box, no generation backend) reads it and feeds the battery and
stats. The artifact is self-contained — it carries each `Problem` (so gold is
present for grading) plus its sampled completions and a provenance record — so the
analysis side needs no network and no dataset re-pinning.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from llm_grpo_gains.provenance import (
    PROVENANCE_PACKAGES,
    git_commit,
    git_is_dirty,
    package_versions,
)
from llm_grpo_gains.schemas import DatasetRef, Problem, ProblemSet, Record

PROVENANCE_FILE = "provenance.json"
COMPLETIONS_FILE = "completions.jsonl"


class SamplingConfig(Record):
    """How completions are drawn. Validated strictly: a bad knob is an explicit error."""

    temperature: float = Field(default=0.0, ge=0.0, description="0 = greedy (pass@1).")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=512, gt=0)
    n: int = Field(default=1, gt=0, description="Completions sampled per problem (uniform).")
    seed: int = 0


class GenerationProvenance(Record):
    """Everything needed to reproduce one generation pass."""

    model: str = Field(description="Model id or checkpoint path.")
    model_revision: str | None
    backend: str = Field(description="'transformers' or 'vllm'.")
    sampling: SamplingConfig
    dataset: DatasetRef
    n_problems: int
    commit: str
    dirty: bool = Field(default=False, description="Worktree had uncommitted changes at capture.")
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
    commit: str | None = None,
    dirty: bool | None = None,
    packages: Sequence[str] = PROVENANCE_PACKAGES,
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
        sampling=sampling,
        dataset=dataset,
        n_problems=n_problems,
        commit=commit if commit is not None else git_commit(),
        dirty=dirty if dirty is not None else git_is_dirty(),
        python_version=sys.version.split()[0],
        package_versions=package_versions(packages),
    )


def write_completion_set(completion_set: CompletionSet, out_dir: Path) -> Path:
    """Write `provenance.json` + a `completions.jsonl` (sorted by id); return `out_dir`.

    Deterministic: equal inputs produce byte-identical files (sorted keys, id-sorted lines).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PROVENANCE_FILE).write_text(
        json.dumps(completion_set.provenance.model_dump(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    items = sorted(completion_set.items, key=lambda item: item.problem.id)
    lines = [json.dumps(item.model_dump(), sort_keys=True) for item in items]
    (out_dir / COMPLETIONS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def load_completion_set(in_dir: Path) -> CompletionSet:
    """Read a `CompletionSet` written by `write_completion_set` (re-validates uniform n)."""
    in_dir = Path(in_dir)
    provenance = GenerationProvenance.model_validate_json(
        (in_dir / PROVENANCE_FILE).read_text(encoding="utf-8")
    )
    raw = (in_dir / COMPLETIONS_FILE).read_text(encoding="utf-8").splitlines()
    items = tuple(ProblemCompletions.model_validate_json(line) for line in raw if line.strip())
    return CompletionSet(provenance=provenance, items=items)
