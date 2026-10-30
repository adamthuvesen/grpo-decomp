"""Validated experimental units shared by CLI, library, and study reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from grpo_decomp.eval.completions import CompletionSet
from grpo_decomp.provenance import git_commit, git_is_dirty, package_versions
from grpo_decomp.registries import verifier_identity
from grpo_decomp.schemas import Record
from grpo_decomp.stats.bootstrap import BOOTSTRAP_ITERATIONS, PAIRED_BOOTSTRAP_SEED


def seed_labels(seeds: Sequence[object]) -> tuple[str, ...]:
    labels = tuple(str(seed) for seed in seeds)
    canonical = tuple(str(int(label)) if label.isdecimal() else label for label in labels)
    if not labels or any(not label.strip() for label in labels):
        raise ValueError("seed labels must be nonempty")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"duplicate seed labels {labels}; supply one input per training replicate")
    return labels


def unique_input_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate input directory; supply each training replicate exactly once")
    return tuple(paths)


def artifact_digest(artifact: CompletionSet) -> str:
    payload = json.dumps(
        artifact.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_grader(artifact: CompletionSet) -> str:
    actual = verifier_identity(artifact.provenance.dataset)
    expected = artifact.provenance.grader_identity
    if expected is not None and expected != actual:
        raise ValueError(
            f"grader mismatch for {artifact.provenance.dataset.name}: expected {expected!r}, "
            f"resolved {actual!r}; install the recorded study plugin before grading"
        )
    return actual


@dataclass(frozen=True)
class ArmInput:
    role: str
    seed: str | None
    completions: CompletionSet


class InputEvidence(Record):
    role: str
    seed: str | None
    content_digest: str
    model: str
    training_run_id: str | None
    training_seed: int | None


class AnalysisProvenance(Record):
    schema_version: Literal[1] = 1
    method: str
    extraction_policy: Literal["strict", "lenient"] = "lenient"
    verifier: str
    inputs: tuple[InputEvidence, ...]
    limitations: tuple[str, ...]
    code_commit: str
    code_dirty: bool | None = None
    package_versions: dict[str, str]
    bootstrap_seed: int | None = PAIRED_BOOTSTRAP_SEED
    bootstrap_iterations: int | None = BOOTSTRAP_ITERATIONS
    confidence_level: float = 0.95


@dataclass(frozen=True)
class ComparisonInputs:
    """Comparable arms with unique replicates within each role.

    Sample counts may differ across roles. Generation seeds may differ. All other
    decoding settings and the complete ordered problem records must match.
    """

    arms: tuple[ArmInput, ...]

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("no comparison inputs")
        anchor = self.arms[0]
        reference = anchor.completions.provenance
        problems = tuple(item.problem for item in anchor.completions.items)
        verifier = validate_grader(anchor.completions)
        roles: dict[str, list[ArmInput]] = {}
        for arm in self.arms:
            roles.setdefault(arm.role, []).append(arm)
            current = arm.completions.provenance
            label = f"{arm.role} seed {arm.seed}"
            if current.dataset != reference.dataset:
                raise ValueError(f"{label}: dataset metadata does not match {anchor.role}")
            if tuple(item.problem for item in arm.completions.items) != problems:
                raise ValueError(f"{label}: problem records do not match {anchor.role}")
            if current.prompt_strategy != reference.prompt_strategy:
                raise ValueError(f"{label}: prompt strategy does not match {anchor.role}")
            if (
                current.prompt_digest is not None
                and reference.prompt_digest is not None
                and current.prompt_digest != reference.prompt_digest
            ):
                raise ValueError(f"{label}: rendered prompts do not match {anchor.role}")
            for field in ("temperature", "top_p", "max_new_tokens"):
                actual = getattr(current.sampling, field)
                expected = getattr(reference.sampling, field)
                if actual != expected:
                    raise ValueError(
                        f"{label}: sampling.{field}={actual} does not match "
                        f"{anchor.role} ({expected}); regenerate with matching decoding settings"
                    )
            if validate_grader(arm.completions) != verifier:
                raise ValueError(f"{label}: verifier does not match {anchor.role}")
        for field in ("prompt_digest", "training_protocol_digest"):
            known = {
                getattr(arm.completions.provenance, field)
                for arm in self.arms
                if getattr(arm.completions.provenance, field) is not None
            }
            if len(known) > 1:
                raise ValueError(f"comparison inputs differ in {field}")
        base = next((arm.completions.provenance for arm in self.arms if arm.role == "base"), None)
        if base is not None:
            base_revision = base.resolved_model_revision or base.model_revision
            for arm in self.arms:
                current = arm.completions.provenance
                if (
                    current.training_base_model is not None
                    and current.training_base_model != base.model
                ):
                    raise ValueError(
                        f"{arm.role}: training base model does not match the base anchor"
                    )
                if (
                    current.training_base_model_revision is not None
                    and base_revision is not None
                    and current.training_base_model_revision != base_revision
                ):
                    raise ValueError(
                        f"{arm.role}: training base revision does not match the base anchor"
                    )
        for role, arms in roles.items():
            labelled = [arm for arm in arms if arm.seed is not None]
            if labelled:
                seed_labels([arm.seed for arm in labelled])
            if len({arm.completions.provenance.sampling.n for arm in arms}) != 1:
                raise ValueError(f"{role} arms must share n (sampling.n)")
            if len({artifact_digest(arm.completions) for arm in arms}) != len(arms):
                raise ValueError(
                    f"{role}: duplicate completion artifact; provide independent replicates"
                )
            provenance = [arm.completions.provenance for arm in arms]
            for field in (
                "training_protocol_digest",
                "training_base_model",
                "training_base_model_revision",
                "training_reward",
            ):
                known = {getattr(p, field) for p in provenance if getattr(p, field) is not None}
                if len(known) > 1:
                    raise ValueError(f"{role}: replicates differ in {field}")
            for field in ("training_run_id", "training_seed"):
                identities = [
                    getattr(p, field) for p in provenance if getattr(p, field) is not None
                ]
                if len(set(identities)) != len(identities):
                    raise ValueError(
                        f"{role}: duplicate {field}; attempts are not independent seeds"
                    )
            for arm in labelled:
                seed = arm.completions.provenance.training_seed
                if seed is not None and arm.seed != str(seed):
                    raise ValueError(
                        f"{role}: seed label {arm.seed!r} disagrees with training_seed={seed}"
                    )

    def provenance(self, method: str, *, bootstrap: bool = True) -> AnalysisProvenance:
        limitations = []
        if any(
            (
                arm.completions.provenance.training_seed is None
                or arm.completions.provenance.training_run_id is None
            )
            for arm in self.arms
            if arm.seed is not None
        ):
            limitations.append(
                "Training identity is unavailable for some inputs. "
                "Seed labels are supplied, not verified."
            )
        if any(arm.completions.provenance.prompt_digest is None for arm in self.arms):
            limitations.append(
                "Rendered prompt identity is unavailable for some inputs. "
                "Only strategy names match."
            )
        if any(arm.completions.provenance.grader_identity is None for arm in self.arms):
            limitations.append(
                "Legacy inputs do not record a verifier; the installed verifier was used."
            )
        return AnalysisProvenance(
            method=method,
            verifier=validate_grader(self.arms[0].completions),
            inputs=tuple(
                InputEvidence(
                    role=arm.role,
                    seed=arm.seed,
                    content_digest=artifact_digest(arm.completions),
                    model=arm.completions.provenance.model,
                    training_run_id=arm.completions.provenance.training_run_id,
                    training_seed=arm.completions.provenance.training_seed,
                )
                for arm in self.arms
            ),
            limitations=tuple(limitations),
            code_commit=git_commit(),
            code_dirty=git_is_dirty(),
            bootstrap_seed=PAIRED_BOOTSTRAP_SEED if bootstrap else None,
            bootstrap_iterations=BOOTSTRAP_ITERATIONS if bootstrap else None,
            package_versions=package_versions(
                ("grpo-decomp", "math-verify", "numpy", "scipy", "eval-audit")
            ),
        )


def anchored_inputs(
    base: CompletionSet,
    replicates: Sequence[tuple[object, CompletionSet]],
    *,
    role: str = "correct",
) -> ComparisonInputs:
    labels = seed_labels([label for label, _ in replicates])
    return ComparisonInputs(
        (
            ArmInput("base", None, base),
            *(ArmInput(role, label, cs) for label, (_, cs) in zip(labels, replicates, strict=True)),
        )
    )
