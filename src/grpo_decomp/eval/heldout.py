"""Held-out validation curves across saved checkpoints."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from grpo_decomp.eval.battery import grade
from grpo_decomp.eval.completions import SamplingConfig
from grpo_decomp.registries import VALIDATION_RECONSTRUCTORS
from grpo_decomp.schemas import ProblemSet, Record
from grpo_decomp.train.provenance import RunProvenance


class HeldoutPoint(Record):
    """One checkpoint's held-out validation accuracy."""

    checkpoint: str
    step: int | None
    accuracy: float
    n_correct: int
    n: int


class HeldoutCurve(Record):
    """A full held-out accuracy curve."""

    run: str
    validation_size: int
    policy: str = Field(description="Answer extraction policy used for grading.")
    points: tuple[HeldoutPoint, ...]


def validation_for_run(provenance: RunProvenance) -> ProblemSet:
    """Reconstruct the exact held-out split a run used (deterministic from the source).

    Dispatches on ``DatasetRef.name`` to the reconstructor a study registered with
    :func:`grpo_decomp.registries.register_validation_reconstructor`.
    """
    name = provenance.dataset.name
    try:
        reconstruct = VALIDATION_RECONSTRUCTORS[name]
    except KeyError as exc:
        raise ValueError(
            f"no held-out reconstructor registered for dataset {name!r}; "
            f"registered: {tuple(sorted(VALIDATION_RECONSTRUCTORS))}"
        ) from exc
    return reconstruct(provenance)


def _checkpoint_step(path: Path) -> int | None:
    return None if path.name == "final" else int(path.name.split("-", 1)[1])


def discover_checkpoints(run_dir: Path) -> list[Path]:
    """Saved checkpoints under ``run_dir/checkpoints``, sorted by step with ``final`` last."""
    root = Path(run_dir) / "checkpoints"
    if not root.is_dir():
        raise ValueError(f"no checkpoints/ directory under {run_dir}")
    numbered: list[Path] = []
    final: Path | None = None
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "final":
            final = sub
        elif sub.name.startswith("checkpoint-") and sub.name.split("-", 1)[1].isdigit():
            numbered.append(sub)
    numbered.sort(key=lambda path: _checkpoint_step(path) or 0)
    if final is not None:
        numbered.append(final)
    if not numbered:
        raise ValueError(f"no 'checkpoint-<step>' or 'final' dirs under {root}")
    return numbered


def _load_run_provenance(run_dir: Path) -> RunProvenance:
    return RunProvenance.model_validate_json(
        (Path(run_dir) / "provenance.json").read_text(encoding="utf-8")
    )


def run_heldout_curve(run_dir: Path, config: SamplingConfig, *, backend: str) -> HeldoutCurve:
    """Generate and grade a held-out curve for every checkpoint in one run directory.

    The model backend import stays lazy so CPU-only commands do not load GPU dependencies.
    """
    from grpo_decomp.eval.generate import generate

    run_dir = Path(run_dir)
    provenance = _load_run_provenance(run_dir)
    validation = validation_for_run(provenance)

    points: list[HeldoutPoint] = []
    for checkpoint in discover_checkpoints(run_dir):
        # Score on the same prompt strategy the run trained on.
        samples = generate(
            str(checkpoint),
            validation,
            config,
            backend=backend,
            prompt_strategy=provenance.prompt_strategy,
        )
        graded = grade(validation, {pid: s[0] for pid, s in samples.items()}, policy="lenient")
        n_correct = sum(graded.values())
        accuracy = n_correct / len(graded)
        points.append(
            HeldoutPoint(
                checkpoint=checkpoint.name,
                step=_checkpoint_step(checkpoint),
                accuracy=accuracy,
                n_correct=n_correct,
                n=len(graded),
            )
        )

    return HeldoutCurve(
        run=str(run_dir),
        validation_size=len(validation),
        policy="lenient",
        points=tuple(points),
    )
