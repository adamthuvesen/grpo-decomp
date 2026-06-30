"""Held-out checkpoint selection and evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    """A full held-out accuracy curve plus the selection rule metadata."""

    run: str
    validation_size: int
    policy: str = Field(description="Answer extraction policy used for grading.")
    points: tuple[HeldoutPoint, ...]
    selected_checkpoint: str | None = None
    selected_step: int | None = None
    rule: str | None = None


def select_checkpoint(
    points: Sequence[HeldoutPoint] | Sequence[dict],
    rule: str,
    final_step: int,
) -> tuple[str, int]:
    """Realize a checkpoint-selection rule over a held-out curve -> (checkpoint dir, step).

    `final` always takes the end-of-training checkpoint; `best_on_validation` takes the
    highest held-out accuracy, breaking ties toward the later (more-trained) step.
    """
    normalized = [
        p if isinstance(p, HeldoutPoint) else HeldoutPoint.model_validate(p) for p in points
    ]
    if rule == "final":
        if not any(point.checkpoint == "final" for point in normalized):
            raise ValueError("final checkpoint selection needs a discovered final checkpoint")
        return "final", final_step
    if rule == "best_on_validation":
        if not normalized:
            raise ValueError("best_on_validation needs a non-empty held-out curve")
        best = max(
            normalized,
            key=lambda p: (p.accuracy, p.step if p.step is not None else final_step),
        )
        return best.checkpoint, best.step if best.step is not None else final_step
    raise ValueError(f"unknown checkpoint_selection rule {rule!r}")


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


def _write_run_provenance(run_dir: Path, provenance: RunProvenance) -> None:
    (Path(run_dir) / "provenance.json").write_text(
        json.dumps(provenance.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def run_heldout_curve(run_dir: Path, config: SamplingConfig, *, backend: str) -> HeldoutCurve:
    """Generate and grade a held-out curve for every checkpoint in one run directory.

    The model backend import stays lazy so report-only commands can import this module
    without loading GPU dependencies.
    """
    from grpo_decomp.eval.generate import generate

    run_dir = Path(run_dir)
    provenance = _load_run_provenance(run_dir)
    validation = validation_for_run(provenance)

    points: list[HeldoutPoint] = []
    for checkpoint in discover_checkpoints(run_dir):
        # Score on the SAME prompt strategy the run trained on, or checkpoint selection
        # would compare an off-distribution prompt against the training distribution.
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

    name, step = select_checkpoint(
        points, provenance.checkpoint_selection, provenance.grpo.max_steps
    )
    return HeldoutCurve(
        run=str(run_dir),
        validation_size=len(validation),
        policy="lenient",
        points=tuple(points),
        selected_checkpoint=name,
        selected_step=step,
        rule=provenance.checkpoint_selection,
    )


def write_selected_provenance(run_dir: Path, curve: HeldoutCurve) -> None:
    """Record the checkpoint selected by a held-out curve in run provenance."""
    run_dir = Path(run_dir)
    provenance = _load_run_provenance(run_dir)
    selected = provenance.model_copy(
        update={
            "selected_step": curve.selected_step,
            "selected_checkpoint": curve.selected_checkpoint,
        }
    )
    _write_run_provenance(run_dir, selected)
