"""Held-out checkpoint selection and evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from llm_grpo_gains.data import load_countdown, load_gsm8k, validation_split
from llm_grpo_gains.eval.battery import grade
from llm_grpo_gains.eval.completions import SamplingConfig
from llm_grpo_gains.schemas import ProblemSet, Record
from llm_grpo_gains.train.provenance import RunProvenance


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
    """Reconstruct the exact held-out split a run used (deterministic from the source)."""
    ref = provenance.dataset
    if ref.name == "countdown":
        return load_countdown("validation")
    if ref.name == "openai/gsm8k":
        train = load_gsm8k(ref.split, revision=ref.revision)
        _, validation = validation_split(train, n=provenance.validation_size, seed=provenance.seed)
        return validation
    raise ValueError(f"held-out reconstruction supports gsm8k train or countdown, got {ref.name!r}")


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
    numbered.sort(key=lambda path: int(path.name.split("-", 1)[1]))
    if final is not None:
        numbered.append(final)
    if not numbered:
        raise ValueError(f"no 'checkpoint-<step>' or 'final' dirs under {root}")
    return numbered


def run_heldout_curve(run_dir: Path, config: SamplingConfig, *, backend: str) -> HeldoutCurve:
    """Generate and grade a held-out curve for every checkpoint in one run directory.

    The model backend import stays lazy so report-only commands can import this module
    without loading GPU dependencies.
    """
    from llm_grpo_gains.eval.generate import generate

    run_dir = Path(run_dir)
    provenance = RunProvenance.model_validate_json(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    validation = validation_for_run(provenance)

    points: list[HeldoutPoint] = []
    for checkpoint in discover_checkpoints(run_dir):
        samples = generate(str(checkpoint), validation, config, backend=backend)
        graded = grade(validation, {pid: s[0] for pid, s in samples.items()}, policy="lenient")
        n_correct = sum(graded.values())
        accuracy = n_correct / len(graded)
        step = None if checkpoint.name == "final" else int(checkpoint.name.split("-", 1)[1])
        points.append(
            HeldoutPoint(
                checkpoint=checkpoint.name,
                step=step,
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
    provenance = RunProvenance.model_validate_json(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    selected = provenance.model_copy(
        update={
            "selected_step": curve.selected_step,
            "selected_checkpoint": curve.selected_checkpoint,
        }
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(selected.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
