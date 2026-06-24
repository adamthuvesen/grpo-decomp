"""Pure held-out checkpoint selection helpers for the eval CLI."""

from __future__ import annotations

from pathlib import Path

from llm_grpo_gains.data import load_countdown, load_gsm8k, validation_split
from llm_grpo_gains.schemas import ProblemSet
from llm_grpo_gains.train.provenance import RunProvenance


def select_checkpoint(points: list[dict], rule: str, final_step: int) -> tuple[str, int]:
    """Realize a checkpoint-selection rule over a held-out curve -> (checkpoint dir, step).

    `final` always takes the end-of-training checkpoint; `best_on_validation` takes the
    highest held-out accuracy, breaking ties toward the later (more-trained) step.
    """
    if rule == "final":
        if not any(point["checkpoint"] == "final" for point in points):
            raise ValueError("final checkpoint selection needs a discovered final checkpoint")
        return "final", final_step
    if rule == "best_on_validation":
        if not points:
            raise ValueError("best_on_validation needs a non-empty held-out curve")
        best = max(
            points,
            key=lambda p: (p["accuracy"], p["step"] if p["step"] is not None else final_step),
        )
        return best["checkpoint"], best["step"] if best["step"] is not None else final_step
    raise ValueError(f"unknown checkpoint_selection rule {rule!r}")


def validation_for_run(provenance: RunProvenance) -> ProblemSet:
    """Reconstruct the exact held-out split a run used (deterministic from the source)."""
    ref = provenance.dataset
    if ref.name == "countdown":
        # Countdown ships a dedicated, seed-independent validation split; regenerate it.
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
