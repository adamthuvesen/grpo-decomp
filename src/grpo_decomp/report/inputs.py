"""Load, validate, and grade CompletionSet artifacts for report commands."""

from __future__ import annotations

from pathlib import Path

from grpo_decomp.eval.battery import grade
from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.registries import EVAL_SETS
from grpo_decomp.report.evidence import ArmInput, ComparisonInputs, anchored_inputs, validate_grader


def base_and_correct_seeds(
    root: Path, task_set: str
) -> tuple[CompletionSet, list[tuple[int | str, CompletionSet]]]:
    """Load ``base__<set>`` plus every ``correct-seed<N>__<set>`` under `root`.

    The seed-replicate layout both `report-passk-seeds` and `report-mechanism` consume.
    Numeric seeds sort first (in order), any non-numeric labels after - never int vs str.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"completions dir {root} does not exist")
    if task_set not in EVAL_SETS:
        raise ValueError(
            f"unknown report set {task_set!r}; known sets are {tuple(sorted(EVAL_SETS))}"
        )
    suffix = f"__{task_set}"
    base = load_completion_set(root / f"base{suffix}")
    correct_by_seed: list[tuple[int | str, CompletionSet]] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("correct-seed") and sub.name.endswith(suffix):
            label = sub.name[: -len(suffix)].partition("-seed")[2]
            seed: int | str = int(label) if label.isdigit() else label
            completion_set = load_completion_set(sub)
            correct_by_seed.append((seed, completion_set))
    if not correct_by_seed:
        raise ValueError(f"no 'correct-seed<N>{suffix}' dirs under {root}")
    correct_by_seed.sort(key=lambda pair: (isinstance(pair[0], str), pair[0]))
    validate_seed_artifacts(task_set, base, correct_by_seed)
    return base, correct_by_seed


def seed_label(battery_dir: Path) -> int | str:
    """Recover a seed label from a battery dir name (``battery`` -> 0, ``battery-seed2`` -> 2)."""
    name = Path(battery_dir).name
    if name == "battery":
        return 0
    head, _, tail = name.rpartition("-seed")
    return int(tail) if head and tail.isdigit() else name


def validate_aligned_artifacts(slug: str, arms: dict[str, CompletionSet]) -> None:
    """Require compared artifacts to carry identical data and prompt provenance."""
    ComparisonInputs(tuple(ArmInput(arm, None, cs) for arm, cs in arms.items()))


def validate_same_prompt_strategy(slug: str, arms: dict[str, CompletionSet]) -> None:
    """Require compared artifacts to share one prompt strategy."""
    strategies = {
        arm: completion_set.provenance.prompt_strategy for arm, completion_set in arms.items()
    }
    if len(set(strategies.values())) > 1:
        raise ValueError(
            f"{slug}: arms were generated with different prompt strategies {strategies}; "
            "a decomposition must compare arms on the same prompt distribution"
        )


def validate_seed_artifacts(
    slug: str, base: CompletionSet, correct_by_seed: list[tuple[int | str, CompletionSet]]
) -> None:
    """Require seed-report artifacts to align without reloading the full registered set."""
    anchored_inputs(base, correct_by_seed)


def greedy_pass1(completion_set: CompletionSet, policy: str) -> dict[str, bool]:
    """Grade the first sample per problem (pass@1) under `policy`."""
    validate_grader(completion_set)
    sampling = completion_set.provenance.sampling
    if sampling.n != 1 or sampling.temperature != 0.0:
        raise ValueError(
            "greedy pass@1 artifacts must have sampling.n=1 and temperature=0.0; "
            f"got n={sampling.n}, temperature={sampling.temperature}"
        )
    first = {item.problem.id: item.samples[0] for item in completion_set.items}
    return grade(completion_set.problem_set(), first, policy=policy)


def battery_inputs(
    directories: list[Path], task_set: str, *, roles: tuple[str, ...], include_base: bool = False
) -> ComparisonInputs:
    """Resolve the legacy battery layout once, then validate every input before grading."""
    from grpo_decomp.report.evidence import seed_labels, unique_input_paths

    paths = unique_input_paths(directories)
    labels = seed_labels([seed_label(path) for path in paths])
    arms = []
    if include_base:
        arms.append(ArmInput("base", None, load_completion_set(paths[0] / f"base__{task_set}")))
    for path, seed in zip(paths, labels, strict=True):
        for role in roles:
            arms.append(ArmInput(role, seed, load_completion_set(path / f"{role}__{task_set}")))
    return ComparisonInputs(tuple(arms))
