"""Pure helpers for reading completion artifacts into report-ready inputs."""

from __future__ import annotations

from pathlib import Path

from grpo_decomp.eval.battery import grade
from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.registries import ARMS, EVAL_SETS
from grpo_decomp.schemas import ProblemSet


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
    suffix = f"__{task_set}"
    base = load_completion_set(root / f"base{suffix}")
    correct_by_seed: list[tuple[int | str, CompletionSet]] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("correct-seed") and sub.name.endswith(suffix):
            label = sub.name[: -len(suffix)].partition("-seed")[2]
            seed: int | str = int(label) if label.isdigit() else label
            correct_by_seed.append((seed, load_completion_set(sub)))
    if not correct_by_seed:
        raise ValueError(f"no 'correct-seed<N>{suffix}' dirs under {root}")
    correct_by_seed.sort(key=lambda pair: (isinstance(pair[0], str), pair[0]))
    return base, correct_by_seed


def seed_label(battery_dir: Path) -> int | str:
    """Recover a seed label from a battery dir name (``battery`` -> 0, ``battery-seed2`` -> 2)."""
    name = Path(battery_dir).name
    if name == "battery":
        return 0
    head, _, tail = name.rpartition("-seed")
    return int(tail) if head and tail.isdigit() else name


def discover_completion_sets(root: Path) -> dict[str, dict[str, CompletionSet]]:
    """Load ``<arm>__<set>`` subdirectories into ``{set: {arm: CompletionSet}}``."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"completions dir {root} does not exist")
    grouped: dict[str, dict[str, CompletionSet]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or "__" not in sub.name:
            continue
        arm, _, slug = sub.name.partition("__")
        if arm not in ARMS:
            continue
        grouped.setdefault(slug, {})[arm] = load_completion_set(sub)
    if not grouped:
        raise ValueError(f"no '<arm>__<set>' completion dirs found under {root}")
    return grouped


def validate_report_artifacts(grouped: dict[str, dict[str, CompletionSet]]) -> None:
    """Require report artifacts to match the registered eval sets they claim to be.

    Also require all arms compared within a set to share one prompt strategy: comparing a
    base arm generated with one strategy against a trained arm generated with another is
    exactly the distribution shift the decomposition must not silently absorb.
    """
    for slug, arms in grouped.items():
        if slug not in EVAL_SETS:
            raise ValueError(
                f"unknown report set {slug!r}; known sets are {tuple(sorted(EVAL_SETS))}"
            )
        strategies = {
            arm: completion_set.provenance.prompt_strategy for arm, completion_set in arms.items()
        }
        if len(set(strategies.values())) > 1:
            raise ValueError(
                f"{slug}: arms were generated with different prompt strategies {strategies}; "
                "a decomposition must compare arms on the same prompt distribution"
            )
        expected = EVAL_SETS[slug]()
        for arm, completion_set in arms.items():
            validate_completion_set(slug, arm, completion_set, expected)


def validate_completion_set(
    slug: str, arm: str, completion_set: CompletionSet, expected: ProblemSet
) -> None:
    """Require one artifact to match its registered dataset metadata and problem ids."""
    if completion_set.provenance.dataset != expected.source:
        raise ValueError(
            f"{arm}__{slug}: dataset metadata does not match registered set "
            f"{expected.source.model_dump()}"
        )
    expected_ids = tuple(problem.id for problem in expected)
    actual_ids = tuple(item.problem.id for item in completion_set.items)
    if len(actual_ids) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError(
            f"{arm}__{slug}: problem ids do not match registered set "
            f"(expected {len(expected_ids)}, got {len(actual_ids)})"
        )


def greedy_pass1(completion_set: CompletionSet, policy: str) -> dict[str, bool]:
    """Grade the first sample per problem (pass@1) under `policy`."""
    sampling = completion_set.provenance.sampling
    if sampling.n != 1 or sampling.temperature != 0.0:
        raise ValueError(
            "greedy pass@1 artifacts must have sampling.n=1 and temperature=0.0; "
            f"got n={sampling.n}, temperature={sampling.temperature}"
        )
    first = {item.problem.id: item.samples[0] for item in completion_set.items}
    return grade(completion_set.problem_set(), first, policy=policy)
