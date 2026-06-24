"""Pure helpers for reading completion artifacts into report-ready inputs."""

from __future__ import annotations

from pathlib import Path

from llm_grpo_gains.eval.battery import BatteryResult, grade, run_battery
from llm_grpo_gains.eval.completions import CompletionSet, load_completion_set
from llm_grpo_gains.eval.registry import ARMS, PROBES, SETS
from llm_grpo_gains.report.decomposition import DecompositionRow
from llm_grpo_gains.schemas import ProblemSet
from llm_grpo_gains.stats.compare import compare


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
    """Require report artifacts to match the registered eval sets they claim to be."""
    for slug, arms in grouped.items():
        if slug not in SETS:
            raise ValueError(f"unknown report set {slug!r}; known sets are {tuple(sorted(SETS))}")
        expected = SETS[slug]()
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


def control_row(slug: str, arms: dict[str, CompletionSet]) -> DecompositionRow:
    """Build one base-vs-correct control row from loaded completion artifacts."""
    comparison = compare(
        "base",
        greedy_pass1(arms["base"], "lenient"),
        "correct",
        greedy_pass1(arms["correct"], "lenient"),
    )
    return DecompositionRow(
        control=f"control ({slug})", probes=PROBES.get(slug, slug), comparison=comparison
    )


def battery_at(completion_set: CompletionSet, k_values: list[int]) -> BatteryResult:
    """Run the eval battery at exactly the requested k values."""
    return run_battery(
        completion_set.problem_set(), completion_set.completions_by_id(), k_values=k_values
    )


def vanilla_at(battery: BatteryResult, k: int) -> float:
    """The vanilla pass@k at exactly `k` from a battery result."""
    for entry in battery.pass_at_k:
        if entry.k == k:
            return entry.vanilla
    raise ValueError(f"pass@{k} was not computed")


def elicitation_note(base: CompletionSet, correct: CompletionSet) -> str:
    """The elicitation / capability-expansion panel line.

    When both arms are sampled at n>1, report the pass@k curve (base vs correct at
    matched k) - the certified-expansion readout: higher pass@k coverage means
    new capability, not just improved pass@1 reliability. Otherwise fall back to
    the base pass@k vs correct pass@1 elicitation line, or a plain pass@1 line when n==1.
    """
    n_base = base.provenance.sampling.n
    n_correct = correct.provenance.sampling.n
    if n_base > 1 and n_correct > 1:
        k = min(n_base, n_correct)
        base_battery = battery_at(base, sorted({1, k}))
        correct_battery = battery_at(correct, sorted({1, k}))
        base_k = vanilla_at(base_battery, k)
        correct_k = vanilla_at(correct_battery, k)
        return (
            f"pass@k curve: base pass@{k}={base_k:.2f} vs correct pass@{k}={correct_k:.2f} "
            f"(Δ={correct_k - base_k:+.2f}); pass@1 base={base_battery.lenient_accuracy:.2f}, "
            f"correct={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    base_battery = battery_at(base, sorted({1, n_base}))
    correct_battery = battery_at(correct, [1])
    if n_base > 1:
        return (
            f"base pass@{n_base}={base_battery.pass_at_k[-1].vanilla:.2f} vs "
            f"correct pass@1={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    return (
        f"pass@1: base={base_battery.lenient_accuracy:.2f}, "
        f"correct={correct_battery.lenient_accuracy:.2f}; "
        "high-n pass@k coverage deferred to a Phase-2 sampling run"
    )
