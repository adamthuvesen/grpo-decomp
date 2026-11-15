"""CPU-only job planning for the study's registered task profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from grpo_decomp.eval.completions import SamplingConfig
from grpo_decomp.prompts import EVAL_MAX_NEW_TOKENS
from grpo_decomp.provenance import (
    completed_checkpoint,
    load_run_provenance,
    resolve_run_dir,
)
from grpo_decomp.registries import (
    DEFAULT_PROMPT_STRATEGY,
    EVAL_SETS,
    EvalTaskProfile,
    default_task,
    get_task_profile,
)
from grpo_decomp.schemas import ProblemSet
from grpo_decomp.splits import dev_slice
from grpo_decomp.train.config import load_arm_config


@dataclass(frozen=True)
class CompletionJob:
    arm: str
    model_ref: str
    model_revision: str | None
    set_name: str
    config: SamplingConfig
    problems: ProblemSet | None = None
    # The prompt strategy this arm trained on; eval MUST match it or the decomposition
    # compares a model against an off-distribution prompt.
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY


def _task_profile(task: str | None) -> EvalTaskProfile:
    name = task if task is not None else default_task()
    if name is None:
        raise ValueError("task is required when no study registers a default task")
    return get_task_profile(name)


def validate_battery_request(task: str | None, scope: str, seed: int) -> None:
    """Validate the complete generation request before dispatch."""
    profile = _task_profile(task)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if scope not in {"full", "placebo", "random-vs-base", "controls"}:
        raise ValueError(f"unknown battery scope {scope!r}")
    if scope == "controls" and not profile.control_sets:
        raise ValueError(f"scope 'controls' needs control sets; task {task!r} has none")
    for name in (profile.task_set, *profile.control_sets):
        if name not in EVAL_SETS:
            raise ValueError(f"unknown set {name!r}")
    load_arm_config(Path(profile.base_config))


def validate_passk_request(
    task: str | None,
    n_base: int,
    n_correct: int,
    correct_seeds: str,
    set_name: str | None,
    limit: int | None,
) -> tuple[int, ...]:
    """Validate panel sampling and seed identity before data or checkpoint access."""
    profile = _task_profile(task)
    eval_set = set_name or profile.task_set
    if eval_set not in EVAL_SETS:
        raise ValueError(f"unknown set {eval_set!r}; known sets are {tuple(sorted(EVAL_SETS))}")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"limit must be >= 1, got {limit}")
    if type(n_base) is not int or type(n_correct) is not int or min(n_base, n_correct) < 1:
        raise ValueError("n_base and n_correct must be positive integers")
    try:
        seeds = tuple(int(seed.strip()) for seed in correct_seeds.split(","))
    except ValueError as exc:
        raise ValueError("correct_seeds must be comma-separated nonnegative integers") from exc
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("correct_seeds must contain distinct nonnegative integers")
    load_arm_config(Path(profile.base_config))
    return seeds


def plan_battery(
    runs_dir: Path,
    *,
    seed: int = 0,
    scope: str = "full",
    task: str | None = None,
) -> tuple[Path, list[CompletionJob]]:
    """Resolve all checkpoints before the first model load."""
    validate_battery_request(task, scope, seed)
    profile = _task_profile(task)
    base = load_arm_config(Path(profile.base_config))

    def trained_checkpoint(role: str) -> tuple[str, str]:
        run_dir = resolve_run_dir(runs_dir / f"{profile.run_prefix}{role}-seed{seed}")
        checkpoint = completed_checkpoint(run_dir)
        prov = load_run_provenance(run_dir)
        return str(checkpoint), prov.prompt_strategy

    base_row = ("base", base.base_model, base.base_model_revision, base.prompt_strategy)

    def trained_row(role: str) -> tuple[str, str, str | None, str]:
        checkpoint, strategy = trained_checkpoint(role)
        return role, checkpoint, None, strategy

    if scope == "full":
        matrix = [
            (*base_row, [profile.task_set, *profile.control_sets]),
            (*trained_row("correct"), [profile.task_set, *profile.control_sets]),
            (*trained_row("random"), [profile.task_set]),
        ]
    elif scope == "placebo":
        matrix = [
            (*trained_row("correct"), [profile.task_set]),
            (*trained_row("random"), [profile.task_set]),
        ]
    elif scope == "random-vs-base":
        matrix = [(*trained_row("random"), [profile.task_set])]
        if seed == 0:
            matrix.insert(0, (*base_row, [profile.task_set]))
    elif scope == "controls":
        if not profile.control_sets:
            raise ValueError(f"scope 'controls' needs control sets; task {task!r} has none")
        matrix = [(*trained_row("correct"), list(profile.control_sets))]
    else:
        raise ValueError(
            f"scope must be 'full', 'placebo', 'random-vs-base', or 'controls', got {scope!r}"
        )

    config = SamplingConfig(
        temperature=0.0, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=1, seed=0
    )
    out_root = runs_dir / (
        profile.battery_root if seed == 0 else f"{profile.battery_root}-seed{seed}"
    )

    jobs = [
        CompletionJob(arm, model_ref, revision, set_name, config, prompt_strategy=strategy)
        for arm, model_ref, revision, strategy, set_names in matrix
        for set_name in set_names
    ]
    return out_root, jobs


def plan_passk(
    runs_dir: Path,
    *,
    task: str | None = None,
    n_base: int = 16,
    n_correct: int = 8,
    correct_seeds: str = "0,1,2,3,4,5",
    set_name: str | None = None,
    limit: int | None = None,
) -> tuple[Path, list[CompletionJob]]:
    """Build a complete pass@k panel before the first model load."""
    profile = _task_profile(task)
    eval_set = set_name or profile.task_set
    seeds = validate_passk_request(task, n_base, n_correct, correct_seeds, set_name, limit)
    base = load_arm_config(Path(profile.base_config))

    def trained_checkpoint(seed: int) -> tuple[str, str]:
        run_dir = resolve_run_dir(runs_dir / f"{profile.run_prefix}correct-seed{seed}")
        checkpoint = completed_checkpoint(run_dir)
        prov = load_run_provenance(run_dir)
        return str(checkpoint), prov.prompt_strategy

    out_root = runs_dir / profile.passk_multiseed_root
    problems = EVAL_SETS[eval_set]()
    if limit is not None:
        problems = dev_slice(problems, n=limit, seed=0)

    def sampled_config(n: int) -> SamplingConfig:
        return SamplingConfig(
            temperature=0.7, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=n, seed=0
        )

    def correct_job(seed: int) -> CompletionJob:
        checkpoint, strategy = trained_checkpoint(seed)
        return CompletionJob(
            f"correct-seed{seed}",
            checkpoint,
            None,
            eval_set,
            sampled_config(n_correct),
            problems,
            prompt_strategy=strategy,
        )

    jobs = [
        CompletionJob(
            "base",
            base.base_model,
            base.base_model_revision,
            eval_set,
            sampled_config(n_base),
            problems,
            prompt_strategy=base.prompt_strategy,
        ),
        *(correct_job(seed) for seed in seeds),
    ]
    return out_root, jobs
