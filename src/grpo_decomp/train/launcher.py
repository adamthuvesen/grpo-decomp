"""Launch one GRPO training arm over TRL, recording provenance.

`trl` / `transformers` are imported lazily inside `launch`, so this module imports
on a CPU box even though the GPU stack is Linux/CUDA-only. The dataset preparation,
provenance recording, and reward selection are GPU-independent and unit-tested; the
trainer construction and `train()` run on the rented instance.

GPU-only checks stay on the rented instance:
periodic held-out accuracy is scored post-hoc via `grpo-decomp heldout`, vLLM-colocate
memory tuning, and a ``completions/clipped_ratio`` check (a non-zero ratio means the
trainer's EOS and vLLM's stop token disagree).
The per-step unparseable rate is already logged by the `correct` reward.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from datasets import Dataset

from grpo_decomp.provenance import RunStatus, problem_set_digest
from grpo_decomp.registries import REWARDS, TRAIN_DATASETS, PromptStrategy, get_prompt_strategy
from grpo_decomp.rewards import get_reward
from grpo_decomp.schemas import ProblemSet, record_json
from grpo_decomp.splits import dev_slice
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.provenance import (
    capture_provenance,
    finish_run,
    write_run_record,
)


def _load_train_and_validation(arm: ArmConfig) -> tuple[ProblemSet, ProblemSet]:
    """Load an arm's ``(train, validation)`` splits from the registered train dataset.

    How the validation split is derived is the dataset's concern (GSM8K carves a per-seed
    split from train; Countdown ships a fixed, seed-independent one) — the launcher just
    asks the registered dataset to ``load(seed)``.
    """
    try:
        dataset = TRAIN_DATASETS[arm.dataset]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset {arm.dataset!r}; registered: {tuple(sorted(TRAIN_DATASETS))}"
        ) from exc
    return dataset.load(arm.seed)


def to_dataset(problems: ProblemSet, strategy: PromptStrategy) -> Dataset:
    """Turn a `ProblemSet` into a TRL GRPO dataset using the arm's prompt strategy.

    A ``prompt`` column drives generation; ``gold_answer`` is forwarded by the
    trainer to the reward function.
    """
    rows = [
        {"prompt": strategy.build_prompt(problem.question), "gold_answer": problem.gold_answer}
        for problem in problems
    ]
    return Dataset.from_list(rows)


def prepare_run(
    arm: ArmConfig,
    problems: ProblemSet,
    *,
    validation_size: int,
    validation: ProblemSet | None = None,
    output_root: Path,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Create the run directory and write ``provenance.json``; return the run dir.

    Records the realized train/validation split sizes, so a small subset is
    distinguishable from a full run (which otherwise share a `DatasetRef`).
    `commit`/`dirty` override the git-derived values — on Modal the image strips
    `.git`, so the local entrypoint computes them and passes them in.
    """
    arm = ArmConfig.model_validate(arm.model_dump())
    output_root = Path(output_root)
    alias = output_root / f"{arm.name}-seed{arm.seed}"
    if not alias.resolve().is_relative_to(output_root.resolve()):
        raise ValueError(f"run path escapes output root: {alias}")
    attempt_id = uuid4().hex
    run_dir = alias / "attempts" / attempt_id
    if not run_dir.resolve().is_relative_to(output_root.resolve()):
        raise ValueError(f"attempt path escapes output root: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    provenance = capture_provenance(
        arm,
        problems.source,
        attempt_id=attempt_id,
        validation_identity=problem_set_digest(validation) if validation is not None else None,
        train_size=len(problems),
        validation_size=validation_size,
        commit=commit,
        dirty=dirty,
    )
    (run_dir / "provenance.json").write_text(record_json(provenance), encoding="utf-8")
    write_run_record(run_dir / "status.json", RunStatus(attempt_id=attempt_id, state="running"))
    return run_dir


def smoke_overrides(arm: ArmConfig, max_steps: int | None) -> ArmConfig:
    """Cap a run to `max_steps` and log every step.

    Returns `arm` unchanged when `max_steps` is None. Logging is forced to every
    step so the ``completions/clipped_ratio`` EOS warning is visible in a handful
    of steps, and a checkpoint is saved at the end. The effective config is what
    gets recorded in provenance, so a capped run is never mistaken for a full run.
    """
    if max_steps is None:
        return arm
    data = arm.model_dump()
    data["grpo"].update(max_steps=max_steps, logging_steps=1, save_steps=max_steps)
    return ArmConfig.model_validate(data)


def preflight_arm(
    arm: ArmConfig,
    *,
    smoke_problems: int | None = None,
    max_steps: int | None = None,
) -> ArmConfig:
    """Validate effective settings and registry keys before writes or GPU dispatch."""
    arm = ArmConfig.model_validate(arm.model_dump())
    arm = smoke_overrides(arm, max_steps)
    if smoke_problems is not None and (type(smoke_problems) is not int or smoke_problems < 1):
        raise ValueError("smoke_problems must be a positive integer")
    if arm.dataset not in TRAIN_DATASETS:
        raise ValueError(f"unknown dataset {arm.dataset!r}")
    if arm.reward not in REWARDS:
        raise ValueError(f"unknown reward {arm.reward!r}")
    get_prompt_strategy(arm.prompt_strategy)
    return arm


def launch(
    arm: ArmConfig,
    *,
    output_root: Path = Path("runs"),
    smoke_problems: int | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Run one GRPO arm end-to-end on a GPU box; return the run directory.

    The validation split is held out for post-training checkpoint diagnostics;
    `smoke_problems`, if set, trains on a small deterministic subset, and `max_steps`
    caps the step count. `commit`/`dirty`
    override git provenance (the Modal container has no `.git`).
    """
    arm = preflight_arm(arm, smoke_problems=smoke_problems, max_steps=max_steps)
    train_problems, validation = _load_train_and_validation(arm)
    if smoke_problems is not None:
        train_problems = dev_slice(train_problems, n=smoke_problems, seed=arm.seed)

    if not train_problems or not validation:
        raise ValueError("training and validation splits must both contain problems")
    strategy = get_prompt_strategy(arm.prompt_strategy)
    dataset = to_dataset(train_problems, strategy)
    reward = get_reward(arm.reward, seed=arm.seed)
    run_dir = prepare_run(
        arm,
        train_problems,
        validation_size=len(validation),
        validation=validation,
        output_root=output_root,
        commit=commit,
        dirty=dirty,
    )
    try:
        try:
            # Lazy GPU imports stay inside the cleanup scope, including trainer construction.
            from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]  # GPU libraries are absent from the CPU environment
            from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportMissingImports]  # GPU libraries are absent from the CPU environment

            config = GRPOConfig(
                output_dir=str(run_dir / "checkpoints"),
                seed=arm.seed,
                run_name=f"{arm.name}-seed{arm.seed}-{run_dir.name}",
                report_to=["wandb"],
                model_init_kwargs=(
                    {"revision": arm.base_model_revision} if arm.base_model_revision else {}
                ),
                **arm.grpo.model_dump(),
            )
            tokenizer = AutoTokenizer.from_pretrained(
                arm.base_model, revision=arm.base_model_revision
            )
            strategy.prepare_tokenizer(tokenizer)
            trainer = GRPOTrainer(
                model=arm.base_model,
                reward_funcs=reward,
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            trainer.train()
            trainer.save_model(str(run_dir / "checkpoints" / "final"))
        finally:
            _shutdown_distributed()
    except BaseException as exc:
        finish_run(run_dir, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        finish_run(run_dir)
    return run_dir


def _shutdown_distributed() -> None:
    """Tear down the torch.distributed process group for a clean NCCL shutdown.

    A GRPO + vLLM-colocate run initializes a process group; without this, exit logs a
    ``destroy_process_group() was not called ... can leak resources`` warning and the
    container teardown can stall. Guarded + lazy so the launcher still imports on a
    box without torch, and is a no-op when no group was initialized.
    """
    try:
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]  # GPU libraries are absent from the CPU environment
    except ImportError:
        return

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
