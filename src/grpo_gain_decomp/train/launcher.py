"""Launch one GRPO training arm over TRL, recording provenance.

`trl` / `transformers` are imported lazily inside `launch`, so this module imports
on a CPU box even though the GPU stack is Linux/CUDA-only. The dataset preparation,
provenance recording, and reward selection are GPU-independent and unit-tested; the
trainer construction and `train()` run on the rented instance.

Box-side items intentionally left to the GPU instance (they need real generation):
periodic held-out accuracy logging via a TRL callback over the validation split,
vLLM-colocate memory tuning, and a ``completions/clipped_ratio`` smoke check on the
day-1 run (a non-zero ratio means the trainer's EOS and vLLM's stop token disagree).
The per-step unparseable rate is already logged by the `correct` reward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from grpo_gain_decomp.data import dev_slice, load_countdown, load_gsm8k, validation_split
from grpo_gain_decomp.prompts import build_prompt
from grpo_gain_decomp.rewards import get_reward
from grpo_gain_decomp.schemas import ProblemSet
from grpo_gain_decomp.train.config import ArmConfig
from grpo_gain_decomp.train.provenance import capture_provenance


def _load_train_and_validation(arm: ArmConfig) -> tuple[ProblemSet, ProblemSet]:
    """Load the train and held-out splits for an arm, dispatched by its dataset.

    GSM8K carves a per-seed validation split from its train split; Countdown ships a
    dedicated, seed-independent `validation` split (the dataset is fixed across seeds, so
    only GRPO's own randomness varies by seed — mirroring the GSM8K placebo comparison).
    """
    if arm.dataset == "countdown":
        return load_countdown("train"), load_countdown("validation")
    train = load_gsm8k(arm.train_split)
    return validation_split(train, seed=arm.seed)


def to_dataset(problems: ProblemSet) -> Dataset:
    """Turn a `ProblemSet` into a TRL GRPO dataset.

    A ``prompt`` column drives generation; ``gold_answer`` is forwarded by the
    trainer to the reward function.
    """
    rows = [
        {"prompt": build_prompt(problem.question), "gold_answer": problem.gold_answer}
        for problem in problems
    ]
    return Dataset.from_list(rows)


def prepare_run(
    arm: ArmConfig,
    problems: ProblemSet,
    *,
    validation_size: int,
    output_root: Path,
    commit: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """Create the run directory and write ``provenance.json``; return the run dir.

    Records the realized train/validation split sizes, so a smoke subset is
    distinguishable from a full run (which otherwise share a `DatasetRef`).
    `commit`/`dirty` override the git-derived values — on Modal the image strips
    `.git`, so the local entrypoint computes them and passes them in.
    """
    run_dir = Path(output_root) / f"{arm.name}-seed{arm.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = capture_provenance(
        arm,
        problems.source,
        train_size=len(problems),
        validation_size=validation_size,
        commit=commit,
        dirty=dirty,
    )
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def smoke_overrides(arm: ArmConfig, max_steps: int | None) -> ArmConfig:
    """Cap a run to `max_steps` and log every step — the cheap day-1 dry run.

    Returns `arm` unchanged when `max_steps` is None. Logging is forced to every
    step so the ``completions/clipped_ratio`` EOS warning is visible in a handful
    of steps, and a checkpoint is saved at the end. The effective config is what
    gets recorded in provenance, so a smoke is never mistaken for a full run.
    """
    if max_steps is None:
        return arm
    grpo = arm.grpo.model_copy(
        update={"max_steps": max_steps, "logging_steps": 1, "save_steps": max_steps}
    )
    return arm.model_copy(update={"grpo": grpo})


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

    The validation split is held out of training (for later checkpoint selection);
    `smoke_problems`, if set, trains on a small deterministic subset, and `max_steps`
    caps the step count — together the cheap day-1 infra-tax dry run. `commit`/`dirty`
    override git provenance (the Modal container has no `.git`).
    """
    arm = smoke_overrides(arm, max_steps)
    train_problems, validation = _load_train_and_validation(arm)
    if smoke_problems is not None:
        train_problems = dev_slice(train_problems, n=smoke_problems, seed=arm.seed)

    run_dir = prepare_run(
        arm,
        train_problems,
        validation_size=len(validation),
        output_root=output_root,
        commit=commit,
        dirty=dirty,
    )
    dataset = to_dataset(train_problems)
    reward = get_reward(arm.reward, seed=arm.seed)

    # Lazy GPU imports: only present with the `train` extra on Linux/CUDA.
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(arm.base_model, revision=arm.base_model_revision)
    _prepare_tokenizer(tokenizer)

    config = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        seed=arm.seed,
        run_name=f"{arm.name}-seed{arm.seed}",
        report_to=["wandb"],
        model_init_kwargs=(
            {"revision": arm.base_model_revision} if arm.base_model_revision else {}
        ),
        **arm.grpo.as_grpo_kwargs(),
    )
    trainer = GRPOTrainer(
        model=arm.base_model,
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    try:
        trainer.train()
        trainer.save_model(str(run_dir / "checkpoints" / "final"))
    finally:
        _shutdown_distributed()
    return run_dir


def _shutdown_distributed() -> None:
    """Tear down the torch.distributed process group for a clean NCCL shutdown.

    A GRPO + vLLM-colocate run initializes a process group; without this, exit logs a
    ``destroy_process_group() was not called ... can leak resources`` warning and the
    container teardown can stall. Guarded + lazy so the launcher still imports on a
    box without torch, and is a no-op when no group was initialized.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _prepare_tokenizer(tokenizer: Any) -> None:
    """Default the pad token to EOS and left-pad for batched generation.

    v1 trains the base model on a raw (non-chat) prompt, so its **native** EOS
    (``<|endoftext|>`` for Qwen2.5-Math) is the correct terminator — and the one
    vLLM colocate stops on, since it loads the model's own ``generation_config``.
    We deliberately do NOT override EOS to ``<|im_end|>``: that would desync the
    trainer's stop token from vLLM's, and with ``mask_truncated_completions`` it
    silently masks the gradient. A v2 chat-template arm that wants ``<|im_end|>``
    must also pass ``stop_token_ids`` to vLLM via ``generation_kwargs``.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
