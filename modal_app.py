"""Modal app: run a grpo-decomp GRPO arm on a single A100.

    modal run modal_app.py --arm configs/correct.yaml          # launch a run
    modal run modal_app.py --arm configs/correct.yaml --smoke-problems 8   # short check
    modal shell modal_app.py                                    # inspect the image

A 1.5B model fits comfortably on one A100 80GB with colocated vLLM. Checkpoints and
provenance persist to a Modal Volume; training curves go to Weights & Biases.

Run it with the `modal` CLI (`uv tool install modal`), not as a project dependency.
Requires a Modal account and a `wandb` secret (`modal secret create wandb
WANDB_API_KEY=...`). CUDA, torch, vLLM, and TRL versions must remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import modal

from grpo_decomp.eval.completions import SamplingConfig
from grpo_decomp.plugins import load_plugins
from grpo_decomp.prompts import EVAL_MAX_NEW_TOKENS
from grpo_decomp.provenance import git_commit, git_is_dirty
from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, EVAL_SETS, get_task_profile
from grpo_decomp.schemas import ProblemSet, record_json
from grpo_decomp.train.checkpoints import (
    final_or_selected_checkpoint_path,
    require_selected_checkpoint_path,
)

# Populate the harness registries with the study's eval sets and task profiles, both
# locally (entrypoint arg handling) and inside each Modal container (module import).
load_plugins()

APP_NAME = "grpo-decomp"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
RUNS_DIR = "/runs"
# Where the project tree is mounted inside the image (arbitrary; kept off the volume).
REMOTE_ROOT = "/root/grpo-decomp"

app = modal.App(APP_NAME)

# Transient local caches: never part of the image, and they mutate during a build
# (a concurrent test run rewrites .pytest_cache), tripping Modal's "modified during
# build" guard. .git is stripped too (the entrypoint passes git state in explicitly).
_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/runs",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.mypy_cache",
    "**/.DS_Store",
]

# Two layers on purpose. The heavy dependency install (layer 1) is keyed only on
# pyproject.toml + uv.lock, so editing source re-runs only the fast editable relink
# (layer 2), never the ~5GB GPU-stack download. `uv export` emits the
# train extra's pinned deps without the project itself; the project is linked
# separately with --no-deps once the full source is present.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git")
    .pip_install("uv")
    .add_local_file(
        Path(__file__).parent / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file(Path(__file__).parent / "uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .workdir(REMOTE_ROOT)
    .run_commands(
        "uv export --frozen --extra train --no-emit-project -o /tmp/reqs.txt",
        "uv pip install --system -r /tmp/reqs.txt",
    )
    # Anchored to this file, not the CWD, so `modal run` works from any directory.
    .add_local_dir(Path(__file__).parent, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .run_commands("uv pip install --system --no-deps -e .")
)

runs = modal.Volume.from_name("assay-runs", create_if_missing=True)


@dataclass(frozen=True)
class _CompletionJob:
    arm: str
    model_ref: str
    model_revision: str | None
    set_name: str
    config: SamplingConfig
    problems: ProblemSet | None = None
    # The prompt strategy this arm trained on; eval MUST match it or the decomposition
    # compares a model against an off-distribution prompt.
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY


def _load_run_provenance(run_dir: Path):
    from grpo_decomp.train.provenance import RunProvenance

    return RunProvenance.model_validate_json(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )


def _generate_completion_set_to_volume(
    *,
    model_ref: str,
    problems,
    config,
    out_dir: Path,
    commit: str | None,
    dirty: bool | None,
    model_revision: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
) -> Path:
    from grpo_decomp.eval.completions import write_completion_set
    from grpo_decomp.eval.generate import generate_completion_set

    completion_set = generate_completion_set(
        model_ref,
        problems,
        config,
        backend="vllm",
        model_revision=model_revision,
        prompt_strategy=prompt_strategy,
        commit=commit,
        dirty=dirty,
    )
    out = write_completion_set(completion_set, out_dir)
    runs.commit()
    return out


def _run_completion_jobs(
    jobs: list[_CompletionJob],
    *,
    out_root: Path,
    commit: str | None,
    dirty: bool | None,
) -> None:
    for job in jobs:
        problems = job.problems if job.problems is not None else EVAL_SETS[job.set_name]()
        out = _generate_completion_set_to_volume(
            model_ref=job.model_ref,
            problems=problems,
            config=job.config,
            out_dir=out_root / f"{job.arm}__{job.set_name}",
            commit=commit,
            dirty=dirty,
            model_revision=job.model_revision,
            prompt_strategy=job.prompt_strategy,
        )
        print(f"  wrote {job.arm}__{job.set_name}: n={job.config.n} -> {out}")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    # wandb is required (curves). huggingface is optional — Modal has no required=False,
    # so an absent secret would block the run; Qwen + GSM8K are public, so anonymous HF
    # works (just rate-limited). To authenticate, create a `huggingface` secret with
    # HF_TOKEN and add `modal.Secret.from_name("huggingface")` here.
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    timeout=24 * 60 * 60,
)
def train_arm(
    arm_yaml: str,
    smoke_problems: int | None = None,
    max_steps: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Run one arm on the GPU; persist checkpoints + provenance to the Volume.

    `commit`/`dirty` are computed by the local entrypoint (the container has no .git)
    and recorded in provenance so a GPU run is traceable to the code that produced it.
    """
    from pathlib import Path

    from grpo_decomp.train.config import load_arm_config
    from grpo_decomp.train.launcher import launch

    arm = load_arm_config(Path(arm_yaml))
    run_dir = launch(
        arm,
        output_root=Path(RUNS_DIR),
        smoke_problems=smoke_problems,
        max_steps=max_steps,
        commit=commit,
        dirty=dirty,
    )
    runs.commit()
    return str(run_dir)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def heldout_arm(arm_yaml: str) -> str:
    """Held-out accuracy curve over a finished arm's checkpoints (check this, not reward)."""
    from pathlib import Path

    from grpo_decomp.eval.heldout import run_heldout_curve, write_selected_provenance
    from grpo_decomp.train.config import load_arm_config

    arm = load_arm_config(Path(arm_yaml))
    run_dir = Path(RUNS_DIR) / f"{arm.name}-seed{arm.seed}"
    config = SamplingConfig(temperature=0.0, n=1, max_new_tokens=EVAL_MAX_NEW_TOKENS, seed=0)
    curve = run_heldout_curve(run_dir, config, backend="vllm")
    out = run_dir / "heldout.json"
    out.write_text(record_json(curve), encoding="utf-8")
    write_selected_provenance(run_dir, curve)
    runs.commit()
    return str(out)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def eval_matrix(
    seed: int = 0,
    scope: str = "full",
    task: str = "gsm8k",
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Generate decomposition completions for one seed; pass@1 greedy.

    ``scope="full"`` (seed 0): base/correct/random over the task set + 3 control sets —
    the layout `grpo-decomp report` (CPU, offline) consumes. ``scope="placebo"`` (replicate
    seeds): correct + random on the task set only, the pre-registered confirmatory metric
    a replicate needs (base is seed-independent, reused from seed 0). Output goes to
    ``<RUNS_DIR>/battery`` for seed 0, else ``<RUNS_DIR>/battery-seed<seed>``. Arm
    checkpoints are the ones the pre-registered rule selected (read from provenance).
    `commit`/`dirty` come from the local entrypoint (no .git in the image) so each
    artifact is traceable to its code.
    """
    from pathlib import Path

    from grpo_decomp.train.config import load_arm_config

    profile = get_task_profile(task)
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(role: str) -> tuple[str, str]:
        """Return (checkpoint path, the prompt strategy that run trained on)."""
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}{role}-seed{seed}"
        prov = _load_run_provenance(run_dir)
        checkpoint = final_or_selected_checkpoint_path(
            run_dir, prov.selected_checkpoint, prov.checkpoint_selection
        )
        return checkpoint, prov.prompt_strategy

    base_row = ("base", base.base_model, base.base_model_revision, base.prompt_strategy)

    def trained_row(role: str) -> tuple[str, str, None, str]:
        checkpoint, strategy = selected_checkpoint(role)
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
    elif scope == "controls":
        if not profile.control_sets:
            raise ValueError(f"scope 'controls' needs control sets; task {task!r} has none")
        matrix = [(*trained_row("correct"), list(profile.control_sets))]
    else:
        raise ValueError(f"scope must be 'full', 'placebo', or 'controls', got {scope!r}")

    config = SamplingConfig(
        temperature=0.0, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=1, seed=0
    )
    out_root = Path(RUNS_DIR) / (
        profile.battery_root if seed == 0 else f"{profile.battery_root}-seed{seed}"
    )

    jobs = [
        _CompletionJob(arm, model_ref, revision, set_name, config, prompt_strategy=strategy)
        for arm, model_ref, revision, strategy, set_names in matrix
        for set_name in set_names
    ]
    _run_completion_jobs(jobs, out_root=out_root, commit=commit, dirty=dirty)
    return str(out_root)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def elicitation(task: str = "gsm8k", commit: str | None = None, dirty: bool | None = None) -> str:
    """Pass@k coverage panel: base + correct (seed 0) on the task set, n=8 sampled.

    Checks whether the RL gain is *new* capability or base capability surfaced, by
    comparing base pass@8 to correct pass@1/pass@8. Writes `CompletionSet`s to
    ``<RUNS_DIR>/elicitation/<arm>__gsm8k-test``; score with `grpo-decomp battery --k 1 8`.
    Sampled (temperature>0) because pass@k needs diverse draws, unlike the greedy battery.
    """
    from pathlib import Path

    from grpo_decomp.train.config import load_arm_config

    profile = get_task_profile(task)
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(role: str) -> tuple[str, str]:
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}{role}-seed0"
        prov = _load_run_provenance(run_dir)
        checkpoint = require_selected_checkpoint_path(run_dir, prov.selected_checkpoint)
        return checkpoint, prov.prompt_strategy

    config = SamplingConfig(
        temperature=0.7, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=8, seed=0
    )
    correct_checkpoint, correct_strategy = selected_checkpoint("correct")
    matrix = [
        ("base", base.base_model, base.base_model_revision, base.prompt_strategy),
        ("correct", correct_checkpoint, None, correct_strategy),
    ]
    out_root = Path(RUNS_DIR) / profile.elicitation_root

    jobs = [
        _CompletionJob(arm, model_ref, revision, profile.task_set, config, prompt_strategy=strategy)
        for arm, model_ref, revision, strategy in matrix
    ]
    _run_completion_jobs(jobs, out_root=out_root, commit=commit, dirty=dirty)
    return str(out_root)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def elicitation_multiseed(
    task: str = "gsm8k",
    n_base: int = 16,
    n_correct: int = 8,
    correct_seeds: str = "0,1,2,3,4,5",
    set_name: str | None = None,
    limit: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
) -> str:
    """Multi-seed pass@k coverage panel: base anchor once + each correct training seed.

    Extends `elicitation` beyond seed 0. The base anchor is seed-independent and sampled
    once at `n_base`; every correct training seed is sampled at `n_correct`. Decoding uses
    the published panel settings: `temperature=0.7, top_p=1.0, max_new_tokens=1024`.
    Writes `CompletionSet`s to
    ``<RUNS_DIR>/passk-multiseed[-<task>]/<arm>__<set>`` (``base`` + ``correct-seed<N>``);
    score with `grpo-decomp report-passk-seeds --task-set <set>`.

    `set_name` re-runs the same checkpoints off a different eval distribution (a control set
    like `gsm-symbolic`/`gsm8k-platinum`) to decontaminate the pass@8 verdict; the default is
    the task set. `limit` takes a deterministic `dev_slice` of that set so an oversized
    control can be matched to the task set's size. The set suffix keeps control panels
    disjoint from the task-set cells in one dir.
    """
    from pathlib import Path

    from grpo_decomp.splits import dev_slice
    from grpo_decomp.train.config import load_arm_config

    profile = get_task_profile(task)
    eval_set = set_name or profile.task_set
    if eval_set not in EVAL_SETS:
        raise ValueError(f"unknown set {eval_set!r}; known sets are {tuple(sorted(EVAL_SETS))}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    seeds = [int(s) for s in correct_seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise ValueError(f"no correct seeds parsed from {correct_seeds!r}")
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(seed: int) -> tuple[str, str]:
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}correct-seed{seed}"
        prov = _load_run_provenance(run_dir)
        checkpoint = final_or_selected_checkpoint_path(
            run_dir, prov.selected_checkpoint, prov.checkpoint_selection
        )
        return checkpoint, prov.prompt_strategy

    out_root = Path(RUNS_DIR) / profile.passk_multiseed_root
    problems = EVAL_SETS[eval_set]()
    if limit is not None:
        problems = dev_slice(problems, n=limit, seed=0)

    def sampled_config(n: int) -> SamplingConfig:
        return SamplingConfig(
            temperature=0.7, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=n, seed=0
        )

    def correct_job(seed: int) -> _CompletionJob:
        checkpoint, strategy = selected_checkpoint(seed)
        return _CompletionJob(
            f"correct-seed{seed}",
            checkpoint,
            None,
            eval_set,
            sampled_config(n_correct),
            problems,
            prompt_strategy=strategy,
        )

    jobs = [
        _CompletionJob(
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
    _run_completion_jobs(jobs, out_root=out_root, commit=commit, dirty=dirty)
    return str(out_root)


@app.local_entrypoint()
def main(
    arm: str = "configs/correct.yaml",
    command: str = "train",
    smoke_problems: int | None = None,
    max_steps: int | None = None,
    seed: int = 0,
    scope: str = "full",
    task: str = "gsm8k",
    n_base: int = 16,
    n_correct: int = 8,
    correct_seeds: str = "0,1,2,3,4,5",
    set_name: str | None = None,
    limit: int | None = None,
    spawn: bool = True,
) -> None:
    """Train an arm, score its held-out curve, or generate eval completions.

    Every command this entrypoint dispatches is a long GPU run whose output is written to
    the `assay-runs` Volume, not the inline return value. `spawn` defaults to true so the
    function starts server-side and the entrypoint returns its FunctionCall id without
    blocking. Pair the default with `--detach`:

    full train:  modal run --detach modal_app.py --arm configs/correct.yaml
    held-out:    modal run --detach modal_app.py --arm configs/correct.yaml --command heldout
    battery:     modal run --detach modal_app.py --command battery               # seed 0, full
    placebo:     modal run --detach modal_app.py --command battery --seed 1 --scope placebo
    controls:    modal run --detach modal_app.py --command battery --scope controls --seed 1
    elicitation: modal run --detach modal_app.py --command elicitation
    passk-seeds: modal run --detach modal_app.py --command elicitation-multiseed
    escalated:   ...same, plus --n-base 32 --n-correct 16 (or --task countdown)
    decontam:    ...same, plus --set-name gsm-symbolic --limit 1319 (or --set-name gsm8k-platinum)
    countdown:   modal run --detach modal_app.py --command battery --task countdown \\
                   --scope placebo --seed 1

    `--task` (gsm8k default, or countdown) selects the eval wiring for the battery /
    elicitation commands: the base arm config, the task set, the control sets (none for
    Countdown), and the `correct`/`random` run-dir prefix.

    A synchronous `.remote()` in a detached app can be canceled when the local client
    disconnects, so long runs use `spawn`. Monitor them via `modal app logs grpo-decomp`
    or the `assay-runs` Volume. Spawn requires `--detach`; otherwise the ephemeral app
    stops when this entrypoint returns.

    For a short blocking check, use Modal's auto-generated `--no-spawn`:

    short check: modal run modal_app.py --no-spawn --arm configs/correct.yaml \\
                   --smoke-problems 8 --max-steps 5

    Use `--no-spawn` for any chained step that must finish before the next one starts.
    """
    # Computed here (locally) because the image ignores .git, so the container can't
    # read git - pass the real commit/dirty into provenance.
    commit, dirty = git_commit(), git_is_dirty()
    if command == "train":
        fn, kwargs = (
            train_arm,
            {
                "arm_yaml": arm,
                "smoke_problems": smoke_problems,
                "max_steps": max_steps,
                "commit": commit,
                "dirty": dirty,
            },
        )
    elif command == "heldout":
        fn, kwargs = heldout_arm, {"arm_yaml": arm}  # preserves train-time provenance
    elif command == "battery":
        fn, kwargs = (
            eval_matrix,
            {
                "seed": seed,
                "scope": scope,
                "task": task,
                "commit": commit,
                "dirty": dirty,
            },
        )
    elif command == "elicitation":
        fn, kwargs = elicitation, {"task": task, "commit": commit, "dirty": dirty}
    elif command == "elicitation-multiseed":
        fn, kwargs = (
            elicitation_multiseed,
            {
                "task": task,
                "n_base": n_base,
                "n_correct": n_correct,
                "correct_seeds": correct_seeds,
                "set_name": set_name,
                "limit": limit,
                "commit": commit,
                "dirty": dirty,
            },
        )
    else:
        raise ValueError(
            "command must be 'train', 'heldout', 'battery', 'elicitation', or "
            f"'elicitation-multiseed', got {command!r}"
        )

    if spawn:
        # Fire-and-return so the work survives a client disconnect (use with --detach).
        # The deliverable lands on the Volume, not here; print the monitor hints.
        call = fn.spawn(**kwargs)
        print(f"{command} dispatched (spawn): function call {call.object_id}")
        print(f"  monitor: modal app logs {APP_NAME}   |   results: modal volume ls {runs.name}")
    else:
        result = fn.remote(**kwargs)
        print(f"{command} complete: {result}")
