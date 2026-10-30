"""Modal app: run a grpo-decomp GRPO arm on a single A100.

    modal run --detach modal_app.py --arm configs/correct.yaml --approved   # launch a run
    modal run modal_app.py --no-spawn --arm configs/correct.yaml --smoke-problems 8 --approved
    modal shell modal_app.py                                    # inspect the image

Every run spends money on an A100, so each entry point refuses to start without `--approved`.

A 1.5B model fits comfortably on one A100 80GB with colocated vLLM. Checkpoints and
provenance persist to a Modal Volume; training curves go to Weights & Biases.

Run it with the `modal` CLI (`uv tool install modal`), not as a project dependency.
Requires a Modal account and a `wandb` secret (`modal secret create wandb
WANDB_API_KEY=...`). CUDA, torch, vLLM, and TRL versions must remain compatible.
"""

from __future__ import annotations

import inspect
import shlex
from pathlib import Path

import modal

from grpo_decomp.eval.completions import SamplingConfig
from grpo_decomp.jobs import (
    CompletionJob,
    plan_battery,
    plan_passk,
    validate_battery_request,
    validate_passk_request,
)
from grpo_decomp.plugins import load_plugins
from grpo_decomp.provenance import git_commit, git_is_dirty
from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, EVAL_SETS
from grpo_decomp.schemas import ProblemSet, record_json
from grpo_decomp.train.provenance import resolve_run_dir

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
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.13")
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

LAUNCH_APPROVAL_FLAG = "--approved"
# A direct `modal run modal_app.py::<function>` skips the entrypoint, so remote functions gate too.
_DIRECT_CALL_COMMAND = f"modal run --detach modal_app.py <same arguments> {LAUNCH_APPROVAL_FLAG}"


def _require_approval(approved: bool, command: str) -> None:
    if not approved:
        raise SystemExit(
            "Refusing to launch: this Modal GPU run spends money. "
            "Stop and ask Adam to approve it.\n"
            f"Once approved, run: {command}"
        )


def _generate_completion_set_to_volume(
    *,
    model_ref: str,
    problems: ProblemSet,
    config: SamplingConfig,
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
    jobs: list[CompletionJob],
    *,
    out_root: Path,
    commit: str | None,
    dirty: bool | None,
) -> None:
    for job in jobs:
        out = out_root / f"{job.arm}__{job.set_name}"
        if not out.resolve().is_relative_to(out_root.resolve()):
            raise ValueError(f"completion output escapes root: {out}")
        if out.exists():
            raise ValueError(f"completion output already exists: {out}")
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
    arm_config_json: str | None = None,
    approved: bool = False,
) -> str:
    """Run one arm on the GPU; persist checkpoints + provenance to the Volume.

    `commit`/`dirty` are computed by the local entrypoint (the container has no .git)
    and recorded in provenance so a GPU run is traceable to the code that produced it.
    """
    _require_approval(approved, _DIRECT_CALL_COMMAND)
    from pathlib import Path

    from grpo_decomp.train.config import ArmConfig, load_arm_config
    from grpo_decomp.train.launcher import launch

    arm = (
        ArmConfig.model_validate_json(arm_config_json)
        if arm_config_json is not None
        else load_arm_config(Path(arm_yaml))
    )
    try:
        run_dir = launch(
            arm,
            output_root=Path(RUNS_DIR),
            smoke_problems=smoke_problems,
            max_steps=max_steps,
            commit=commit,
            dirty=dirty,
        )
    finally:
        runs.commit()
    return str(run_dir)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def heldout_arm(arm_yaml: str, approved: bool = False) -> str:
    """Held-out accuracy curve over a finished arm's checkpoints (check this, not reward)."""
    _require_approval(approved, _DIRECT_CALL_COMMAND)
    from pathlib import Path

    from grpo_decomp.eval.heldout import run_heldout_curve
    from grpo_decomp.train.config import load_arm_config

    arm = load_arm_config(Path(arm_yaml))
    run_dir = resolve_run_dir(Path(RUNS_DIR) / f"{arm.name}-seed{arm.seed}")
    curve = run_heldout_curve(run_dir, backend="vllm")
    out = run_dir / "heldout.json"
    out.write_text(record_json(curve), encoding="utf-8")
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
    task: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    approved: bool = False,
) -> str:
    """Generate decomposition completions for one seed; pass@1 greedy.

    ``scope="full"`` (seed 0): base/correct/random over the task set + 3 control sets —
    the inputs for the seed-aggregated reports. ``scope="placebo"`` (replicate seeds):
    correct + random on the task set only, the pre-registered confirmatory metric
    a replicate needs (base is seed-independent, reused from seed 0).
    ``scope="random-vs-base"``: the random arm on the task set, plus the base at seed 0 —
    the inputs for ``report-arm-vs-base`` when an arm has no correct twin. Output goes to
    ``<RUNS_DIR>/battery`` for seed 0, else ``<RUNS_DIR>/battery-seed<seed>``. Arm
    trained arms use their final checkpoints.
    `commit`/`dirty` come from the local entrypoint (no .git in the image) so each
    artifact is traceable to its code.
    """
    _require_approval(approved, _DIRECT_CALL_COMMAND)
    out_root, jobs = plan_battery(Path(RUNS_DIR), seed=seed, scope=scope, task=task)
    _run_completion_jobs(jobs, out_root=out_root, commit=commit, dirty=dirty)
    return str(out_root)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={RUNS_DIR: runs},
    timeout=6 * 60 * 60,
)
def passk_multiseed(
    task: str | None = None,
    n_base: int = 16,
    n_correct: int = 8,
    correct_seeds: str = "0,1,2,3,4,5",
    set_name: str | None = None,
    limit: int | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
    approved: bool = False,
) -> str:
    """Multi-seed pass@k coverage panel: base anchor once + each correct training seed.

    The base anchor is seed-independent and sampled once at `n_base`; every correct
    training seed is sampled at `n_correct`. Decoding uses
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
    _require_approval(approved, _DIRECT_CALL_COMMAND)
    out_root, jobs = plan_passk(
        Path(RUNS_DIR),
        task=task,
        n_base=n_base,
        n_correct=n_correct,
        correct_seeds=correct_seeds,
        set_name=set_name,
        limit=limit,
    )
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
    task: str | None = None,
    n_base: int = 16,
    n_correct: int = 8,
    correct_seeds: str = "0,1,2,3,4,5",
    set_name: str | None = None,
    limit: int | None = None,
    spawn: bool = True,
    approved: bool = False,
) -> None:
    """Train an arm, score its held-out curve, or generate eval completions.

    Every command this entrypoint dispatches is a long GPU run whose output is written to
    the `assay-runs` Volume, not the inline return value. `spawn` defaults to true so the
    function starts server-side and the entrypoint returns its FunctionCall id without
    blocking. Pair the default with `--detach`. Every launch needs `--approved` (it spends money):

    full train:  modal run --detach modal_app.py --approved --arm configs/correct.yaml
    held-out:    modal run --detach modal_app.py --approved --arm configs/correct.yaml \\
                   --command heldout
    battery:     modal run --detach modal_app.py --approved --command battery  # seed 0, full
    placebo:     modal run --detach modal_app.py --approved --command battery --seed 1 \\
                   --scope placebo
    controls:    modal run --detach modal_app.py --approved --command battery --seed 1 \\
                   --scope controls
    passk-seeds: modal run --detach modal_app.py --approved --command passk-multiseed
    escalated:   ...same, plus --n-base 32 --n-correct 16 (or --task countdown)
    decontam:    ...same, plus --set-name gsm-symbolic --limit 1319 (or --set-name gsm8k-platinum)
    countdown:   modal run --detach modal_app.py --approved --command battery --task countdown \\
                   --scope placebo --seed 1

    `--task` (gsm8k default, or countdown) selects the eval wiring for generation:
    the base arm config, task set, control sets (none for Countdown), and run-dir prefix.

    A synchronous `.remote()` in a detached app can be canceled when the local client
    disconnects, so long runs use `spawn`. Monitor them via `modal app logs grpo-decomp`
    or the `assay-runs` Volume. Spawn requires `--detach`; otherwise the ephemeral app
    stops when this entrypoint returns.

    For a short blocking check, use Modal's auto-generated `--no-spawn`:

    short check: modal run modal_app.py --approved --no-spawn --arm configs/correct.yaml \\
                   --smoke-problems 8 --max-steps 5

    Use `--no-spawn` for any chained step that must finish before the next one starts.
    """
    # Computed here (locally) because the image ignores .git, so the container can't
    # read git - pass the real commit/dirty into provenance.
    launch_args = dict(locals())
    from grpo_decomp.train.config import load_arm_config
    from grpo_decomp.train.launcher import preflight_arm

    if command == "train":
        configured_arm = preflight_arm(
            load_arm_config(Path(arm)), smoke_problems=smoke_problems, max_steps=max_steps
        )
    elif command == "heldout":
        load_arm_config(Path(arm))
    elif command == "battery":
        validate_battery_request(task, scope, seed)
    elif command == "passk-multiseed":
        validate_passk_request(task, n_base, n_correct, correct_seeds, set_name, limit)
    _require_approval(approved, _approved_command(launch_args))
    commit, dirty = git_commit(), git_is_dirty()
    if command == "train":
        fn, kwargs = (
            train_arm,
            {
                "arm_yaml": arm,
                "arm_config_json": configured_arm.model_dump_json(),
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
    elif command == "passk-multiseed":
        fn, kwargs = (
            passk_multiseed,
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
            f"command must be 'train', 'heldout', 'battery', or 'passk-multiseed', got {command!r}"
        )

    kwargs["approved"] = True
    if spawn:
        # Fire-and-return so the work survives a client disconnect (use with --detach).
        # The deliverable lands on the Volume, not here; print the monitor hints.
        call = fn.spawn(**kwargs)
        print(f"{command} dispatched (spawn): function call {call.object_id}")
        print(f"  monitor: modal app logs {APP_NAME}   |   results: modal volume ls {runs.name}")
    else:
        result = fn.remote(**kwargs)
        print(f"{command} complete: {result}")


def _approved_command(launch_args: dict[str, object]) -> str:
    defaults = inspect.signature(main).parameters
    parts = ["modal run", "--detach" if launch_args["spawn"] else "", "modal_app.py"]
    for name, value in launch_args.items():
        if name == "approved" or value == defaults[name].default:
            continue
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            parts.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            parts.append(f"--{flag} {shlex.quote(str(value))}")
    return " ".join(part for part in [*parts, LAUNCH_APPROVAL_FLAG] if part)
