"""Modal app: run a llm-grpo-gains GRPO arm on a single A100.

    modal run modal_app.py --arm configs/correct.yaml          # launch a run
    modal run modal_app.py --arm configs/correct.yaml --smoke-problems 8   # day-1 smoke
    modal shell modal_app.py                                    # interactive debugging

A 1.5B model fits comfortably on one A100 80GB with colocated vLLM. Checkpoints and
provenance persist to a Modal Volume; training curves go to Weights & Biases.

Run it with the `modal` CLI (`uv tool install modal`), not as a project dependency.
Requires a Modal account and a `wandb` secret (`modal secret create wandb
WANDB_API_KEY=...`); it touches no existing Modal workspace.

This is a starting point: CUDA / torch / vLLM / TRL versions must agree, so expect
to pin them against the box on the first run (the proposal's "re-verify at build").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import modal

from llm_grpo_gains.eval.completions import SamplingConfig
from llm_grpo_gains.eval.registry import SETS, get_task_profile
from llm_grpo_gains.prompts import EVAL_MAX_NEW_TOKENS
from llm_grpo_gains.provenance import git_commit, git_is_dirty
from llm_grpo_gains.schemas import ProblemSet
from llm_grpo_gains.train.checkpoints import (
    final_or_selected_checkpoint_path,
    require_selected_checkpoint_path,
)

APP_NAME = "llm-grpo-gains"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
RUNS_DIR = "/runs"
# Where the project tree is mounted inside the image (arbitrary; kept off the volume).
REMOTE_ROOT = "/root/llm-grpo-gains"

app = modal.App(APP_NAME)

# Bake the GPU stack + the project into the image. For faster iteration you can
# instead bake only the deps and mount the source at runtime.
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


def _load_run_provenance(run_dir: Path):
    from llm_grpo_gains.train.provenance import RunProvenance

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
) -> Path:
    from llm_grpo_gains.eval.completions import write_completion_set
    from llm_grpo_gains.eval.generate import generate_completion_set

    completion_set = generate_completion_set(
        model_ref,
        problems,
        config,
        backend="vllm",
        model_revision=model_revision,
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
        problems = job.problems if job.problems is not None else SETS[job.set_name]()
        out = _generate_completion_set_to_volume(
            model_ref=job.model_ref,
            problems=problems,
            config=job.config,
            out_dir=out_root / f"{job.arm}__{job.set_name}",
            commit=commit,
            dirty=dirty,
            model_revision=job.model_revision,
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

    from llm_grpo_gains.train.config import load_arm_config
    from llm_grpo_gains.train.launcher import launch

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

    from llm_grpo_gains.eval.heldout import run_heldout_curve, write_selected_provenance
    from llm_grpo_gains.train.config import load_arm_config

    arm = load_arm_config(Path(arm_yaml))
    run_dir = Path(RUNS_DIR) / f"{arm.name}-seed{arm.seed}"
    config = SamplingConfig(temperature=0.0, n=1, max_new_tokens=EVAL_MAX_NEW_TOKENS, seed=0)
    curve = run_heldout_curve(run_dir, config, backend="vllm")
    out = run_dir / "heldout.json"
    out.write_text(
        json.dumps(curve.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
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

    from llm_grpo_gains.train.config import load_arm_config

    profile = get_task_profile(task)
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(role: str) -> str:
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}{role}-seed{seed}"
        prov = _load_run_provenance(run_dir)
        return final_or_selected_checkpoint_path(
            run_dir, prov.selected_checkpoint, prov.checkpoint_selection
        )

    if scope == "full":
        matrix = [
            (
                "base",
                base.base_model,
                base.base_model_revision,
                [profile.task_set, *profile.control_sets],
            ),
            (
                "correct",
                selected_checkpoint("correct"),
                None,
                [profile.task_set, *profile.control_sets],
            ),
            ("random", selected_checkpoint("random"), None, [profile.task_set]),
        ]
    elif scope == "placebo":
        matrix = [
            ("correct", selected_checkpoint("correct"), None, [profile.task_set]),
            ("random", selected_checkpoint("random"), None, [profile.task_set]),
        ]
    elif scope == "controls":
        if not profile.control_sets:
            raise ValueError(f"scope 'controls' needs control sets; task {task!r} has none")
        matrix = [("correct", selected_checkpoint("correct"), None, list(profile.control_sets))]
    else:
        raise ValueError(f"scope must be 'full', 'placebo', or 'controls', got {scope!r}")

    config = SamplingConfig(
        temperature=0.0, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=1, seed=0
    )
    out_root = Path(RUNS_DIR) / (
        profile.battery_root if seed == 0 else f"{profile.battery_root}-seed{seed}"
    )

    jobs = [
        _CompletionJob(arm, model_ref, revision, set_name, config)
        for arm, model_ref, revision, set_names in matrix
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
    """Pass@k elicitation panel: base + correct (seed 0) on the task set, n=8 sampled.

    Probes whether the RL gain is *new* capability or base capability surfaced, by
    comparing base pass@8 to correct pass@1/pass@8. Writes `CompletionSet`s to
    ``<RUNS_DIR>/elicitation/<arm>__gsm8k-test``; score with `grpo-decomp battery --k 1 8`.
    Sampled (temperature>0) because pass@k needs diverse draws, unlike the greedy battery.
    """
    from pathlib import Path

    from llm_grpo_gains.train.config import load_arm_config

    profile = get_task_profile(task)
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(role: str) -> str:
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}{role}-seed0"
        prov = _load_run_provenance(run_dir)
        return require_selected_checkpoint_path(run_dir, prov.selected_checkpoint)

    config = SamplingConfig(
        temperature=0.7, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=8, seed=0
    )
    matrix = [
        ("base", base.base_model, base.base_model_revision),
        ("correct", selected_checkpoint("correct"), None),
    ]
    out_root = Path(RUNS_DIR) / profile.elicitation_root

    jobs = [
        _CompletionJob(arm, model_ref, revision, profile.task_set, config)
        for arm, model_ref, revision in matrix
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

    Generalizes `elicitation` (seed 0, n=8) so the elicitation/expansion verdict does not
    rest on a single seed. The base anchor (seed-independent) is sampled once at `n_base`;
    every correct training seed is sampled at `n_correct`. Decoding is byte-for-byte the
    published panel's — `temperature=0.7, top_p=1.0, max_new_tokens=1024` — so only `n` and
    the checkpoint change. Writes `CompletionSet`s to
    ``<RUNS_DIR>/passk-multiseed[-<task>]/<arm>__<set>`` (``base`` + ``correct-seed<N>``);
    score with `grpo-decomp report-passk-seeds --task-set <set>`.

    `set_name` re-runs the same checkpoints off a different eval distribution (a control set
    like `gsm-symbolic`/`gsm8k-platinum`) to decontaminate the pass@8 verdict; the default is
    the task set. `limit` takes a deterministic `dev_slice` of that set (mirrors `grpo-decomp
    generate --limit`) so an oversized control can be matched to the task set's size. The set
    suffix keeps decontam cells disjoint from the published `__<task-set>` cells in one dir.
    """
    from pathlib import Path

    from llm_grpo_gains.data import dev_slice
    from llm_grpo_gains.train.config import load_arm_config

    profile = get_task_profile(task)
    eval_set = set_name or profile.task_set
    if eval_set not in SETS:
        raise ValueError(f"unknown set {eval_set!r}; known sets are {tuple(sorted(SETS))}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    seeds = [int(s) for s in correct_seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise ValueError(f"no correct seeds parsed from {correct_seeds!r}")
    base = load_arm_config(Path(profile.base_config))

    def selected_checkpoint(seed: int) -> str:
        run_dir = Path(RUNS_DIR) / f"{profile.run_prefix}correct-seed{seed}"
        prov = _load_run_provenance(run_dir)
        return final_or_selected_checkpoint_path(
            run_dir, prov.selected_checkpoint, prov.checkpoint_selection
        )

    out_root = Path(RUNS_DIR) / profile.passk_multiseed_root
    problems = SETS[eval_set]()
    if limit is not None:
        problems = dev_slice(problems, n=limit, seed=0)

    def sampled_config(n: int) -> SamplingConfig:
        return SamplingConfig(
            temperature=0.7, top_p=1.0, max_new_tokens=EVAL_MAX_NEW_TOKENS, n=n, seed=0
        )

    jobs = [
        _CompletionJob(
            "base",
            base.base_model,
            base.base_model_revision,
            eval_set,
            sampled_config(n_base),
            problems,
        ),
        *(
            _CompletionJob(
                f"correct-seed{seed}",
                selected_checkpoint(seed),
                None,
                eval_set,
                sampled_config(n_correct),
                problems,
            )
            for seed in seeds
        ),
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

    Every command this entrypoint dispatches is a long GPU run (6-24 h) whose deliverable is
    written to the `assay-runs` Volume, not the inline return value — so `spawn` DEFAULTS TO
    TRUE: the function fires server-side and the entrypoint returns its FunctionCall id without
    blocking. Pair the default with `--detach` for the durable, disconnect-proof run:

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

    Why spawn-by-default: a synchronous `.remote()` in a detached app can be canceled when
    the local client disconnects (Modal's own guidance), so a long training/eval run can show
    wandb "finished" yet lose its final checkpoint. Spawn fires the function and returns its
    FunctionCall id without blocking, so it runs server-side independent of the client; monitor
    via `modal app logs llm-grpo-gains` or the `assay-runs` Volume. Spawn REQUIRES `--detach`
    (without it the ephemeral app, and the spawned function, stop when this entrypoint returns).

    For the short day-1 smoke — where you DO want the inline result and don't need detach —
    force blocking with Modal's auto-generated `--no-spawn`:

    dry run:     modal run modal_app.py --no-spawn --arm configs/correct.yaml \\
                   --smoke-problems 8 --max-steps 5

    Use `--no-spawn` likewise for any quick `&&`-chained step that must finish before the next.
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
