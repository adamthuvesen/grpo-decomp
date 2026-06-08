"""Modal app: run a grpo-gain-decomposition GRPO arm on a single A100.

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

import subprocess
from pathlib import Path

import modal

APP_NAME = "grpo-gain-decomp"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
RUNS_DIR = "/runs"
# Where the project tree is mounted inside the image (arbitrary; kept off the volume).
REMOTE_ROOT = "/root/grpo-gain-decomposition"

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
# pyproject.toml + uv.lock, so editing source no longer re-downloads the ~5GB GPU
# stack — only the fast editable relink (layer 2) re-runs. `uv export` emits the
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


def _eval_task(task: str) -> tuple[str, str, list[str], str]:
    """Per-task eval wiring: (base arm-config path, task set, control sets, arm-name prefix).

    GSM8K carries the perturbation/clean-label controls and unprefixed `correct`/`random`
    run dirs; Countdown is a generated set with no controls and `countdown-`-prefixed runs.
    """
    if task == "countdown":
        return "configs/countdown-correct.yaml", "countdown-test", [], "countdown-"
    if task == "gsm8k":
        return (
            "configs/correct.yaml",
            "gsm8k-test",
            ["gsm-symbolic", "gsm-plus", "gsm8k-platinum"],
            "",
        )
    raise ValueError(f"eval task must be 'gsm8k' or 'countdown', got {task!r}")


def _selected_checkpoint_path(run_dir: Path, selected_checkpoint: str | None) -> str:
    """Return the pre-selected checkpoint path, or fail before evaluation can peek."""
    if selected_checkpoint is None:
        raise ValueError(
            f"{run_dir} has no selected checkpoint; run `modal run modal_app.py --command heldout` "
            "for this arm before battery or elicitation"
        )
    return str(run_dir / "checkpoints" / selected_checkpoint)


def _final_or_selected_checkpoint(run_dir: Path, selected_checkpoint: str | None, rule: str) -> str:
    """The checkpoint to evaluate for a finished arm.

    Prefer the recorded selection (a `heldout` run sets it). When it is unset, realize the
    deterministic ``final`` rule directly — no held-out curve is needed to know its answer,
    and ``checkpoints/final`` is what every published placebo/elicitation artifact used. Any
    other rule without a recorded selection is an explicit error (run `heldout` first). This
    lets the multi-seed panel reuse the replicate checkpoints (rule ``final``, selection unset)
    without an extra GPU pass.
    """
    if selected_checkpoint is None:
        if rule != "final":
            raise ValueError(
                f"{run_dir} has no selected checkpoint and rule is {rule!r}; run "
                "`modal run modal_app.py --command heldout` for this arm first"
            )
        selected_checkpoint = "final"
    return str(run_dir / "checkpoints" / selected_checkpoint)


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

    from grpo_gain_decomp.train.config import load_arm_config
    from grpo_gain_decomp.train.launcher import launch

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

    from grpo_gain_decomp.eval.cli import main as eval_main
    from grpo_gain_decomp.train.config import load_arm_config

    arm = load_arm_config(Path(arm_yaml))
    run_dir = Path(RUNS_DIR) / f"{arm.name}-seed{arm.seed}"
    status = eval_main(["heldout", "--run", str(run_dir), "--backend", "vllm"])
    if status:
        raise RuntimeError(f"held-out eval failed (exit {status}) for {run_dir}")
    runs.commit()
    return str(run_dir / "heldout.json")


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

    from grpo_gain_decomp.eval.cli import SETS
    from grpo_gain_decomp.eval.completions import (
        CompletionSet,
        ProblemCompletions,
        SamplingConfig,
        capture_generation_provenance,
        write_completion_set,
    )
    from grpo_gain_decomp.eval.generate import generate
    from grpo_gain_decomp.train.config import load_arm_config
    from grpo_gain_decomp.train.provenance import RunProvenance

    # Base model + revision come from the arm config (the source training used), so the
    # base eval is the same weights the gain is measured against.
    base_cfg, task_set, control_sets, prefix = _eval_task(task)
    base = load_arm_config(Path(base_cfg))

    def selected_checkpoint(role: str) -> str:
        run_dir = Path(RUNS_DIR) / f"{prefix}{role}-seed{seed}"
        prov = RunProvenance.model_validate_json(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        return _selected_checkpoint_path(run_dir, prov.selected_checkpoint)

    # full: base/correct span task + every control (control rows compare base vs correct);
    # random needs only the task set (placebo row is task-set only). placebo: just the
    # correct-vs-random confirmatory pair on the task set (all a replicate seed needs).
    if scope == "full":
        matrix = [
            ("base", base.base_model, base.base_model_revision, [task_set, *control_sets]),
            ("correct", selected_checkpoint("correct"), None, [task_set, *control_sets]),
            ("random", selected_checkpoint("random"), None, [task_set]),
        ]
    elif scope == "placebo":
        matrix = [
            ("correct", selected_checkpoint("correct"), None, [task_set]),
            ("random", selected_checkpoint("random"), None, [task_set]),
        ]
    else:
        raise ValueError(f"scope must be 'full' or 'placebo', got {scope!r}")
    # 1024 tokens matches training's max_completion_length, so eval isn't truncating
    # answers the model was trained to produce (the 512 held-out curve clips the tail).
    config = SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=1024, n=1, seed=0)
    base_out = "battery" if task == "gsm8k" else f"battery-{task}"
    out_root = Path(RUNS_DIR) / (base_out if seed == 0 else f"{base_out}-seed{seed}")

    for arm, model_ref, revision, sets in matrix:
        for set_name in sets:
            problems = SETS[set_name]()
            samples = generate(model_ref, problems, config, backend="vllm", model_revision=revision)
            items = tuple(
                ProblemCompletions(problem=problem, samples=tuple(samples[problem.id]))
                for problem in problems
            )
            provenance = capture_generation_provenance(
                model=model_ref,
                dataset=problems.source,
                sampling=config,
                backend="vllm",
                n_problems=len(problems),
                model_revision=revision,
                commit=commit,
                dirty=dirty,
            )
            out = write_completion_set(
                CompletionSet(provenance=provenance, items=items), out_root / f"{arm}__{set_name}"
            )
            print(f"  wrote {arm}__{set_name}: {len(items)} problems -> {out}")
        runs.commit()  # persist per arm so a late failure doesn't discard finished cells
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

    from grpo_gain_decomp.eval.cli import SETS
    from grpo_gain_decomp.eval.completions import (
        CompletionSet,
        ProblemCompletions,
        SamplingConfig,
        capture_generation_provenance,
        write_completion_set,
    )
    from grpo_gain_decomp.eval.generate import generate
    from grpo_gain_decomp.train.config import load_arm_config
    from grpo_gain_decomp.train.provenance import RunProvenance

    base_cfg, task_set, _controls, prefix = _eval_task(task)
    base = load_arm_config(Path(base_cfg))

    def selected_checkpoint(role: str) -> str:
        run_dir = Path(RUNS_DIR) / f"{prefix}{role}-seed0"
        prov = RunProvenance.model_validate_json(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        return _selected_checkpoint_path(run_dir, prov.selected_checkpoint)

    config = SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=1024, n=8, seed=0)
    matrix = [
        ("base", base.base_model, base.base_model_revision),
        ("correct", selected_checkpoint("correct"), None),
    ]
    out_root = Path(RUNS_DIR) / ("elicitation" if task == "gsm8k" else f"elicitation-{task}")
    problems = SETS[task_set]()

    for arm, model_ref, revision in matrix:
        samples = generate(model_ref, problems, config, backend="vllm", model_revision=revision)
        items = tuple(
            ProblemCompletions(problem=problem, samples=tuple(samples[problem.id]))
            for problem in problems
        )
        provenance = capture_generation_provenance(
            model=model_ref,
            dataset=problems.source,
            sampling=config,
            backend="vllm",
            n_problems=len(problems),
            model_revision=revision,
            commit=commit,
            dirty=dirty,
        )
        out = write_completion_set(
            CompletionSet(provenance=provenance, items=items), out_root / f"{arm}__{task_set}"
        )
        print(f"  wrote {arm}__{task_set}: {len(items)} problems x n={config.n} -> {out}")
        runs.commit()
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

    Generalizes `elicitation` (seed 0, n=8) so the elicitation/expansion verdict no longer
    rests on a single seed. The base anchor (seed-independent) is sampled once at `n_base`;
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

    from grpo_gain_decomp.data import dev_slice
    from grpo_gain_decomp.eval.cli import SETS
    from grpo_gain_decomp.eval.completions import (
        CompletionSet,
        ProblemCompletions,
        SamplingConfig,
        capture_generation_provenance,
        write_completion_set,
    )
    from grpo_gain_decomp.eval.generate import generate
    from grpo_gain_decomp.train.config import load_arm_config
    from grpo_gain_decomp.train.provenance import RunProvenance

    base_cfg, task_set, _controls, prefix = _eval_task(task)
    eval_set = set_name or task_set
    if eval_set not in SETS:
        raise ValueError(f"unknown set {eval_set!r}; known sets are {tuple(sorted(SETS))}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    seeds = [int(s) for s in correct_seeds.split(",") if s.strip() != ""]
    if not seeds:
        raise ValueError(f"no correct seeds parsed from {correct_seeds!r}")
    base = load_arm_config(Path(base_cfg))

    def selected_checkpoint(seed: int) -> str:
        run_dir = Path(RUNS_DIR) / f"{prefix}correct-seed{seed}"
        prov = RunProvenance.model_validate_json(
            (run_dir / "provenance.json").read_text(encoding="utf-8")
        )
        return _final_or_selected_checkpoint(
            run_dir, prov.selected_checkpoint, prov.checkpoint_selection
        )

    base_out = "passk-multiseed" if task == "gsm8k" else f"passk-multiseed-{task}"
    out_root = Path(RUNS_DIR) / base_out
    problems = SETS[eval_set]()
    if limit is not None:
        problems = dev_slice(problems, n=limit, seed=0)

    def run_arm(arm: str, model_ref: str, revision: str | None, n: int) -> None:
        # Same decoding as the published seed-0 panel; only n and the checkpoint vary.
        config = SamplingConfig(temperature=0.7, top_p=1.0, max_new_tokens=1024, n=n, seed=0)
        samples = generate(model_ref, problems, config, backend="vllm", model_revision=revision)
        items = tuple(
            ProblemCompletions(problem=problem, samples=tuple(samples[problem.id]))
            for problem in problems
        )
        provenance = capture_generation_provenance(
            model=model_ref,
            dataset=problems.source,
            sampling=config,
            backend="vllm",
            n_problems=len(problems),
            model_revision=revision,
            commit=commit,
            dirty=dirty,
        )
        out = write_completion_set(
            CompletionSet(provenance=provenance, items=items), out_root / f"{arm}__{eval_set}"
        )
        print(f"  wrote {arm}__{eval_set}: {len(items)} problems x n={n} -> {out}")
        runs.commit()  # persist per arm so a late failure doesn't discard finished cells

    run_arm("base", base.base_model, base.base_model_revision, n_base)
    for seed in seeds:
        run_arm(f"correct-seed{seed}", selected_checkpoint(seed), None, n_correct)
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
    spawn: bool = False,
) -> None:
    """Train an arm, score its held-out curve, or generate eval completions.

    full train:  modal run modal_app.py --arm configs/correct.yaml
    dry run:     modal run modal_app.py --arm configs/correct.yaml --smoke-problems 8 --max-steps 5
    held-out:    modal run modal_app.py --arm configs/correct.yaml --command heldout
    battery:     modal run modal_app.py --command battery                       # seed 0, full
    placebo:    modal run modal_app.py --command battery --seed 1 --scope placebo
    elicitation: modal run modal_app.py --command elicitation
    passk-seeds: modal run --detach modal_app.py --command elicitation-multiseed --spawn
    escalated:   ...same, plus --n-base 32 --n-correct 16 (or --task countdown)
    decontam:    ...same, plus --set-name gsm-symbolic --limit 1319 (or --set-name gsm8k-platinum)
    countdown:   modal run modal_app.py --command battery --task countdown --scope placebo --seed 1
    durable:     modal run --detach modal_app.py --arm <cfg> --spawn            # long runs

    `--task` (gsm8k default, or countdown) selects the eval wiring for the battery /
    elicitation commands: the base arm config, the task set, the control sets (none for
    Countdown), and the `correct`/`random` run-dir prefix.

    Long runs should use `--detach --spawn`. A synchronous `.remote()` in a detached app
    can be canceled when the local client disconnects (Modal's own guidance), so a long
    training/eval run can show wandb "finished" yet lose its final checkpoint. `--spawn`
    fires the function and returns its FunctionCall id without blocking, so it runs
    server-side independent of the client; monitor via `modal app logs` or the Volume.
    `--spawn` REQUIRES `--detach` (without it the ephemeral app, and the spawned function,
    stop when this entrypoint returns). The default (`.remote()`, blocking) is right for
    short runs and `&&`-chained commands.
    """
    # Computed here (locally) because the image ignores .git, so the container can't
    # read git - pass the real commit/dirty into provenance.
    commit, dirty = _local_git_commit(), _local_git_dirty()
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
        call = fn.spawn(**kwargs)
        print(f"{command} dispatched (spawn): function call {call.object_id}")
    else:
        result = fn.remote(**kwargs)
        print(f"{command} complete: {result}")


# Local git helpers (the entrypoint runs in the `modal` tool env, which can't import
# grpo_gain_decomp; and the image strips .git, so the container can't read git either). Mirrors
# grpo_gain_decomp.provenance.git_commit / git_is_dirty.
def _local_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _local_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(result.stdout.strip())
