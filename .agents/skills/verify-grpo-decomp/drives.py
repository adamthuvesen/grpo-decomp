"""grpo-decomp drives: the DRIVES table, the control, and the subprocess primitives they share."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
PACKAGE = "grpo_decomp"
INSTALL_HINT = "make install"
FIXTURES = ROOT / "tests" / "fixtures" / "mini"
SHARED = [
    "pyproject.toml",
    "uv.lock",
    ".agents/skills/verify-grpo-decomp/**",
    "src/grpo_decomp/plugins.py",
    "src/grpo_decomp/registries.py",
]
TRIPWIRE_TOOLS = ["modal", "wandb", "huggingface-cli", "hf", "nvidia-smi"]
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class DriveError(Exception):
    pass


@dataclass(frozen=True)
class Drive:
    proves: str
    paths: list[str]
    run: Callable[[Path, dict[str, str]], str]


def run(
    evidence: Path, env: dict[str, str], name: str, argv: list[str], timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        raise DriveError(f"{name}: timed out after {timeout}s") from None
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    redacted = {
        k: ("<redacted>" if any(m in k for m in SECRET_MARKERS) else v) for k, v in env.items()
    }
    (evidence / f"{name}.cmd.json").write_text(
        json.dumps({"argv": argv, "env": redacted, "exit": proc.returncode}, indent=2)
    )
    (evidence / f"{name}.stdout").write_text(out)
    (evidence / f"{name}.stderr").write_text(err)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise DriveError(message)


def drive_battery(evidence: Path, env: dict[str, str], expected: float = 4 / 12) -> str:
    out = evidence / "base.battery.json"
    result = run(
        evidence,
        env,
        "battery",
        [
            str(PYTHON),
            "-m",
            "grpo_decomp.eval.cli",
            "battery",
            "--completions",
            str(FIXTURES / "base__mini"),
            "--k",
            "1",
            "2",
            "--out",
            str(out),
        ],
    )
    expect(
        result.returncode == 0,
        f"battery exited {result.returncode}: {result.stderr.strip()[-300:]}",
    )
    battery = json.loads(out.read_text())
    expect(battery["n_problems"] == 12, f"expected 12 graded problems, got {battery['n_problems']}")
    expect(
        abs(battery["strict_accuracy"] - expected) < 1e-9,
        f"expected strict accuracy {expected:.4f}, got {battery['strict_accuracy']:.4f}",
    )
    expect(
        [p["k"] for p in battery["pass_at_k"]] == [1, 2],
        f"expected pass@1 and pass@2, got {battery['pass_at_k']}",
    )
    return f"graded 12 problems x 4 samples, strict accuracy {battery['strict_accuracy']:.3f}"


def control_battery(evidence: Path, env: dict[str, str]) -> str:
    return drive_battery(evidence, env, expected=0.9)


def drive_passk_report(evidence: Path, env: dict[str, str]) -> str:
    sets = Path(tempfile.mkdtemp(dir=env["HOME"]))
    for arm in ["base", "correct-seed0", "correct-seed1"]:
        shutil.copytree(FIXTURES / f"{arm}__mini", sets / f"{arm}__dev")
    result = run(
        evidence,
        env,
        "report-passk-seeds",
        [
            str(PYTHON),
            "-m",
            "grpo_decomp.eval.cli",
            "report-passk-seeds",
            "--completions-dir",
            str(sets),
            "--task-set",
            "dev",
            "--k",
            "2",
        ],
    )
    expect(
        result.returncode == 0,
        f"report-passk-seeds exited {result.returncode}: {result.stderr.strip()[-300:]}",
    )
    for needle in [
        "correct pass@2 73.6% vs base 50.0%",
        "[PRELIMINARY <3 seeds]",
        "| 1 | 58.3% | 80.6% |",
    ]:
        expect(
            needle in result.stdout,
            f"expected report to contain {needle!r}; got:\n{result.stdout[:400]}",
        )
    return "2-seed pass@2 report renders +23.6 pp with CI and labels it PRELIMINARY"


APPROVED = {
    "train": "modal run modal_app.py --smoke-problems 8 --max-steps 5 --no-spawn --approved",
    "heldout": "modal run --detach modal_app.py --command heldout --approved",
    "battery": (
        "modal run --detach modal_app.py --command battery --seed 1 --scope placebo --approved"
    ),
    "passk-multiseed": (
        "modal run --detach modal_app.py --command passk-multiseed --task countdown --approved"
    ),
    "train_arm": "modal run --detach modal_app.py <same arguments> --approved",
    "heldout_arm": "modal run --detach modal_app.py <same arguments> --approved",
    "eval_matrix": "modal run --detach modal_app.py <same arguments> --approved",
    "passk_multiseed": "modal run --detach modal_app.py <same arguments> --approved",
}


def drive_launch_plan(evidence: Path, env: dict[str, str]) -> str:
    probe = [str(PYTHON), "-W", "ignore", str(HERE / "launch_plan.py")]
    result = run(evidence, env | {"PYTHONPATH": str(ROOT)}, "launch-plan", probe)
    expect(
        result.returncode == 0,
        f"launch plan exited {result.returncode}: {result.stderr.strip()[-400:]}",
    )
    plan = json.loads(result.stdout.strip().splitlines()[-1])
    (evidence / "launch-plan.json").write_text(json.dumps(plan, indent=2))
    expect(plan["entrypoint_is_modal"], "expected main to be a real modal LocalEntrypoint")
    expect(
        set(plan["gpus"].values()) == {"A100-80GB"},
        f"expected every function on A100-80GB, got {plan['gpus']}",
    )
    expect(
        plan["arm"]["max_steps"] == 5 and plan["arm"]["base_model"] == "Qwen/Qwen2.5-Math-1.5B",
        f"unexpected effective arm {plan['arm']}",
    )
    for name, command in APPROVED.items():
        refusal = plan["refusals"][name]
        expect(
            refusal.startswith("Refusing to launch") and refusal.endswith(f"run: {command}"),
            f"{name}: expected refusal ending 'run: {command}', got {refusal!r}",
        )
    return f"A100-80GB plan for a 5-step Qwen2.5-Math-1.5B arm; {len(APPROVED)} launch paths refuse unapproved"


DRIVES = {
    "battery": Drive(
        "grpo-decomp battery grades a committed CompletionSet into pass@k JSON",
        [
            "src/grpo_decomp/eval/**",
            "src/grpo_decomp/grading.py",
            "src/grpo_decomp/rewards/**",
            "src/grpo_decomp/schemas.py",
            "tests/fixtures/mini/**",
        ],
        drive_battery,
    ),
    "passk-report": Drive(
        "report-passk-seeds renders the multi-seed pass@k report with CI and the <3-seed label",
        [
            "src/grpo_decomp/eval/**",
            "src/grpo_decomp/report/**",
            "src/grpo_decomp/stats/**",
            "src/grpo_decomp/grading.py",
            "src/llm_grpo_gains/**",
        ],
        drive_passk_report,
    ),
    "launch-plan": Drive(
        "modal_app resolves the paid A100 plan and every launch path refuses without --approved",
        [
            "modal_app.py",
            "configs/**",
            "src/grpo_decomp/train/**",
            "src/grpo_decomp/jobs.py",
            "src/llm_grpo_gains/registration.py",
        ],
        drive_launch_plan,
    ),
}
SMOKE = "battery"
# The control passes only when its drive fails on exactly this planted expectation.
CONTROL_FAILURE = "expected strict accuracy 0.9000, got "
CONTROL = ("battery", control_battery)
