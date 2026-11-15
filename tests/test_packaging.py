"""Pin the runtime dependency list and install the wheel outside the source checkout."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_RUNTIME_GROUPS = ("generate", "train", "eval")
_ALLOWED_RUNTIME_DEPENDENCIES = {
    "pydantic", "datasets", "numpy", "scipy", "math-verify", "pyyaml",
    "transformers", "torch", "grpo-decomp", "trl", "vllm", "wandb", "polars", "eval-audit",
}  # fmt: skip


def _name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _runtime_requirements() -> list[str]:
    project = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    return project["dependencies"] + [req for group in _RUNTIME_GROUPS for req in extras[group]]


def test_runtime_dependencies_are_the_approved_stack() -> None:
    added = {_name(req) for req in _runtime_requirements()} - _ALLOWED_RUNTIME_DEPENDENCIES
    assert not added, (
        f"New runtime dependencies {sorted(added)}: ask Adam before adding them. "
        "Dev-only tools go in the `dev` extra; once approved, extend _ALLOWED_RUNTIME_DEPENDENCIES."
    )


def test_trl_and_vllm_are_exact_pins() -> None:
    pins = {_name(req): req for req in _runtime_requirements()}
    for name in ("trl", "vllm"):
        assert "==" in pins[name], f"{name} must stay an exact == pin: its GRPO behavior moves."


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.slow
def test_installed_wheel_registers_study_grades_and_loads_packaged_fixture(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    uv = shutil.which("uv")
    assert uv, "wheel smoke requires uv"
    wheels = tmp_path / "wheels"
    _run([uv, "build", "--wheel", "--offline", "--out-dir", str(wheels)], cwd=_REPO, env=env)
    (wheel,) = wheels.glob("*.whl")
    project = tmp_path / "installed"
    project.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(_REPO / name, project / name)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    _run(
        [
            uv,
            "sync",
            "--locked",
            "--offline",
            "--no-install-project",
            "--extra",
            "eval",
            "--python",
            sys.executable,
            "--project",
            str(project),
        ],
        cwd=tmp_path,
        env=env,
    )
    clean_env = project / ".venv"
    python = clean_env / "bin" / "python"
    _run(
        [uv, "pip", "install", "--offline", "--no-deps", "--python", str(python), str(wheel)],
        cwd=tmp_path,
        env=env,
    )
    script = """
from pathlib import Path
import sys
import grpo_decomp
from grpo_decomp.registries import EVAL_SETS

assert Path(grpo_decomp.__file__).is_relative_to(sys.prefix)
assert not EVAL_SETS
from llm_grpo_gains.registration import register
register()
assert 'gsm8k-test' in EVAL_SETS
from llm_grpo_gains.esme_countdown import load_esme_countdown
assert len(load_esme_countdown()) == 200
assert not {'torch', 'transformers', 'vllm', 'trl'}.intersection(sys.modules)
"""
    _run([str(python), "-I", "-c", script], cwd=tmp_path, env=env)
    fixture = tmp_path / "mini"
    shutil.copytree(_REPO / "tests" / "fixtures" / "mini" / "base__mini", fixture)
    result = _run(
        [
            str(clean_env / "bin" / "grpo-decomp"),
            "battery",
            "--completions",
            str(fixture),
            "--k",
            "1",
        ],
        cwd=tmp_path,
        env=env,
    )
    assert json.loads(result.stdout)["n_problems"] == 12
