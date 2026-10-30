"""Install the wheel in a clean CPU environment outside the source checkout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_installed_wheel_discovers_plugin_grades_and_loads_packaged_fixture(tmp_path) -> None:
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
from grpo_decomp.plugins import load_plugins
from grpo_decomp.registries import EVAL_SETS

assert Path(grpo_decomp.__file__).is_relative_to(sys.prefix)
assert not EVAL_SETS
assert 'gsm8k' in load_plugins()
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
