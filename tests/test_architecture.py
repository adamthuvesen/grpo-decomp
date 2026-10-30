"""Import every harness module without loading a study or the GPU stack."""

from __future__ import annotations

import os
import subprocess
import sys


def test_harness_imports_without_a_study_or_gpu_stack(tmp_path) -> None:
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

forbidden = {'llm_grpo_gains', 'torch', 'transformers', 'trl', 'vllm', 'modal'}

class RejectLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise AssertionError(f'harness imported {module.__name__}')

class RejectForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            return importlib.util.spec_from_loader(fullname, RejectLoader())
        return None

sys.meta_path.insert(0, RejectForbidden())
import grpo_decomp
for module in pkgutil.walk_packages(grpo_decomp.__path__, grpo_decomp.__name__ + '.'):
    importlib.import_module(module.name)
assert not forbidden.intersection(sys.modules)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
