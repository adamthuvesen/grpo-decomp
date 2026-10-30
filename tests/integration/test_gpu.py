"""Explicit local GPU smoke. This test never provisions hardware or downloads a model."""

# The CPU environment has no GPU libraries (torch, transformers, trl, vllm).
# pyright: reportMissingImports=false

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_real_vllm_generation_and_grading() -> None:
    model = os.environ.get("GRPO_GPU_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail("set GRPO_GPU_MODEL to a local model directory before make verify-gpu")
    try:
        import torch
        import trl
        import vllm
    except ImportError as exc:
        pytest.fail(f"GPU smoke requires the locked train extra: {exc}")
    assert torch.cuda.is_available(), "GPU smoke requires visible CUDA hardware"
    assert trl.__version__ == "1.0.0"
    assert vllm.__version__ == "0.17.1"

    from grpo_decomp.eval.battery import run_battery
    from grpo_decomp.eval.completions import SamplingConfig
    from grpo_decomp.eval.generate import generate
    from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet

    problems = ProblemSet(
        source=DatasetRef(name="gpu-smoke", config=None, split="smoke", revision="local-v1"),
        problems=(Problem(id="addition", question="What is 2+2?", gold_answer="4"),),
    )
    samples = generate(
        model, problems, SamplingConfig(temperature=0.0, n=1, max_new_tokens=8), backend="vllm"
    )
    assert set(samples) == {"addition"}
    assert len(samples["addition"]) == 1
    battery = run_battery(problems, {"addition": tuple(samples["addition"])}, k_values=[1])
    assert battery.n_problems == 1
    assert battery.n_samples == 1
