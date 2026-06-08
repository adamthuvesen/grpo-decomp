"""Unit tests for the backend-agnostic sampler — fake backend, no model, no CUDA."""

from __future__ import annotations

import importlib

import pytest

from grpo_gain_decomp.eval.completions import SamplingConfig
from grpo_gain_decomp.eval.generate import generate, resolve_backend
from grpo_gain_decomp.schemas import DatasetRef, Problem, ProblemSet

#: The real submodule (via sys.modules), not the `generate` *function* that
#: `grpo_gain_decomp.eval` re-exports under the same name — that shadows the package attribute.
_GENERATE_MODULE = importlib.import_module("grpo_gain_decomp.eval.generate")


def _problems() -> ProblemSet:
    ref = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")
    problems = (
        Problem(id="p0", question="What is 2+2?", gold_answer="4"),
        Problem(id="p1", question="What is 3+3?", gold_answer="6"),
    )
    return ProblemSet(source=ref, problems=problems)


def _fake_backend(samples_per_prompt: int):
    def backend(model, prompts, config, *, revision):
        return [
            [f"{model}:{i}:s{j}" for j in range(samples_per_prompt)] for i in range(len(prompts))
        ]

    return backend


def test_resolve_backend_explicit() -> None:
    assert resolve_backend("transformers") == "transformers"
    assert resolve_backend("vllm") == "vllm"


def test_resolve_backend_auto_without_cuda_is_transformers() -> None:
    # The dev/eval env has no torch (and no CUDA), so auto must fall back to transformers.
    assert resolve_backend("auto") == "transformers"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        resolve_backend("jax")


def test_generate_keys_by_problem_id(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(1))
    out = generate("m", _problems(), SamplingConfig(temperature=0.0, n=1), backend="transformers")
    assert set(out) == {"p0", "p1"}
    assert all(len(samples) == 1 for samples in out.values())


def test_generate_respects_sample_count(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(4))
    out = generate("m", _problems(), SamplingConfig(temperature=0.8, n=4), backend="transformers")
    assert all(len(samples) == 4 for samples in out.values())


def test_generate_greedy_with_multiple_samples_is_loud() -> None:
    with pytest.raises(ValueError, match="greedy"):
        generate("m", _problems(), SamplingConfig(temperature=0.0, n=2), backend="transformers")


def test_generate_detects_backend_count_mismatch(monkeypatch) -> None:
    # Backend hands back 2 samples though the config asked for 1: explicit error.
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(2))
    with pytest.raises(ValueError, match="expected n=1"):
        generate("m", _problems(), SamplingConfig(temperature=0.0, n=1), backend="transformers")


def test_interface_is_callable_without_cuda() -> None:
    # The module imported on a CUDA-free host and exposes the interface.
    assert callable(generate)
    assert callable(resolve_backend)
