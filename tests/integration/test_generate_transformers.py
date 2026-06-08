"""Integration: the real transformers backend on a tiny model (needs `generate` extra + net)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_transformers_backend_end_to_end(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from grpo_gain_decomp.eval.battery import run_battery
    from grpo_gain_decomp.eval.completions import SamplingConfig
    from grpo_gain_decomp.eval.generate import generate
    from grpo_gain_decomp.schemas import DatasetRef, Problem, ProblemSet

    ref = DatasetRef(name="toy", config=None, split="test", revision="rev")
    problems = ProblemSet(
        source=ref,
        problems=(
            Problem(id="p0", question="What is 2+2?", gold_answer="4"),
            Problem(id="p1", question="What is 3+3?", gold_answer="6"),
        ),
    )
    config = SamplingConfig(temperature=0.0, n=1, max_new_tokens=8)
    samples = generate(
        "hf-internal-testing/tiny-random-gpt2", problems, config, backend="transformers"
    )

    # We assert the wiring (shape, keys), not accuracy — the tiny random model can't do math.
    assert set(samples) == {"p0", "p1"}
    assert all(len(group) == 1 for group in samples.values())

    result = run_battery(
        problems, {pid: tuple(group) for pid, group in samples.items()}, k_values=[1]
    )
    assert result.n_problems == 2
    assert result.n_samples == 1
