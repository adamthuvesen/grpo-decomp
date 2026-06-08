"""End-to-end seam test: grade -> compare -> decomposition (no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.eval.battery import grade
from grpo_gain_decomp.report.decomposition import DecompositionRow, build_decomposition
from grpo_gain_decomp.report.render import render_table
from grpo_gain_decomp.schemas import DatasetRef, Problem, ProblemSet
from grpo_gain_decomp.stats.compare import compare

_REF = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")


def _problems() -> ProblemSet:
    return ProblemSet(
        source=_REF,
        problems=tuple(Problem(id=f"p{i}", question="q", gold_answer=str(i)) for i in range(6)),
    )


def test_grade_compare_decompose_end_to_end() -> None:
    problems = _problems()
    # base boxes the right answer for 2/6; rl for 5/6.
    base_completions = {f"p{i}": (rf"\boxed{{{i}}}" if i < 2 else r"\boxed{999}") for i in range(6)}
    rl_completions = {f"p{i}": (rf"\boxed{{{i}}}" if i < 5 else r"\boxed{999}") for i in range(6)}

    base = grade(problems, base_completions)
    rl = grade(problems, rl_completions)
    assert sum(base.values()) == 2
    assert sum(rl.values()) == 5

    comparison = compare("base", base, "rl", rl)
    assert comparison.n == 6
    assert comparison.accuracy_a == pytest.approx(2 / 6)
    assert comparison.accuracy_b == pytest.approx(5 / 6)

    decomposition = build_decomposition(
        base_model="Qwen2.5-Math-1.5B",
        task="GSM8K",
        seeds=1,
        raw_gain=DecompositionRow(control="raw gain", probes="RL vs base", comparison=comparison),
        control_rows=[],
        format_row=DecompositionRow(
            control="format", probes="lenient vs strict", comparison=comparison
        ),
        placebo=DecompositionRow(
            control="placebo", probes="reward-signal gain", comparison=comparison
        ),
        elicitation_note="n/a (toy)",
    )
    assert "rl beats base" in render_table(decomposition)


def test_grade_requires_a_completion_for_every_problem() -> None:
    with pytest.raises(ValueError, match="no completion"):
        grade(_problems(), {"p0": r"\boxed{0}"})  # missing p1..p5
