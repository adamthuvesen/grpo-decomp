"""Integration test: load real GSM8K from HuggingFace at the pinned revision.

Network-bound, so marked `integration` and deselected by default. Run with:
    make test-integration   (or: uv run pytest -m integration)
"""

from __future__ import annotations

import pytest

from grpo_gain_decomp.data.gsm8k import CONFIG, REVISION, load_gsm8k

pytestmark = pytest.mark.integration


def test_load_gsm8k_test_split_conforms() -> None:
    problem_set = load_gsm8k("test")

    # Pinned revision => exact, known size.
    assert len(problem_set) == 1319
    assert problem_set.source.revision == REVISION
    assert problem_set.source.config == CONFIG

    ids = [p.id for p in problem_set]
    assert len(set(ids)) == len(ids), "synthesized ids must be unique"
    assert ids[0] == "gsm8k/main/test/0"

    for problem in problem_set:
        assert problem.question.strip()
        # gold_answer is normalized: a comma-free signed int/decimal.
        float(problem.gold_answer)
