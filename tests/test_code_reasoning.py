"""Unit tests for the code-reasoning detector (no network)."""

from __future__ import annotations

import pytest

from grpo_decomp.eval.code_reasoning import code_reasoning_frequency, is_code_reasoning


@pytest.mark.parametrize(
    "text",
    [
        "```python\nprint(x)\n```",
        "print(42)",
        "def solve():",
        "import math",
        ">>> 2 + 2",
    ],
)
def test_is_code_reasoning_detects_code(text: str) -> None:
    assert is_code_reasoning(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 42.",
        "I defined the variables and imported the totals mentally",  # 'defined'/'imported' not cues
        "the fingerprint(s) matched",  # 'print(' mid-word must not match
    ],
)
def test_is_code_reasoning_ignores_prose(text: str) -> None:
    assert is_code_reasoning(text) is False


def test_code_reasoning_frequency() -> None:
    completions = ["print(1)", "just prose", "```\ncode\n```", "more prose"]
    assert code_reasoning_frequency(completions) == 0.5


def test_code_reasoning_frequency_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        code_reasoning_frequency([])
