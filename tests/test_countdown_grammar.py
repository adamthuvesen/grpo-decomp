"""The shared evaluator preserves the distinct general and Esme expression contracts."""

from __future__ import annotations

from fractions import Fraction

import pytest

from llm_grpo_gains.data.countdown import (
    ESME_COUNTDOWN_GRAMMAR,
    evaluate_expression,
)
from llm_grpo_gains.esme_countdown import (
    is_valid_esme_countdown_solution,
    is_wellformed_esme_countdown,
)


@pytest.mark.parametrize("text", ["1 + +2", "-1 + 2", "1 - -2", "1 / 2", r"1 \times 2"])
def test_esme_excludes_unary_division_and_operator_aliases(text: str) -> None:
    assert evaluate_expression(text, grammar=ESME_COUNTDOWN_GRAMMAR) is None
    assert not is_wellformed_esme_countdown(text, (1, 2))


def test_esme_evaluates_long_legal_expressions_without_recursion() -> None:
    text = "+".join(["1"] * 1200)
    assert evaluate_expression(text, grammar=ESME_COUNTDOWN_GRAMMAR) == (Fraction(1200), [1] * 1200)
    assert not is_wellformed_esme_countdown(text, (1, 1, 1))
    assert not is_valid_esme_countdown_solution(text, (1, 1, 1), 1200)


def test_esme_rejects_parser_recursion_without_an_exception() -> None:
    assert not is_wellformed_esme_countdown("+" * 10000 + "1", (1,))
