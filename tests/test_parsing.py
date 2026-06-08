"""Unit tests for the shared gold parsers (no network)."""

from __future__ import annotations

import pytest

from grpo_gain_decomp.data._common import GoldAnswerError, extract_marker_gold, parse_numeric_gold


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("She bakes ... <<9*2=18>>18 per day\n#### 18", "18"),
        ("#### 1,000", "1000"),
        ("a long chain\n#### -5", "-5"),
        ("decimal gold\n#### 3.5", "3.5"),
        ("trailing space\n####   42  ", "42"),
        # Gold is the FINAL marker, not the first.
        ("subtotal\n#### 99\nand finally\n#### 42", "42"),
    ],
)
def test_extract_marker_gold(answer: str, expected: str) -> None:
    assert extract_marker_gold(answer, record_id="t") == expected


@pytest.mark.parametrize(
    "answer",
    [
        "no marker here, just prose",
        "",
        "#### ",  # marker, no number
        "#### apples",  # marker, non-number
        "#### ,",  # comma-only body must not normalize to ''
        "#### -",  # sign-only body
    ],
)
def test_extract_marker_gold_raises_when_no_number(answer: str) -> None:
    with pytest.raises(GoldAnswerError, match="rec/7"):
        extract_marker_gold(answer, record_id="rec/7")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("27", "27"), ("18624", "18624"), ("14.4", "14.4"), ("1,200", "1200"), ("  7 ", "7")],
)
def test_parse_numeric_gold(raw: str, expected: str) -> None:
    assert parse_numeric_gold(raw, record_id="t") == expected


@pytest.mark.parametrize("raw", ["None", "3/4", "abc", "", ","])
def test_parse_numeric_gold_rejects_non_numeric(raw: str) -> None:
    with pytest.raises(GoldAnswerError, match="rec/9"):
        parse_numeric_gold(raw, record_id="rec/9")
