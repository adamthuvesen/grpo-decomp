"""Unit tests for answer extraction and grading (no network)."""

from __future__ import annotations

from grpo_gain_decomp.eval.answers import extract_lenient, extract_strict, is_correct


def test_strict_extracts_final_boxed() -> None:
    assert extract_strict(r"work... so \boxed{72}.") == "72"
    assert extract_strict(r"\boxed{1} then later \boxed{42}") == "42"  # final wins


def test_strict_returns_none_without_box() -> None:
    assert extract_strict("the answer is 72") is None


def test_strict_handles_nested_braces() -> None:
    # Balanced scan must keep inner braces (\frac), not stop at the first '}'.
    assert extract_strict(r"so \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_strict(r"thus \boxed{-\frac{1}{2}}") == r"-\frac{1}{2}"
    # final-wins still holds with a nested earlier box.
    assert extract_strict(r"\boxed{\frac{1}{2}} ... \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_strict(r"unbalanced \boxed{\frac{3}{4}") is None


def test_boxed_fraction_grades_correct_under_both_policies() -> None:
    # The format-sensitivity and fraction controls depend on this round-trip.
    text = r"reasoning ... \boxed{\frac{3}{4}}"
    assert is_correct(extract_strict(text), "3/4") is True
    assert is_correct(extract_lenient(text), "3/4") is True  # lenient == strict here
    assert extract_lenient(text) == extract_strict(text)


def test_lenient_prefers_box_then_last_number() -> None:
    assert extract_lenient(r"... \boxed{72}") == "72"
    assert extract_lenient("I think the answer is 72") == "72"
    assert extract_lenient("I think the answer is 3/4") == "3/4"
    assert extract_lenient("I think the answer is -1/2") == "-1/2"
    assert extract_lenient("first try 2, then final answer is 3/4") == "3/4"
    assert extract_lenient("she has 1,000 left") == "1000"
    assert extract_lenient("no number anywhere") is None


def test_lenient_is_a_superset_of_strict() -> None:
    # Whenever strict extracts, lenient returns the SAME answer -> strict <= lenient.
    boxed = r"reasoning \boxed{72}"
    assert extract_lenient(boxed) == extract_strict(boxed)
    # Strict misses an unboxed answer that lenient still recovers.
    unboxed = "I think it is 72 but forgot to box it"
    assert extract_strict(unboxed) is None
    assert extract_lenient(unboxed) == "72"


def test_is_correct_grades_via_math_verify() -> None:
    assert is_correct("72", "72") is True
    assert is_correct("73", "72") is False
    assert is_correct(None, "72") is False
    # math-verify equivalence: decimal vs fraction, and LaTeX from a boxed answer.
    assert is_correct("0.75", "3/4") is True
    assert is_correct(r"\frac{3}{4}", "0.75") is True


def test_format_sensitivity_is_isolable() -> None:
    # An unboxed-but-correct completion: wrong under strict, right under lenient.
    text = "after working it out, 72"
    gold = "72"
    assert is_correct(extract_strict(text), gold) is False
    assert is_correct(extract_lenient(text), gold) is True
