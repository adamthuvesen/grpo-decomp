"""Unit tests for CoT chain verification (no network)."""

from __future__ import annotations

from grpo_decomp.eval.cot import chain_is_valid, has_verifiable_chain, verify_steps


def test_verify_steps_counts_correct_and_total() -> None:
    assert verify_steps("she bakes <<9*2=18>>18 and <<18-3=15>>15") == (2, 2)
    assert verify_steps("first <<5+3=9>>9 (wrong) then <<2*2=4>>4") == (1, 2)
    assert verify_steps("no calculator steps here") == (0, 0)


def test_verify_steps_handles_division_and_decimals() -> None:
    assert verify_steps("<<6/2=3>>") == (1, 1)
    assert verify_steps("<<1.5*2=3.0>>") == (1, 1)
    assert verify_steps("<<5/0=0>>") == (0, 1)  # div-by-zero is malformed, never correct


def test_chain_is_valid_requires_all_steps_correct() -> None:
    assert chain_is_valid("<<2+2=4>> so <<4*3=12>>") is True
    assert chain_is_valid("<<2+2=5>>") is False  # a wrong step invalidates
    assert chain_is_valid("no steps, just a guess: 42") is False  # unverifiable -> invalid


def test_has_verifiable_chain_reports_coverage() -> None:
    assert has_verifiable_chain("<<2+2=4>>") is True
    assert has_verifiable_chain("the answer is 4") is False
