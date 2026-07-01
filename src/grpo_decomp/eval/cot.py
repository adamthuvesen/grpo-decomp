"""Verifiable chain-of-thought checking for CoT-gated pass@k.

GSM8K-style reasoning embeds calculator annotations ``<<a op b = c>>``. A chain is
*valid* iff it shows at least one such step and every step computes correctly — a
cheap, non-neural proxy for "the answer came from correct arithmetic, not a lucky
guess", which is the critique CoT-gating addresses.

An LLM judge is not used: it would break the verifiable-only
constraint and add a runtime dependency. The trade-off is coverage — completions
with no calculator steps are unverifiable — so the battery reports the verifiable
fraction alongside the gated metric.
"""

from __future__ import annotations

import math
import operator
import re

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_STEP_RE = re.compile(
    r"<<\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)\s*>>"
)


def verify_steps(text: str) -> tuple[int, int]:
    """Return ``(num_correct, num_total)`` over the binary ``<<a op b=c>>`` steps.

    A division-by-zero step is counted in the total but never correct.
    """
    total = 0
    correct = 0
    for a, op, b, c in _STEP_RE.findall(text):
        total += 1
        left, right, expected = float(a), float(b), float(c)
        if op == "/" and right == 0:
            continue
        if math.isclose(_OPS[op](left, right), expected, rel_tol=1e-9, abs_tol=1e-9):
            correct += 1
    return correct, total


def chain_is_valid(text: str) -> bool:
    """True iff the chain shows >=1 calculator step and all steps compute correctly."""
    correct, total = verify_steps(text)
    return total > 0 and correct == total


def has_verifiable_chain(text: str) -> bool:
    """True iff the completion contains >=1 parseable calculator step (coverage)."""
    return _STEP_RE.search(text) is not None
