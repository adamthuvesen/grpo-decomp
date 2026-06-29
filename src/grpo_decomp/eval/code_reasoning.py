"""Detect code-style reasoning in completions (the Qwen elicitation signature).

The literature documents that under RL, Qwen-family models increasingly "reason in
code" (emitting Python-like snippets) *regardless* of the reward — an elicitation
signature worth tracking before vs after RL. This is a logged diagnostic, not a
headline-table number, so it is a fixed, non-neural heuristic: presence of fenced
code, common Python cues, or an interpreter prompt.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_CODE_MARKERS = re.compile(r"```|\bprint\(|\bdef\s|\bimport\s|>>>")


def is_code_reasoning(text: str) -> bool:
    """True iff the completion shows code-style reasoning (fenced code / Python cues)."""
    return _CODE_MARKERS.search(text) is not None


def code_reasoning_frequency(completions: Sequence[str]) -> float:
    """Fraction of completions exhibiting code-style reasoning."""
    if not completions:
        raise ValueError("completions is empty")
    return sum(is_code_reasoning(c) for c in completions) / len(completions)
