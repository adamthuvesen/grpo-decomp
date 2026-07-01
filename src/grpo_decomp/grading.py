r"""Answer extraction policies and correctness grading.

Two extraction policies make format-driven gains separable from reasoning gains:

- **strict** — only the designated answer format (the final ``\boxed{...}``).
- **lenient** — the boxed answer if present, else the last number anywhere.

Lenient is a strict superset: whenever strict extracts an answer, lenient returns
the same one, so strict accuracy <= lenient accuracy by construction.
"""

from __future__ import annotations

import re

from math_verify import parse, verify

_BOXED_OPEN = r"\boxed{"
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:/\d[\d,]*|\.\d+)?")


def _last_boxed(text: str) -> str | None:
    """Return the brace-balanced content of the final ``\\boxed{...}``, or None."""
    start = text.rfind(_BOXED_OPEN)
    if start == -1:
        return None
    depth = 0
    content: list[str] = []
    for char in text[start + len(_BOXED_OPEN) - 1 :]:  # start at the opening brace
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
        content.append(char)
    return None


def extract_strict(text: str) -> str | None:
    """The content of the final ``\\boxed{...}``, or None if the answer is not boxed."""
    return _last_boxed(text)


def extract_lenient(text: str) -> str | None:
    """The boxed answer if present, else the last number in the text; None if neither."""
    boxed = extract_strict(text)
    if boxed is not None:
        return boxed
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def is_correct(extracted: str | None, gold: str) -> bool:
    """True iff an extracted answer is mathematically equal to the gold answer."""
    if extracted is None:
        return False
    parsed = parse(extracted)
    return bool(parsed) and bool(verify(parse(gold), parsed))
