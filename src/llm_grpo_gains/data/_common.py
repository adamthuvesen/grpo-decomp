"""Shared primitives for turning a raw HuggingFace dataset into a `ProblemSet`.

Every loader reduces its source to the same canonical schema; the differences are
where the gold answer lives and how it is normalized. Keeping the assembly and the
gold parsers here means the fragile parsing seam is written and tested once, not
copied per dataset.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from llm_grpo_gains.schemas import DatasetRef, Problem, ProblemSet

# A signed integer or decimal with optional thousands separators, anchored on a
# digit so a sign- or comma-only body (e.g. ',') never matches as a "number".
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_MARKER_RE = re.compile(r"####\s*(" + _NUMBER + r")")
_NUMERIC_RE = re.compile(r"\A" + _NUMBER + r"\Z")


class GoldAnswerError(ValueError):
    """A record has no parseable gold answer (keyed to the offending record id)."""


def _strip_separators(number: str) -> str:
    return number.replace(",", "")


def extract_marker_gold(answer: str, *, record_id: str) -> str:
    """Gold = the *final* ``#### <number>`` marker, comma-normalized.

    Used by the GSM8K-family sets that ship a chain-of-thought `answer` field
    (GSM8K, GSM-Symbolic, GSM8K-Platinum). Raises `GoldAnswerError` naming the
    record if no marker is present.
    """
    matches = _MARKER_RE.findall(answer)
    if not matches:
        raise GoldAnswerError(f"{record_id}: no '#### <number>' gold answer in answer={answer!r}")
    return _strip_separators(matches[-1])


def parse_numeric_gold(raw: str, *, record_id: str) -> str:
    """Gold = a bare numeric field: surrounding whitespace trimmed, commas stripped.

    Raises `GoldAnswerError` naming the record if the value is not an int/decimal.
    """
    cleaned = raw.strip()
    if not _NUMERIC_RE.match(cleaned):
        raise GoldAnswerError(f"{record_id}: gold answer is not numeric: {raw!r}")
    return _strip_separators(cleaned)


def build_problem_set(
    rows: Iterable[dict[str, Any]],
    *,
    ref: DatasetRef,
    id_prefix: str,
    gold_of: Callable[[dict[str, Any], str], str],
) -> ProblemSet:
    """Assemble a `ProblemSet`, synthesizing ids as ``<id_prefix>/<index>``.

    `gold_of(row, record_id)` returns the normalized gold (or raises
    `GoldAnswerError`). One assembly path so id synthesis and provenance wiring
    stay identical across datasets.
    """
    problems = []
    for index, row in enumerate(rows):
        record_id = f"{id_prefix}/{index}"
        problems.append(
            Problem(id=record_id, question=row["question"], gold_answer=gold_of(row, record_id))
        )
    return ProblemSet(source=ref, problems=tuple(problems))
