"""GSM-Plus: adversarially perturbed GSM8K variants — the robustness control.

Three ways GSM-Plus differs from the GSM8K-family marker datasets, all verified
against the pinned revision:

1. The gold lives in a bare `answer` field (no ``####`` marker).
2. Golds are int/decimal **or** fraction (e.g. ``3/4`` from the "reversing
   operation" perturbation). Fractions are kept verbatim; the reward layer
   (math-verify) handles numeric equivalence.
3. The "critical thinking" perturbation (1319 rows) is intentionally
   *unanswerable*, with gold ``"None"``. v1's decomposition is numeric
   exact-match, so these are excluded and logged explicitly, never silently dropped.
   Recognizing unanswerability is a separate task, deferred to v2.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from datasets import load_dataset

from llm_grpo_gains.data._common import parse_numeric_gold
from llm_grpo_gains.schemas import DatasetRef, Problem, ProblemSet

logger = logging.getLogger(__name__)

SLUG = "gsm-plus"
NAME = "qintongli/GSM-Plus"
#: Pinned commit SHA (verified 2026-06-06). Re-confirm on any bump.
REVISION = "3b708db57b96a16e8e3368ed2956990c0809440e"

_SPLITS = ("test", "testmini")
#: Sentinel gold for the "critical thinking" (unanswerable) perturbation.
_UNANSWERABLE = "None"
_FRACTION_RE = re.compile(r"\A-?\d+/\d+\Z")


def _gold_of(row: dict[str, Any], record_id: str) -> str:
    """Normalize a GSM-Plus gold: int/decimal via the shared parser, fraction kept."""
    raw = str(row["answer"]).strip()
    if _FRACTION_RE.match(raw):
        return raw
    return parse_numeric_gold(raw, record_id=record_id)


def load_gsm_plus(split: str = "test", *, revision: str = REVISION) -> ProblemSet:
    """Load a GSM-Plus split as a `ProblemSet`, excluding unanswerable problems.

    The "critical thinking" perturbation (gold ``"None"``) is dropped and the
    count logged; a non-numeric, non-fraction, non-``None`` gold is an explicit
    `GoldAnswerError`. Ids keep their original row index for traceability, so the
    surviving ids are stable but not contiguous.
    """
    if split not in _SPLITS:
        raise ValueError(f"GSM-Plus has splits {_SPLITS}, got {split!r}")

    ref = DatasetRef(name=NAME, config=None, split=split, revision=revision)
    rows = load_dataset(NAME, split=split, revision=revision)

    problems = []
    excluded = 0
    for index, row in enumerate(rows):
        if str(row["answer"]).strip() == _UNANSWERABLE:
            excluded += 1
            continue
        record_id = f"{SLUG}/{split}/{index}"
        problems.append(
            Problem(id=record_id, question=row["question"], gold_answer=_gold_of(row, record_id))
        )

    if excluded:
        logger.warning(
            "GSM-Plus %s: excluded %d unanswerable ('critical thinking') problems; %d kept",
            split,
            excluded,
            len(problems),
        )
    return ProblemSet(source=ref, problems=tuple(problems))
