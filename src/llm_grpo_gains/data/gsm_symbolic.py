"""GSM-Symbolic: matched-difficulty symbolic templates over GSM8K problems.

The memorization/contamination probe — same questions re-instantiated with
different numbers. Three configs of increasing difficulty: ``main`` (5000),
``p1`` (5000, +1 clause), ``p2`` (2500, +2 clauses); all test split only. The
`answer` field carries the same ``#### <number>`` marker as GSM8K.
"""

from __future__ import annotations

from datasets import load_dataset

from grpo_decomp.schemas import DatasetRef, ProblemSet
from llm_grpo_gains.data._hf_problem_sets import build_problem_set, extract_marker_gold

SLUG = "gsm-symbolic"
NAME = "apple/GSM-Symbolic"
#: Pinned commit SHA (verified 2026-06-06). Re-confirm on any bump.
REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"

_CONFIGS = ("main", "p1", "p2")


def load_gsm_symbolic(config: str = "main", *, revision: str = REVISION) -> ProblemSet:
    """Load a GSM-Symbolic config as a `ProblemSet` (test split only).

    `config` selects difficulty: ``main`` < ``p1`` < ``p2``.
    """
    if config not in _CONFIGS:
        raise ValueError(f"GSM-Symbolic has configs {_CONFIGS}, got {config!r}")

    ref = DatasetRef(name=NAME, config=config, split="test", revision=revision)
    rows = load_dataset(NAME, config, split="test", revision=revision)
    return build_problem_set(
        rows,
        ref=ref,
        id_prefix=f"{SLUG}/{config}/test",
        gold_of=lambda row, record_id: extract_marker_gold(row["answer"], record_id=record_id),
    )
