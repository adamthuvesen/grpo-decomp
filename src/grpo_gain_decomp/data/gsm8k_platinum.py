"""GSM8K-Platinum: a re-labeled, ambiguity-cleaned GSM8K test set (1209 rows).

The clean-label control — same `#### <number>` answer format as GSM8K, so it
reuses the marker parser. Smaller than GSM8K test (1319) because mislabeled and
ambiguous items were removed.
"""

from __future__ import annotations

from datasets import load_dataset

from grpo_gain_decomp.data._common import build_problem_set, extract_marker_gold
from grpo_gain_decomp.schemas import DatasetRef, ProblemSet

SLUG = "gsm8k-platinum"
NAME = "madrylab/gsm8k-platinum"
CONFIG = "main"
#: Pinned commit SHA (verified 2026-06-06). Re-confirm on any bump.
REVISION = "e762492455a1cf7967de89f05b6bef72fc713b66"


def load_gsm8k_platinum(*, revision: str = REVISION) -> ProblemSet:
    """Load the GSM8K-Platinum test split as a `ProblemSet` (test split only)."""
    ref = DatasetRef(name=NAME, config=CONFIG, split="test", revision=revision)
    rows = load_dataset(NAME, CONFIG, split="test", revision=revision)
    return build_problem_set(
        rows,
        ref=ref,
        id_prefix=f"{SLUG}/{CONFIG}/test",
        gold_of=lambda row, record_id: extract_marker_gold(row["answer"], record_id=record_id),
    )
