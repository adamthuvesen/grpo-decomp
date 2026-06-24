"""Load GSM8K into llm_grpo_gains's canonical `ProblemSet`, pinned to a fixed revision.

GSM8K stores the gold answer at the end of a chain-of-thought `answer` field,
after a ``#### `` marker; `extract_marker_gold` pulls that final number. Per the
data-pipeline spec, an unparseable gold is an explicit error keyed to the record id —
never a silent drop, which would quietly corrupt every downstream accuracy number.
"""

from __future__ import annotations

from datasets import load_dataset

from llm_grpo_gains.data._common import build_problem_set, extract_marker_gold
from llm_grpo_gains.schemas import DatasetRef, ProblemSet

SLUG = "gsm8k"
NAME = "openai/gsm8k"
CONFIG = "main"
#: Pinned commit SHA (verified 2026-06-06). Re-confirm on any bump.
REVISION = "740312add88f781978c0658806c59bc2815b9866"

_SPLITS = ("train", "test")


def load_gsm8k(split: str, *, revision: str = REVISION) -> ProblemSet:
    """Load a GSM8K split as a `ProblemSet`.

    Ids are synthesized as ``gsm8k/<config>/<split>/<index>`` (GSM8K ships no
    stable per-row id); the source revision is recorded on the `ProblemSet`.
    """
    if split not in _SPLITS:
        raise ValueError(f"GSM8K has splits {_SPLITS}, got {split!r}")

    ref = DatasetRef(name=NAME, config=CONFIG, split=split, revision=revision)
    rows = load_dataset(NAME, CONFIG, split=split, revision=revision)
    return build_problem_set(
        rows,
        ref=ref,
        id_prefix=f"{SLUG}/{CONFIG}/{split}",
        gold_of=lambda row, record_id: extract_marker_gold(row["answer"], record_id=record_id),
    )
