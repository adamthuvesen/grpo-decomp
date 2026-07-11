"""Esme Countdown-Lite: the held-out task-set + verifier for the Esme-214M-RL decomposition.

`esme-posttrain` trains a from-scratch 214M model on a *Countdown-Lite* variant (each supplied
number used **exactly once**, operators **+ - ***, no division, integer result equal to the
target). This module plugs that task into the harness so the Esme sampled analysis grades
Esme-emitted `CompletionSet`s.

Two pieces, both study-specific (they live on the study side of the one-way boundary):

- ``load_esme_countdown`` — loads the pinned held-out problem set from a committed fixture.
  The fixture is the exact set the `esme-posttrain` emitter samples over, so the arms' problem
  records and ``DatasetRef`` match it byte-for-byte.
- ``esme_countdown_is_correct`` — the verifier the harness routes to for ``DatasetRef.name ==
  'esme-countdown'``. It applies Esme's rules, which are stricter than the general Countdown
  reward here (exact-once, no division), so grading agrees with Esme's own training reward.

The emitter writes each sample as ``\\boxed{<expression>}``, so the harness's strict/lenient
extractor recovers the expression before this verifier grades it.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from llm_grpo_gains.data.countdown import _evaluate_expression, parse_countdown_key

#: The ``DatasetRef.name`` the emitter stamps and the verifier keys on.
ESME_COUNTDOWN_SOURCE = "esme-countdown"

_FIXTURE_PATH = Path(__file__).parent / "data" / "esme_countdown_heldout_fresh.json"


@lru_cache(maxsize=1)
def _load_fixture() -> ProblemSet:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    ref = DatasetRef(**payload["dataset"])
    problems = tuple(
        Problem(id=p["id"], question=p["question"], gold_answer=p["gold_answer"])
        for p in payload["problems"]
    )
    return ProblemSet(source=ref, problems=problems)


def load_esme_countdown(split: str = "heldout_fresh") -> ProblemSet:
    """Load the pinned Esme held-out Countdown set (a `ProblemSet`).

    Only ``heldout_fresh`` is registered; the fixture carries its own `DatasetRef`, so a
    different split name is an explicit error rather than a silent empty set.
    """
    problem_set = _load_fixture()
    if split != problem_set.source.split:
        raise ValueError(f"Esme Countdown set is {problem_set.source.split!r}, got {split!r}")
    return problem_set


def _evaluate(text: str) -> tuple[Fraction, list[int]] | None:
    """Exactly evaluate an Esme Countdown expression, or None if malformed/disallowed."""
    return _evaluate_expression(text, allow_division=False, normalize_operators=False)


def is_valid_esme_countdown_solution(text: str, numbers: tuple[int, ...], target: int) -> bool:
    """True iff `text` is a legal Esme Countdown-Lite solution.

    Esme's rules: each supplied number is used **exactly once** (not merely at most once),
    only ``+ - *`` operators, and the expression evaluates to an integer equal to `target`.
    """
    evaluated = _evaluate(text)
    if evaluated is None:
        return False
    value, leaves = evaluated
    if value.denominator != 1 or int(value) != target:
        return False
    # Exactly-once: the multiset of leaf numbers must equal the supplied multiset.
    return Counter(leaves) == Counter(numbers)


def is_wellformed_esme_countdown(text: str, numbers: tuple[int, ...]) -> bool:
    """True iff `text` is a legal Esme Countdown-Lite *expression*, target aside.

    Same grammar as `is_valid_esme_countdown_solution` (parses, only ``+ - *`` and unary ±,
    integer literals, each supplied number used **exactly once**) but with the ``== target``
    check dropped. This is the axis the training reward's ``valid_expression`` rung pays for
    (invalid 0.0 < valid 0.3 < exact 1.0): does the model emit a well-formed arithmetic
    expression over the given numbers at all, independent of whether it hits the target. It is
    where a real verifier reward concentrates its gradient and where a random-reward placebo
    has none — so it separates the arms far more sharply than the sparse exact-solve rung.
    """
    evaluated = _evaluate(text)
    if evaluated is None:
        return False
    _value, leaves = evaluated
    return Counter(leaves) == Counter(numbers)


def esme_countdown_is_wellformed(extracted: str | None, gold: str) -> bool:
    """`is_wellformed_esme_countdown` on an extracted expression + Esme key (verifier signature).

    Mirrors `esme_countdown_is_correct` but grades well-formedness rather than exact solve,
    so the same extraction path feeds both the exact-solve and valid-expression axes.
    """
    if extracted is None:
        return False
    numbers, _target = parse_countdown_key(gold)
    return is_wellformed_esme_countdown(extracted, numbers)


def esme_countdown_is_correct(extracted: str | None, gold: str) -> bool:
    """Grade an extracted expression against an Esme Countdown key (the eval-layer verifier).

    Same signature as `grpo_decomp.grading.is_correct`, so the battery routes to it for
    ``DatasetRef.name == 'esme-countdown'``. The key format matches the general Countdown
    key (`target=<t>;numbers=<n1,...>`), so its parser is reused.
    """
    if extracted is None:
        return False
    numbers, target = parse_countdown_key(gold)
    return is_valid_esme_countdown_solution(extracted, numbers, target)
