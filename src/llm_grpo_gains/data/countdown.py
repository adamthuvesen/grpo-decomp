"""Countdown: a procedurally generated, verifiable search task — the positive control.

Reach a target by combining the given source numbers with ``+ - * /`` (each source used at
most once). Unlike the GSM8K family this is *generated*, not loaded from HuggingFace, so it is
uncontaminated by construction, and it is a search skill a small base model genuinely lacks —
so RL has room to *expand* capability, not just elicit it.

This module is the task's single source of truth, shared by three callers:

- the generator (`load_countdown`) builds guaranteed-solvable problems on the canonical
  `Problem` schema, encoding the `(numbers, target)` answer key in `gold_answer`;
- the `countdown` reward and the eval grader both verify a model's boxed expression with the
  **restricted evaluator** here — a parsed AST that admits only the four operators and the
  given numbers, evaluated exactly with `fractions.Fraction`. It NEVER calls `eval` on model
  output.

Solvability is guaranteed because a target is only chosen if it is reachable from the numbers
(`reachable_targets`), which also gives `solve_countdown` a witness for tests.
"""

from __future__ import annotations

import ast
import hashlib
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from fractions import Fraction
from functools import lru_cache

from pydantic import Field

from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet, Record

# --- The canonical answer key (encoded in Problem.gold_answer) --------------------------

#: LaTeX / unicode operators a boxed answer may use, normalized to ASCII before parsing.
_OPERATOR_ALIASES = {
    "\\times": "*",
    "\\cdot": "*",
    "\\div": "/",
    "\\left": "",
    "\\right": "",
    "×": "*",  # noqa: RUF001 — normalizing the unicode multiplication sign is the point
    "÷": "/",
}


class CountdownKeyError(ValueError):
    """A `gold_answer` is not a parseable Countdown key."""


def format_countdown_key(numbers: Sequence[int], target: int) -> str:
    """Encode the answer key as ``target=<t>;numbers=<n1,n2,...>`` (numbers sorted).

    Kept in `Problem.gold_answer` (a plain string), so the frozen canonical schema is
    unchanged and one parser serves both the reward and the eval grader.
    """
    encoded_numbers = ",".join(str(number) for number in sorted(int(n) for n in numbers))
    return f"target={int(target)};numbers={encoded_numbers}"


def parse_countdown_key(key: str) -> tuple[tuple[int, ...], int]:
    """Decode a Countdown key into ``(numbers, target)``; explicit error if malformed."""
    try:
        target_part, numbers_part = key.split(";")
        target_label, target_raw = target_part.split("=", 1)
        numbers_label, numbers_raw = numbers_part.split("=", 1)
        if target_label != "target" or numbers_label != "numbers" or not numbers_raw:
            raise ValueError
        numbers = tuple(sorted(int(number) for number in numbers_raw.split(",")))
        target = int(target_raw)
    except ValueError as exc:
        raise CountdownKeyError(f"not a Countdown key: {key!r}") from exc
    return numbers, target


# --- The restricted verifier (used on untrusted model output) ---------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class _InvalidExpressionError(Exception):
    """The expression uses a token outside the Countdown grammar (or divides by zero)."""


def _normalize_expression(text: str) -> str:
    """Map common LaTeX/unicode operators to ASCII so a boxed answer parses as Python."""
    for alias, ascii_op in _OPERATOR_ALIASES.items():
        text = text.replace(alias, ascii_op)
    return text


def _eval_node(
    node: ast.AST, allowed_binops: tuple[type[ast.operator], ...]
) -> tuple[Fraction, list[int]]:
    """Evaluate an AST node exactly, returning ``(value, leaf_numbers)``.

    Only numeric literals, ``+ - * /``, unary ±, and parentheses are admitted; anything
    else (a name, call, attribute, ``**``) raises `_InvalidExpressionError`.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
        left, left_leaves = _eval_node(node.left, allowed_binops)
        right, right_leaves = _eval_node(node.right, allowed_binops)
        leaves = left_leaves + right_leaves
        if isinstance(node.op, ast.Add):
            return left + right, leaves
        if isinstance(node.op, ast.Sub):
            return left - right, leaves
        if isinstance(node.op, ast.Mult):
            return left * right, leaves
        if right == 0:  # Div
            raise _InvalidExpressionError("division by zero")
        return left / right, leaves
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        value, leaves = _eval_node(node.operand, allowed_binops)
        return (value if isinstance(node.op, ast.UAdd) else -value), leaves
    if isinstance(node, ast.Constant) and _is_plain_int(node.value):
        return Fraction(node.value), [node.value]
    raise _InvalidExpressionError(f"disallowed node {type(node).__name__}")


def _is_plain_int(value: object) -> bool:
    """An integer literal, excluding `bool` (an int subclass) so `True` isn't the number 1."""
    return isinstance(value, int) and not isinstance(value, bool)


def _evaluate_expression(
    text: str, *, allow_division: bool = True, normalize_operators: bool = True
) -> tuple[Fraction, list[int]] | None:
    """Exactly evaluate a Countdown expression, or None if it is malformed/disallowed.

    Returns ``(value, leaf_numbers)`` — the rational value and the integer literals used —
    so a caller can check both the target and the number budget. Never executes the string.
    """
    try:
        if normalize_operators:
            text = _normalize_expression(text)
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    try:
        allowed_binops = _ALLOWED_BINOPS if allow_division else (ast.Add, ast.Sub, ast.Mult)
        return _eval_node(tree.body, allowed_binops)
    except _InvalidExpressionError:
        return None


def evaluate_expression(text: str) -> tuple[Fraction, list[int]] | None:
    """Exactly evaluate a general Countdown expression, or None when invalid."""
    return _evaluate_expression(text)


def is_valid_countdown_solution(text: str, numbers: Sequence[int], target: int) -> bool:
    """True iff `text` is a legal Countdown solution: each source number used at most once,
    only ``+ - * /``, and it evaluates exactly to the target.
    """
    evaluated = evaluate_expression(text)
    if evaluated is None:
        return False
    value, leaves = evaluated
    if value != target:
        return False
    # Every leaf number must be drawn from the source multiset (no reuse, no invention).
    return not (Counter(leaves) - Counter(numbers))


def countdown_is_correct(extracted: str | None, gold: str) -> bool:
    """Grade an extracted expression against a Countdown key — the eval-layer verifier.

    Same signature as `grpo_decomp.grading.is_correct`, so the battery routes to it by task.
    """
    if extracted is None:
        return False
    numbers, target = parse_countdown_key(gold)
    return is_valid_countdown_solution(extracted, numbers, target)


# --- Reachability (guarantees solvability + provides witnesses) -------------------------


def _combine(a: Fraction, b: Fraction) -> Iterator[Fraction]:
    """The values reachable by applying one operator to an ordered pair ``(a, b)``."""
    yield a + b
    yield a - b
    yield a * b
    if b != 0:
        yield a / b


def _reachable_values(items: tuple[Fraction, ...]) -> Iterator[Fraction]:
    """Every value reachable by combining the items pairwise — subsets included.

    Each current item is reachable on its own because the task permits using fewer than all
    source numbers. Pairwise combinations then cover larger subsets and all-number solutions.
    """
    yield from items
    if len(items) == 1:
        return
    for i in range(len(items)):
        for j in range(len(items)):
            if i == j:
                continue
            rest = tuple(items[k] for k in range(len(items)) if k != i and k != j)
            for combined in _combine(items[i], items[j]):
                yield from _reachable_values((*rest, combined))


@lru_cache(maxsize=4096)
def _reachable_integers(numbers: tuple[int, ...]) -> frozenset[int]:
    """The set of positive-or-any integer targets reachable from `numbers` (memoized)."""
    integers: set[int] = set()
    for value in _reachable_values(tuple(Fraction(n) for n in numbers)):
        if value.denominator == 1:
            integers.add(int(value))
    return frozenset(integers)


def reachable_targets(numbers: Sequence[int], *, lo: int, hi: int) -> list[int]:
    """Sorted integer targets in ``[lo, hi]`` reachable from `numbers` (guarantees a solution)."""
    return sorted(t for t in _reachable_integers(tuple(sorted(numbers))) if lo <= t <= hi)


def solve_countdown(numbers: Sequence[int], target: int) -> bool:
    """True iff some legal expression over `numbers` reaches `target` — for tests/guards."""
    return target in _reachable_integers(tuple(sorted(numbers)))


# --- The generator ----------------------------------------------------------------------


class CountdownConfig(Record):
    """Difficulty + size parameters for one pinned Countdown dataset version."""

    # v2: recalibrated easier after the v1 smoke (3-4 nums, targets <=100) left the general
    # 1.5B base at ~8% solve — below the 10-30% cold-start band. 3 numbers + smaller targets.
    # Sizes sum (800) sits well under this difficulty's reachable pool (~1070 unique problems),
    # so generation terminates with margin; an over-sized config fails clearly (see _MAX_STALL).
    version: str = "v2"
    min_numbers: int = 3
    max_numbers: int = 3
    min_value: int = 1
    max_value: int = 9
    target_lo: int = 10
    target_hi: int = 50
    sizes: dict[str, int] = Field(
        default_factory=lambda: {"train": 512, "validation": 64, "test": 192, "dev": 32}
    )

    @property
    def slug(self) -> str:
        """A compact config id recorded on the `DatasetRef` (the dataset's 'config')."""
        return f"n{self.min_numbers}-{self.max_numbers}_v{self.min_value}-{self.max_value}"


DEFAULT_COUNTDOWN_CONFIG = CountdownConfig()

#: Splits are filled in this fixed order from one deduplicated stream, so they are disjoint.
_SPLIT_ORDER = ("train", "validation", "test", "dev")

#: Consecutive duplicate samples before declaring the reachable pool exhausted. Far above
#: the dedup-rejection rate of a healthy run (sizes well under the pool), so it only trips
#: when the requested splits genuinely exceed the unique problems a difficulty can produce —
#: turning a silent infinite loop into an actionable error.
_MAX_STALL = 5000

#: Consecutive unsolvable source-number draws before declaring a config impossible.
_MAX_EMPTY_TARGET_DRAWS = 5000


def _validate_config(config: CountdownConfig) -> None:
    if config.min_numbers < 1 or config.min_numbers > config.max_numbers:
        raise ValueError(
            f"Countdown min/max numbers must satisfy 1 <= min <= max, got "
            f"{config.min_numbers}..{config.max_numbers}"
        )
    if config.min_value > config.max_value:
        raise ValueError(
            f"Countdown values must satisfy min <= max, got {config.min_value}..{config.max_value}"
        )
    if config.target_lo > config.target_hi:
        raise ValueError(
            f"Countdown target range must satisfy lo <= hi, got "
            f"{config.target_lo}..{config.target_hi}"
        )
    expected_splits = set(_SPLIT_ORDER)
    actual_splits = set(config.sizes)
    if actual_splits != expected_splits:
        raise ValueError(
            f"Countdown sizes must define exactly {_SPLIT_ORDER}, got {tuple(config.sizes)}"
        )
    bad_sizes = {name: size for name, size in config.sizes.items() if size < 0}
    if bad_sizes:
        raise ValueError(f"Countdown split sizes must be non-negative, got {bad_sizes}")


def _sample_problem(rng: random.Random, config: CountdownConfig) -> tuple[tuple[int, ...], int]:
    """Sample a source-number multiset and a reachable target (retrying until solvable)."""
    empty_target_draws = 0
    while True:
        count = rng.randint(config.min_numbers, config.max_numbers)
        numbers = tuple(rng.randint(config.min_value, config.max_value) for _ in range(count))
        targets = reachable_targets(numbers, lo=config.target_lo, hi=config.target_hi)
        if targets:
            return numbers, rng.choice(targets)
        empty_target_draws += 1
        if empty_target_draws > _MAX_EMPTY_TARGET_DRAWS:
            raise ValueError(
                f"Countdown config {config.slug} produced no reachable targets in "
                f"[{config.target_lo}, {config.target_hi}] after {_MAX_EMPTY_TARGET_DRAWS} draws"
            )


def _render_question(numbers: Sequence[int], target: int) -> str:
    listed = ", ".join(str(n) for n in numbers)
    return (
        f"Use the numbers {listed} and the operations +, -, *, / to make {target}. "
        "Each number may be used at most once."
    )


def _content_revision(version: str, problems: Sequence[Problem]) -> str:
    """A content hash pinning a generated split exactly (the generated-set 'revision')."""
    digest = hashlib.sha256()
    for problem in problems:
        digest.update(f"{problem.id}\t{problem.question}\t{problem.gold_answer}\n".encode())
    return f"{version}+{digest.hexdigest()[:16]}"


def load_countdown(
    split: str, *, seed: int = 0, config: CountdownConfig = DEFAULT_COUNTDOWN_CONFIG
) -> ProblemSet:
    """Generate a disjoint Countdown split as a `ProblemSet`, deterministic in ``(seed, config)``.

    The full pool (all splits) is generated from one seeded RNG and deduplicated by
    ``(sorted numbers, target)``, then partitioned in `_SPLIT_ORDER`, so no problem appears in
    two splits. The dataset is fixed across training seeds (like GSM8K's shared train set);
    `seed` parameterizes only the generation itself.
    """
    _validate_config(config)
    if split not in config.sizes:
        raise ValueError(f"Countdown splits are {tuple(config.sizes)}, got {split!r}")

    rng = random.Random(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    by_split: dict[str, list[Problem]] = {name: [] for name in _SPLIT_ORDER}
    for name in _SPLIT_ORDER:
        stalled = 0
        while len(by_split[name]) < config.sizes[name]:
            numbers, target = _sample_problem(rng, config)
            signature = (tuple(sorted(numbers)), target)
            if signature in seen:
                stalled += 1
                if stalled > _MAX_STALL:
                    raise ValueError(
                        f"Countdown pool exhausted at difficulty {config.slug}: requested "
                        f"{sum(config.sizes.values())} unique problems but the reachable pool "
                        "is smaller. Reduce config.sizes or widen the difficulty "
                        "(more numbers / wider value or target range)."
                    )
                continue
            stalled = 0
            seen.add(signature)
            index = len(by_split[name])
            by_split[name].append(
                Problem(
                    id=f"countdown/{config.slug}/{name}/{index}",
                    question=_render_question(numbers, target),
                    gold_answer=format_countdown_key(numbers, target),
                )
            )

    problems = tuple(by_split[split])
    ref = DatasetRef(
        name="countdown",
        config=config.slug,
        split=split,
        revision=_content_revision(config.version, problems),
    )
    return ProblemSet(source=ref, problems=problems)
