"""Unit tests for the Countdown positive-control task (no network, no GPU).

Covers the task's single source of truth (`data/countdown.py`) plus the grading and
capability-expansion seams.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from grpo_decomp.eval.battery import grade, verifier_for
from grpo_decomp.grading import is_correct
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from llm_grpo_gains.data.countdown import (
    DEFAULT_COUNTDOWN_CONFIG,
    CountdownConfig,
    CountdownKeyError,
    countdown_is_correct,
    evaluate_expression,
    format_countdown_key,
    is_valid_countdown_solution,
    load_countdown,
    parse_countdown_key,
    reachable_targets,
    solve_countdown,
)

#: Tiny config so generator tests stay fast.
_SMALL = CountdownConfig(sizes={"train": 8, "validation": 4, "test": 6, "dev": 3})
# --- the answer-key codec ---------------------------------------------------------------


def test_key_round_trips_with_sorted_numbers() -> None:
    key = format_countdown_key([7, 4, 6, 5], 30)
    assert key == "target=30;numbers=4,5,6,7"
    assert parse_countdown_key(key) == ((4, 5, 6, 7), 30)


def test_parse_rejects_a_malformed_key() -> None:
    for key in ("not a key", "target=30;numbers=", "answer=30;numbers=4,5"):
        with pytest.raises(CountdownKeyError):
            parse_countdown_key(key)


# --- the restricted, eval-free evaluator ------------------------------------------------


def test_evaluate_basic_arithmetic_and_leaves() -> None:
    value, leaves = evaluate_expression("4 * 5 + 6")
    assert value == 26
    assert sorted(leaves) == [4, 5, 6]


def test_evaluate_is_exact_rational_not_float() -> None:
    value, _ = evaluate_expression("6 / 4")
    assert value == Fraction(3, 2)


def test_evaluate_normalizes_latex_operators() -> None:
    value, leaves = evaluate_expression(r"4 \times 5")
    assert value == 20
    assert sorted(leaves) == [4, 5]


@pytest.mark.parametrize("text", ["2 ** 3", "pow(2, 3)", "x + 1", "4 +", "__import__('os')"])
def test_evaluate_rejects_anything_outside_the_grammar(text: str) -> None:
    assert evaluate_expression(text) is None


# --- solution validity ------------------------------------------------------------------


def test_valid_solution_each_number_once_hits_target() -> None:
    assert is_valid_countdown_solution("4 * 5 + 6", [4, 5, 6, 7], 26)


def test_subset_solution_is_allowed() -> None:
    assert is_valid_countdown_solution("5 * 6", [5, 6, 9], 30)  # uses 2 of 3 numbers


def test_subset_solution_counts_as_reachable() -> None:
    assert is_valid_countdown_solution("9 + 1", [1, 3, 9], 10)
    assert solve_countdown([1, 3, 9], 10)


def test_wrong_target_is_invalid() -> None:
    assert not is_valid_countdown_solution("4 * 5 + 6", [4, 5, 6, 7], 30)


def test_reusing_a_number_is_invalid() -> None:
    assert not is_valid_countdown_solution("5 * 5", [5, 6], 25)  # only one 5 is available


def test_foreign_number_is_invalid() -> None:
    assert not is_valid_countdown_solution("5 * 8", [5, 6], 40)  # 8 is not in the set


def test_countdown_is_correct_grades_against_a_key() -> None:
    key = format_countdown_key([4, 5, 6, 7], 26)
    assert countdown_is_correct("4 * 5 + 6", key)
    assert not countdown_is_correct("4 * 5 + 7", key)
    assert not countdown_is_correct(None, key)


# --- reachability guarantees solvability ------------------------------------------------


def test_reachable_targets_are_all_solvable() -> None:
    numbers = [3, 5, 7]
    targets = reachable_targets(numbers, lo=1, hi=100)
    assert targets
    assert all(solve_countdown(numbers, t) for t in targets)


def test_unreachable_target_has_no_solution() -> None:
    assert not solve_countdown([3, 5], 100)


# --- the generator ----------------------------------------------------------------------


def test_load_countdown_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="splits"):
        load_countdown("nope", config=_SMALL)


def test_generated_problems_are_solvable_and_well_formed() -> None:
    problems = load_countdown("test", config=_SMALL)
    assert len(problems) == 6
    assert problems.source.name == "countdown"
    assert problems.source.split == "test"
    for index, problem in enumerate(problems):
        assert problem.id == f"countdown/{_SMALL.slug}/test/{index}"
        numbers, target = parse_countdown_key(problem.gold_answer)
        assert solve_countdown(numbers, target)  # independently confirm a solution exists
        assert str(target) in problem.question


def test_splits_are_disjoint_in_id_and_content() -> None:
    ids: list[str] = []
    contents: list[str] = []
    for split in ("train", "validation", "test", "dev"):
        for problem in load_countdown(split, config=_SMALL):
            ids.append(problem.id)
            contents.append(problem.gold_answer)  # encodes sorted numbers + target
    assert len(set(ids)) == len(ids)
    assert len(set(contents)) == len(contents)  # no problem reappears across splits


def test_generation_is_deterministic() -> None:
    first = load_countdown("test", config=_SMALL)
    second = load_countdown("test", config=_SMALL)
    assert [p.gold_answer for p in first] == [p.gold_answer for p in second]
    assert first.source.revision == second.source.revision  # content-hash revision is stable


def test_default_config_split_sizes_fit_the_pool() -> None:
    # Regression: the shipped difficulty's reachable pool must exceed the requested split
    # sizes, or the generator would spin forever looking for uniques that don't exist.
    for split, size in DEFAULT_COUNTDOWN_CONFIG.sizes.items():
        assert len(load_countdown(split)) == size


def test_oversized_config_fails_loud_not_infinite_loop() -> None:
    # The pool-exhaustion guard turns a silent hang into a clear error.
    oversized = CountdownConfig(sizes={"train": 5000, "validation": 1, "test": 1, "dev": 1})
    with pytest.raises(ValueError, match="pool exhausted"):
        load_countdown("train", config=oversized)


def test_impossible_target_range_fails_loud_not_infinite_loop() -> None:
    impossible = CountdownConfig(target_lo=10_000, target_hi=10_001, sizes=_SMALL.sizes)
    with pytest.raises(ValueError, match="no reachable targets"):
        load_countdown("train", config=impossible)


# --- task-routed grading ----------------------------------------------------------------


def _countdown_problems() -> ProblemSet:
    ref = DatasetRef(name="countdown", config="x", split="test", revision="r")
    return ProblemSet(
        source=ref,
        problems=(
            Problem(id="c0", question="q", gold_answer=format_countdown_key([4, 5, 6, 7], 26)),
        ),
    )


def test_verifier_routes_by_task() -> None:
    countdown_ref = DatasetRef(name="countdown", config="x", split="test", revision="r")
    gsm8k_ref = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="r")
    assert verifier_for(countdown_ref) is countdown_is_correct
    assert verifier_for(gsm8k_ref) is is_correct


def test_grade_uses_the_countdown_checker_for_countdown_sets() -> None:
    problems = _countdown_problems()
    assert grade(problems, {"c0": r"\boxed{4 * 5 + 6}"})["c0"] is True
    assert grade(problems, {"c0": r"\boxed{4 * 5 + 7}"})["c0"] is False


def test_reward_and_eval_share_countdown_key_parsing() -> None:
    key = format_countdown_key([4, 5, 6, 7], 26)
    assert countdown_is_correct("4 * 5 + 6", key)
    assert grade(
        ProblemSet(
            source=DatasetRef(name="countdown", config="x", split="test", revision="r"),
            problems=(Problem(id="c0", question="q", gold_answer=key),),
        ),
        {"c0": r"\boxed{4 * 5 + 6}"},
    ) == {"c0": True}
