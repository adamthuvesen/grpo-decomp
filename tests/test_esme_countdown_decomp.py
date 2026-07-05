"""CPU end-to-end proof for the Esme-214M-RL decomposition (grpo-decomp side).

Proves the grpo-decomp half without Modal or a model: the registered ``esme-countdown``
eval-set + verifier grade Esme-shaped ``CompletionSet``s, and ``report --task-set
esme-countdown`` turns three arms (base/correct/random) into a decomposition. The sibling
esme-posttrain test proves the emitter and the placebo GRPO mode that produce these arms.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import permutations, product
from pathlib import Path

import pytest

from grpo_decomp.eval.cli import main as cli_main
from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    write_completion_set,
)
from grpo_decomp.plugins import load_plugins
from grpo_decomp.registries import EVAL_SETS, verifier_for
from llm_grpo_gains.data.countdown import parse_countdown_key
from llm_grpo_gains.esme_countdown import (
    ESME_COUNTDOWN_SOURCE,
    esme_countdown_is_correct,
    esme_countdown_is_wellformed,
    load_esme_countdown,
)


@pytest.fixture(autouse=True)
def _plugins() -> None:
    load_plugins(force=True)


def _solve(numbers: tuple[int, ...], target: int) -> str:
    """A legal Esme Countdown solution (each number once, + - *) reaching `target`."""
    ops = ("+", "-", "*")
    for ordered in set(permutations(numbers)):
        if len(ordered) == 2:
            a, b = ordered
            for op in ops:
                expr = f"{a} {op} {b}"
                if esme_countdown_is_correct(expr, f"target={target};numbers={_key(numbers)}"):
                    return expr
        else:
            a, b, c = ordered
            for left, right in product(ops, repeat=2):
                for expr in (f"({a} {left} {b}) {right} {c}", f"{a} {left} ({b} {right} {c})"):
                    if esme_countdown_is_correct(expr, f"target={target};numbers={_key(numbers)}"):
                        return expr
    raise AssertionError(f"no solution found for {numbers} -> {target}")


def _key(numbers: tuple[int, ...]) -> str:
    return ",".join(str(n) for n in sorted(numbers))


def _write_arm(path: Path, *, model: str, samples_for: dict[str, str]) -> None:
    problems = load_esme_countdown()
    items = tuple(
        ProblemCompletions(problem=problem, samples=(samples_for[problem.id],))
        for problem in problems
    )
    provenance = GenerationProvenance(
        model=model,
        model_revision=None,
        backend="transformers",
        prompt_strategy="esme_countdown_chat",
        sampling=SamplingConfig(temperature=0.0, n=1),
        dataset=problems.source,
        n_problems=len(problems),
        commit="c" * 40,
        python_version="3.11.0",
        package_versions={},
    )
    write_completion_set(CompletionSet(provenance=provenance, items=items), path)


def test_esme_countdown_registered() -> None:
    load_plugins(force=True)
    assert "esme-countdown" in EVAL_SETS
    problems = load_esme_countdown()
    assert problems.source.name == ESME_COUNTDOWN_SOURCE
    assert len(problems) == 30
    # The harness routes esme-countdown grading to the Esme verifier, not the default.
    assert verifier_for(problems.source) is esme_countdown_is_correct


def test_esme_countdown_fixture_revision_is_pinned() -> None:
    """Guard the cross-repo contract: this must equal esme-posttrain's emitter revision.

    The revision is a content hash of the 30 held-out problem records the esme-posttrain
    emitter samples over. esme-posttrain pins the same value in its emitter test; if either
    side changes, the pins diverge and one repo's test fails, catching silent drift before
    a report grades arms over mismatched problems.
    """
    problems = load_esme_countdown()
    assert problems.source.revision == "heldout_fresh+e6e671c24ca56d27"


def test_esme_verifier_rules() -> None:
    gold = "target=10;numbers=1,9"
    assert esme_countdown_is_correct("1 + 9", gold) is True
    assert esme_countdown_is_correct("9 * 1", gold) is False  # wrong value
    assert esme_countdown_is_correct("9 / 1 + 1", gold) is False  # division illegal
    assert esme_countdown_is_correct("9", gold) is False  # not all numbers used
    assert esme_countdown_is_correct("1 + 1", "target=2;numbers=1,9") is False  # number reuse
    assert esme_countdown_is_correct(None, gold) is False


def test_esme_wellformed_rules() -> None:
    """Well-formedness is the exact-solve grammar minus the ``== target`` check.

    This is the valid-expression axis the sampled decomposition scores: a legal expression
    over the given numbers is well-formed even when it misses the target (where a real reward
    concentrates its signal), while grammar violations — division, number reuse/omission —
    are still rejected.
    """
    gold = "target=10;numbers=1,9"
    # Hits the target: well-formed AND correct.
    assert esme_countdown_is_wellformed("1 + 9", gold) is True
    assert esme_countdown_is_correct("1 + 9", gold) is True
    # Legal expression, wrong value: well-formed but NOT correct — the axis separation.
    assert esme_countdown_is_wellformed("9 * 1", gold) is True
    assert esme_countdown_is_correct("9 * 1", gold) is False
    # Grammar violations are rejected on both axes.
    assert esme_countdown_is_wellformed("9 / 1 + 1", gold) is False  # division illegal
    assert esme_countdown_is_wellformed("9", gold) is False  # not all numbers used
    assert esme_countdown_is_wellformed("1 + 1", gold) is False  # number reuse / omission
    assert esme_countdown_is_wellformed(None, gold) is False


def test_report_decomposes_three_arms(tmp_path: Path) -> None:
    problems = load_esme_countdown()
    solutions = {
        problem.id: _solve(*parse_countdown_key(problem.gold_answer)) for problem in problems
    }

    # correct solves every problem; base/random emit an empty box (graded wrong). The
    # emitter wraps expressions in \boxed{...}; the report's strict/lenient extractor
    # must recover them before the Esme verifier grades — this exercises that path.
    completions_dir = tmp_path / "runs"
    _write_arm(
        completions_dir / "correct__esme-countdown",
        model="Esme-214M-RL",
        samples_for={pid: f"\\boxed{{{expr}}}" for pid, expr in solutions.items()},
    )
    _write_arm(
        completions_dir / "base__esme-countdown",
        model="Esme-214M-Chat",
        samples_for={problem.id: "\\boxed{}" for problem in problems},
    )
    _write_arm(
        completions_dir / "random__esme-countdown",
        model="Esme-214M-RL-random",
        samples_for={problem.id: "\\boxed{}" for problem in problems},
    )

    out = tmp_path / "out"
    exit_code = cli_main(
        [
            "report",
            "--completions-dir",
            str(completions_dir),
            "--task-set",
            "esme-countdown",
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "esme-countdown"
    # correct solves all 30, base solves 0 -> a full raw gain, and correct beats the
    # random placebo by the same margin (the placebo carries no signal here).
    raw_gain = next(row for row in summary["rows"] if row["control"] == "raw gain")
    assert raw_gain["comparison"]["delta"] == pytest.approx(1.0)
    placebo = summary["confirmatory_comparison"]
    assert placebo["comparison"]["delta"] == pytest.approx(1.0)
    assert (out / "decomposition.md").is_file()


def test_sampled_decomp_writes_multiseed_summary(tmp_path: Path) -> None:
    problems = load_esme_countdown()
    solutions = {
        problem.id: _solve(*parse_countdown_key(problem.gold_answer)) for problem in problems
    }
    base = tmp_path / "base__esme-countdown"
    _write_arm(
        base,
        model="Esme-214M-Chat",
        samples_for={problem.id: "\\boxed{}" for problem in problems},
    )

    seed_args: list[str] = []
    for seed in ("214", "215", "216"):
        seed_dir = tmp_path / f"seed{seed}"
        _write_arm(
            seed_dir / "correct__esme-countdown",
            model=f"Esme-214M-RL-seed{seed}",
            samples_for={pid: f"\\boxed{{{expr}}}" for pid, expr in solutions.items()},
        )
        _write_arm(
            seed_dir / "random__esme-countdown",
            model=f"Esme-214M-RL-random-seed{seed}",
            samples_for={problem.id: "\\boxed{}" for problem in problems},
        )
        seed_args.extend(["--seed", f"{seed}={seed_dir}"])

    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "esme_sampled_decomp.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *seed_args,
            "--base-completions-dir",
            str(base),
            "--task-set",
            "esme-countdown",
            "--k",
            "1",
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "sampled multiseed decomposition" in result.stdout

    summary = json.loads((out / "sampled_multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["n_seeds"] == 3
    assert summary["preliminary"] is False
    assert summary["valid_rate_seed_aggregate"]["mean_delta"] == pytest.approx(1.0)
    assert summary["valid_rate_seed_aggregate"]["ci_low"] == pytest.approx(1.0)
    assert summary["conclusion"].startswith("supported:")
