"""Sampled (n>1) decomposition of the Esme-214M-RL gain on held-out Countdown.

The committed `report --task-set esme-countdown` result is greedy pass@1: one deterministic
sample per problem, graded on exact-solve only. For a 214M model on Countdown that is the
sparsest, lowest-power slice there is — the whole dynamic range is 1-2 solved problems, so real
reward and a random-reward placebo look identical (the accepted table: +3.3pp, p=1.0).

This script scores the *sampled* arms (n=16, temperature 1.0) on the two axes where the reward
signal actually lives:

- **valid-expression rate** — the fraction of samples that are well-formed Esme Countdown-Lite
  expressions (each number used once, ``+ - *`` only, parses), target aside. This is the rung
  the training reward pays 0.3 for; a real verifier reward drives it hard, a random reward has
  no gradient toward it.
- **exact-solve pass@k** — the accepted acceptance-eval metric, restored to a sampled estimate
  so the easy-band solves the model *can* reach are not thrown away by a single greedy decode.

Both axes are paired per problem across arms. Offline, CPU-only; reads the CompletionSets the
`esme-posttrain` emitter wrote. Prints a markdown table and (with ``--out``) writes summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo_decomp.eval.battery import run_battery
from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.grading import extract_lenient
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.stats.bootstrap import bootstrap_mean_ci
from grpo_decomp.stats.compare import compare
from grpo_decomp.stats.seed_aggregate import seed_level_mean_ci
from llm_grpo_gains.data.countdown import parse_countdown_key
from llm_grpo_gains.esme_countdown import (
    esme_countdown_is_correct,
    is_wellformed_esme_countdown,
)
from llm_grpo_gains.registration import register as register_study

ARMS = ("base", "correct", "random")
ARM_LABELS = {
    "base": "base (Esme-214M-Chat)",
    "correct": "correct (Esme-214M-RL, real reward)",
    "random": "random (placebo, random reward)",
}


def _parse_seed_arg(value: str) -> tuple[str, Path]:
    label, sep, raw_path = value.partition("=")
    if not sep or not label or not raw_path:
        raise argparse.ArgumentTypeError("--seed must be LABEL=DIR")
    return label, Path(raw_path)


def _valid_by_problem(cset: CompletionSet) -> dict[str, float]:
    """problem id -> fraction of its samples that are well-formed expressions."""
    rates: dict[str, float] = {}
    for item in cset.items:
        numbers, _target = parse_countdown_key(item.problem.gold_answer)
        valid = sum(
            1
            for s in item.samples
            if (e := extract_lenient(s)) is not None and is_wellformed_esme_countdown(e, numbers)
        )
        rates[item.problem.id] = valid / len(item.samples)
    return rates


def _any_exact(cset: CompletionSet) -> dict[str, bool]:
    """problem id -> whether any sample exactly solves it (pass@n on exact-solve)."""
    return {
        item.problem.id: any(
            esme_countdown_is_correct(extract_lenient(s), item.problem.gold_answer)
            for s in item.samples
        )
        for item in cset.items
    }


def _arm_rows(csets: dict[str, CompletionSet], *, problems, k_values: list[int]) -> dict[str, dict]:
    valid_by_problem = {arm: _valid_by_problem(csets[arm]) for arm in csets}
    any_exact = {arm: _any_exact(csets[arm]) for arm in csets}
    rows = {}
    for arm, cset in csets.items():
        vbp = valid_by_problem[arm]
        battery = run_battery(problems, cset.completions_by_id(), k_values=k_values)
        rows[arm] = {
            "valid_rate": sum(vbp.values()) / len(vbp),
            "pass_at_k": {pk.k: pk.vanilla for pk in battery.pass_at_k},
            "any_exact_solved": sum(any_exact[arm].values()),
        }
    return rows


def _single_seed_payload(*, completions_dir: Path, task_set: str, k_values: list[int]) -> dict:
    csets: dict[str, CompletionSet] = {}
    problems = None
    for arm in ARMS:
        sub = completions_dir / f"{arm}__{task_set}"
        cset = load_completion_set(sub)
        csets[arm] = cset
        if problems is None:
            problems = cset.problem_set()

    valid_by_problem = {arm: _valid_by_problem(csets[arm]) for arm in ARMS}
    any_exact = {arm: _any_exact(csets[arm]) for arm in ARMS}
    rows = _arm_rows(csets, problems=problems, k_values=k_values)

    n_problems = len(problems.problems)
    n_samples = csets["base"].provenance.sampling.n

    # Headline test — valid-expression rate, correct vs the two controls, paired per problem.
    ids = sorted(valid_by_problem["base"])
    valid_tests = {}
    for baseline in ("random", "base"):
        deltas = [valid_by_problem["correct"][i] - valid_by_problem[baseline][i] for i in ids]
        mean_delta, ci_low, ci_high = bootstrap_mean_ci(deltas)
        valid_tests[baseline] = {
            "mean_delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # Exact-solve axis — any-of-n binary per problem, McNemar (sampled analogue of the
    # committed greedy confirmatory test), correct vs random.
    exact_cmp = compare("random", any_exact["random"], "correct", any_exact["correct"])

    return {
        "task_set": task_set,
        "n_problems": n_problems,
        "n_samples": n_samples,
        "temperature": csets["base"].provenance.sampling.temperature,
        "k_values": k_values,
        "arms": rows,
        "valid_rate_tests": valid_tests,
        "exact_solve_test": {
            "label_a": exact_cmp.label_a,
            "label_b": exact_cmp.label_b,
            "accuracy_a": exact_cmp.accuracy_a,
            "accuracy_b": exact_cmp.accuracy_b,
            "delta": exact_cmp.delta,
            "ci_low": exact_cmp.ci_low,
            "ci_high": exact_cmp.ci_high,
            "p_value": exact_cmp.p_value,
            "n_discordant": exact_cmp.n_discordant,
            "test": exact_cmp.test,
        },
    }


def _seed_level_ci(values: list[float]) -> dict:
    mean, sem, ci_low, ci_high, ci_kind = seed_level_mean_ci(values)
    return {
        "mean_delta": mean,
        "sem": sem,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_kind": ci_kind,
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def _multiseed_payload(
    *,
    seeds: list[tuple[str, Path]],
    task_set: str,
    k_values: list[int],
    base_completions_dir: Path | None,
) -> dict:
    if not seeds:
        raise ValueError("at least one --seed is required for multiseed mode")

    per_seed = []
    base: dict | None = None
    n_problems: int | None = None
    n_samples: int | None = None
    temperature: float | None = None
    for seed_label, seed_dir in seeds:
        csets = {
            "correct": load_completion_set(seed_dir / f"correct__{task_set}"),
            "random": load_completion_set(seed_dir / f"random__{task_set}"),
        }
        if base is None:
            base_path = base_completions_dir or (seed_dir / f"base__{task_set}")
            base_cset = load_completion_set(base_path)
            base_rows = _arm_rows(
                {"base": base_cset},
                problems=base_cset.problem_set(),
                k_values=k_values,
            )
            base = base_rows["base"]
        problems = csets["correct"].problem_set()
        rows = _arm_rows(csets, problems=problems, k_values=k_values)
        correct_valid = _valid_by_problem(csets["correct"])
        random_valid = _valid_by_problem(csets["random"])
        correct_exact = _any_exact(csets["correct"])
        random_exact = _any_exact(csets["random"])
        ids = sorted(correct_valid)
        if ids != sorted(random_valid):
            raise ValueError(f"seed {seed_label} has mismatched correct/random problem ids")

        valid_delta = _mean([correct_valid[i] - random_valid[i] for i in ids])
        exact_cmp = compare("random", random_exact, "correct", correct_exact)
        entry = {
            "seed": seed_label,
            "correct": rows["correct"],
            "random": rows["random"],
            "valid_rate_delta": valid_delta,
            "exact_any_delta": exact_cmp.delta,
            "exact_solve_test": {
                "accuracy_a": exact_cmp.accuracy_a,
                "accuracy_b": exact_cmp.accuracy_b,
                "delta": exact_cmp.delta,
                "ci_low": exact_cmp.ci_low,
                "ci_high": exact_cmp.ci_high,
                "p_value": exact_cmp.p_value,
                "n_discordant": exact_cmp.n_discordant,
                "test": exact_cmp.test,
            },
        }
        per_seed.append(entry)

        seed_n_problems = len(problems.problems)
        seed_n_samples = csets["correct"].provenance.sampling.n
        seed_temperature = csets["correct"].provenance.sampling.temperature
        if n_problems is None:
            n_problems = seed_n_problems
            n_samples = seed_n_samples
            temperature = seed_temperature
        elif (
            n_problems != seed_n_problems
            or n_samples != seed_n_samples
            or temperature != seed_temperature
        ):
            raise ValueError(f"seed {seed_label} uses a different problem/sample axis")

    valid_deltas = [float(row["valid_rate_delta"]) for row in per_seed]
    exact_deltas = [float(row["exact_any_delta"]) for row in per_seed]
    n = len(per_seed)
    valid_aggregate = _seed_level_ci(valid_deltas)
    exact_aggregate = _seed_level_ci(exact_deltas)
    preliminary = n < MIN_HEADLINE_SEEDS
    conclusion = (
        "supported: real verifier reward separates from the random-reward placebo "
        "on sampled held-out Countdown validity"
        if not preliminary and valid_aggregate["ci_low"] > 0.0
        else "blocked: seed-level validity interval does not clear zero"
    )

    return {
        "task_set": task_set,
        "n_seeds": n,
        "seeds": [row["seed"] for row in per_seed],
        "n_problems": n_problems,
        "n_samples": n_samples,
        "temperature": temperature,
        "k_values": k_values,
        "preliminary": preliminary,
        "arms": {
            "base": base,
            "correct": {
                "mean_valid_rate": _mean([row["correct"]["valid_rate"] for row in per_seed]),
                "mean_pass_at_k": {
                    str(k): _mean([row["correct"]["pass_at_k"][k] for row in per_seed])
                    for k in k_values
                },
                "mean_any_exact_solved": _mean(
                    [row["correct"]["any_exact_solved"] for row in per_seed]
                ),
            },
            "random": {
                "mean_valid_rate": _mean([row["random"]["valid_rate"] for row in per_seed]),
                "mean_pass_at_k": {
                    str(k): _mean([row["random"]["pass_at_k"][k] for row in per_seed])
                    for k in k_values
                },
                "mean_any_exact_solved": _mean(
                    [row["random"]["any_exact_solved"] for row in per_seed]
                ),
            },
        },
        "per_seed": per_seed,
        "valid_rate_seed_aggregate": valid_aggregate,
        "exact_any_seed_aggregate": exact_aggregate,
        "conclusion": conclusion,
    }


def _print_single(payload: dict) -> None:
    n_samples = payload["n_samples"]
    k_values = payload["k_values"]
    n_problems = payload["n_problems"]
    rows = payload["arms"]
    print(f"\n# Esme-214M-RL sampled decomposition (n={n_samples}, temp=1.0, held-out Countdown)\n")
    kcols = " | ".join(f"pass@{k}" for k in k_values)
    print(f"| Arm | valid-expr rate | {kcols} | any-exact solved |")
    print("| --- | ---: | " + " | ".join("---:" for _ in k_values) + " | ---: |")
    for arm in ARMS:
        r = rows[arm]
        ks = " | ".join(f"{r['pass_at_k'][k] * 100:.1f}%" for k in k_values)
        print(
            f"| {ARM_LABELS[arm]} | {r['valid_rate'] * 100:.1f}% | {ks} "
            f"| {r['any_exact_solved']}/{n_problems} |"
        )
    print()

    vr = payload["valid_rate_tests"]["random"]
    vb = payload["valid_rate_tests"]["base"]
    exact = payload["exact_solve_test"]
    print("Paired per-problem tests (unit = problem, n=30):")
    print(
        f"- valid-expr rate, correct vs random: "
        f"delta {vr['mean_delta'] * 100:+.1f}pp, "
        f"95% CI [{vr['ci_low'] * 100:+.1f}, {vr['ci_high'] * 100:+.1f}]pp"
    )
    print(
        f"- valid-expr rate, correct vs base:   "
        f"delta {vb['mean_delta'] * 100:+.1f}pp, "
        f"95% CI [{vb['ci_low'] * 100:+.1f}, {vb['ci_high'] * 100:+.1f}]pp"
    )
    print(
        f"- any-exact solve, correct vs random: "
        f"{exact['accuracy_b'] * 100:.1f}% vs {exact['accuracy_a'] * 100:.1f}%, "
        f"delta {exact['delta'] * 100:+.1f}pp, {exact['test']} p={exact['p_value']:.3f}, "
        f"n_discordant={exact['n_discordant']}"
    )
    print()


def _print_multiseed(payload: dict) -> None:
    valid = payload["valid_rate_seed_aggregate"]
    exact = payload["exact_any_seed_aggregate"]
    print(
        f"\n# Esme-214M-RL sampled multiseed decomposition "
        f"({payload['n_seeds']} seeds, n={payload['n_samples']}, temp=1.0)\n"
    )
    print(
        "| Seed | correct valid | random valid | Δ valid | correct any-exact | random any-exact |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["per_seed"]:
        print(
            f"| {row['seed']} | {row['correct']['valid_rate'] * 100:.1f}% | "
            f"{row['random']['valid_rate'] * 100:.1f}% | "
            f"{row['valid_rate_delta'] * 100:+.1f}pp | "
            f"{row['correct']['any_exact_solved']:.0f}/30 | "
            f"{row['random']['any_exact_solved']:.0f}/30 |"
        )
    print()
    print(
        f"Valid-expression seed-level mean Δ: {valid['mean_delta'] * 100:+.1f}pp, "
        f"95% CI [{valid['ci_low'] * 100:+.1f}, {valid['ci_high'] * 100:+.1f}]pp "
        f"({valid['ci_kind']})."
    )
    print(
        f"Any-exact seed-level mean Δ: {exact['mean_delta'] * 100:+.1f}pp, "
        f"95% CI [{exact['ci_low'] * 100:+.1f}, {exact['ci_high'] * 100:+.1f}]pp "
        f"({exact['ci_kind']})."
    )
    print(f"Conclusion: {payload['conclusion']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions-dir", type=Path)
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        type=_parse_seed_arg,
        metavar="LABEL=DIR",
        help=(
            "Multiseed input. DIR must contain correct__<task-set> and random__<task-set>. "
            "Repeat once per training seed."
        ),
    )
    parser.add_argument(
        "--base-completions-dir",
        type=Path,
        help="Optional path to base__<task-set> CompletionSet for multiseed reporting.",
    )
    parser.add_argument("--task-set", default="esme-countdown")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    register_study()  # routes verifier_for(esme-countdown) to the exact-solve grader

    if args.seed:
        payload = _multiseed_payload(
            seeds=args.seed,
            task_set=args.task_set,
            k_values=args.k,
            base_completions_dir=args.base_completions_dir,
        )
        _print_multiseed(payload)
        output_name = "sampled_multiseed_summary.json"
    else:
        if args.completions_dir is None:
            parser.error("--completions-dir is required unless --seed is provided")
        payload = _single_seed_payload(
            completions_dir=args.completions_dir,
            task_set=args.task_set,
            k_values=args.k,
        )
        _print_single(payload)
        output_name = "sampled_summary.json"

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / output_name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out / output_name}")


if __name__ == "__main__":
    main()
