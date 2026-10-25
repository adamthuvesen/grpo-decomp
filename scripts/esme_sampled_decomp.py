"""Sampled (n>1) decomposition of the Esme-214M-RL gain on held-out Countdown-Lite.

Scores CompletionSets written by the `esme-posttrain` emitter, offline on CPU. Each training
seed contributes a real-reward arm and a random-reward placebo arm at the same training step;
the base model is scored once. Per sample it grades:

- **valid expression**: a legal Countdown-Lite expression (each number once, ``+ - *``),
  whether or not it hits the target.
- **sum template**: a valid expression that uses only ``+``, so its value is the sum of the
  numbers whatever the target.
- **exact solve**: the expression hits the target.

Real minus placebo is paired by seed and problem. Each effect gets a seed x problem bootstrap
interval and an exact sign-flip p-value over seed effects. Exact solves are split by whether
the problem's target equals the sum of its numbers, which shows where the solves come from.
A markdown table is printed, and the summary JSON is written when ``--out`` is provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.grading import extract_lenient
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.stats.bootstrap import seed_problem_bootstrap_ci
from grpo_decomp.stats.significance import sign_flip_p_value
from llm_grpo_gains.data.countdown import parse_countdown_key
from llm_grpo_gains.esme_countdown import (
    esme_countdown_is_correct,
    is_wellformed_esme_countdown,
)

#: Per-problem rates compared real vs placebo; `any_exact` is 1.0 if any sample solves it.
EFFECT_METRICS = ("valid_rate", "exact_rate", "any_exact")
SUMMARY_NAME = "sampled_multiseed_summary.json"


def _parse_seed_arg(value: str) -> tuple[str, Path]:
    label, sep, raw_path = value.partition("=")
    if not sep or not label or not raw_path:
        raise argparse.ArgumentTypeError("--seed must be LABEL=DIR")
    return label, Path(raw_path)


def _is_sum_target(gold_answer: str) -> bool:
    numbers, target = parse_countdown_key(gold_answer)
    return target == sum(numbers)


def _problem_metrics(cset: CompletionSet) -> dict[str, dict[str, float]]:
    """problem id -> valid, sum-template, and exact rates over its samples, plus any-exact."""
    metrics: dict[str, dict[str, float]] = {}
    for item in cset.items:
        numbers, _target = parse_countdown_key(item.problem.gold_answer)
        valid = sum_template = exact = 0
        for sample in item.samples:
            expression = extract_lenient(sample)
            if expression is None or not is_wellformed_esme_countdown(expression, numbers):
                continue
            valid += 1
            sum_template += not any(op in expression for op in "-*")
            exact += esme_countdown_is_correct(expression, item.problem.gold_answer)
        n = len(item.samples)
        metrics[item.problem.id] = {
            "valid_rate": valid / n,
            "sum_template_rate": sum_template / n,
            "exact_rate": exact / n,
            "any_exact": float(exact > 0),
        }
    return metrics


def _arm_row(cset: CompletionSet, by_problem: dict, sum_target_ids: set[str]) -> dict:
    solved = {pid for pid, m in by_problem.items() if m["any_exact"]}
    return {
        "model": cset.provenance.model,
        "valid_rate": _mean([m["valid_rate"] for m in by_problem.values()]),
        "sum_template_rate": _mean([m["sum_template_rate"] for m in by_problem.values()]),
        "exact_rate": _mean([m["exact_rate"] for m in by_problem.values()]),
        "any_exact_rate": _mean([m["any_exact"] for m in by_problem.values()]),
        "solved_sum_target": len(solved & sum_target_ids),
        "solved_other": len(solved - sum_target_ids),
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def _mean_row(rows: list[dict]) -> dict:
    return {key: _mean([row[key] for row in rows]) for key in rows[0] if key != "model"}


def _sampling_axis(cset: CompletionSet) -> tuple:
    p = cset.provenance
    return (p.dataset, p.n_problems, p.sampling)


def _multiseed_payload(
    *, seeds: list[tuple[str, Path]], task_set: str, base_completions_dir: Path
) -> dict:
    if not seeds:
        raise ValueError("at least one --seed is required")
    base_cset = load_completion_set(base_completions_dir)
    axis = _sampling_axis(base_cset)
    problem_ids = sorted(item.problem.id for item in base_cset.items)
    sum_target_ids = {
        item.problem.id for item in base_cset.items if _is_sum_target(item.problem.gold_answer)
    }

    per_seed = []
    deltas: dict[str, list[list[float]]] = {metric: [] for metric in EFFECT_METRICS}
    for seed_label, seed_dir in seeds:
        csets = {
            arm: load_completion_set(seed_dir / f"{arm}__{task_set}")
            for arm in ("correct", "random")
        }
        for arm, cset in csets.items():
            if _sampling_axis(cset) != axis:
                raise ValueError(f"seed {seed_label} {arm} differs from base in data/sampling")
        correct, random = (_problem_metrics(csets[arm]) for arm in ("correct", "random"))
        if sorted(correct) != problem_ids or sorted(random) != problem_ids:
            raise ValueError(f"seed {seed_label} scores a different problem set")
        for metric in EFFECT_METRICS:
            deltas[metric].append([correct[i][metric] - random[i][metric] for i in problem_ids])
        per_seed.append(
            {
                "seed": seed_label,
                "correct": _arm_row(csets["correct"], correct, sum_target_ids),
                "random": _arm_row(csets["random"], random, sum_target_ids),
            }
        )

    effects = {}
    for metric, rows in deltas.items():
        mean, ci_low, ci_high = seed_problem_bootstrap_ci(rows)
        effects[metric] = {
            "mean_delta": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_kind": f"seed x problem bootstrap ({len(rows)} seeds x {len(rows[0])} problems)",
            "sign_flip_p": sign_flip_p_value([_mean(row) for row in rows]),
        }

    dataset, n_problems, sampling = axis
    return {
        "task_set": task_set,
        "dataset_revision": dataset.revision,
        "n_seeds": len(per_seed),
        "seeds": [row["seed"] for row in per_seed],
        "preliminary": len(per_seed) < MIN_HEADLINE_SEEDS,
        "n_problems": n_problems,
        "n_sum_target_problems": len(sum_target_ids),
        "n_samples": sampling.n,
        "temperature": sampling.temperature,
        "max_new_tokens": sampling.max_new_tokens,
        "arms": {
            "base": _arm_row(base_cset, _problem_metrics(base_cset), sum_target_ids),
            "correct": _mean_row([row["correct"] for row in per_seed]),
            "random": _mean_row([row["random"] for row in per_seed]),
        },
        "per_seed": per_seed,
        "effects": effects,
    }


def _print_multiseed(payload: dict) -> None:
    n_problems = payload["n_problems"]
    print(
        f"\n# Esme-214M-RL sampled multiseed decomposition "
        f"({payload['n_seeds']} seeds, {n_problems} problems, n={payload['n_samples']})\n"
    )
    print("| Seed | arm | valid | sum template | exact/sample | solved (sum + other) |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in payload["per_seed"]:
        for arm in ("correct", "random"):
            r = row[arm]
            print(
                f"| {row['seed']} | {arm} | {r['valid_rate'] * 100:.1f}% | "
                f"{r['sum_template_rate'] * 100:.1f}% | {r['exact_rate'] * 100:.2f}% | "
                f"{r['solved_sum_target']} + {r['solved_other']} |"
            )
    print()
    for metric, effect in payload["effects"].items():
        print(
            f"{metric} real - placebo: {effect['mean_delta'] * 100:+.1f}pp, "
            f"95% CI [{effect['ci_low'] * 100:+.1f}, {effect['ci_high'] * 100:+.1f}]pp, "
            f"sign-flip p={effect['sign_flip_p']:.3f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        required=True,
        type=_parse_seed_arg,
        metavar="LABEL=DIR",
        help=(
            "DIR must contain correct__<task-set> and random__<task-set>. "
            "Repeat once per training seed."
        ),
    )
    parser.add_argument(
        "--base-completions-dir",
        type=Path,
        required=True,
        help="Path to the base model's base__<task-set> CompletionSet.",
    )
    parser.add_argument("--task-set", default="esme-countdown")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = _multiseed_payload(
        seeds=args.seed, task_set=args.task_set, base_completions_dir=args.base_completions_dir
    )
    _print_multiseed(payload)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / SUMMARY_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
