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
from grpo_decomp.stats.bootstrap import bootstrap_mean_ci
from grpo_decomp.stats.compare import compare
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions-dir", required=True, type=Path)
    parser.add_argument("--task-set", default="esme-countdown")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 8, 16])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    register_study()  # routes verifier_for(esme-countdown) to the exact-solve grader

    csets: dict[str, CompletionSet] = {}
    problems = None
    for arm in ARMS:
        sub = args.completions_dir / f"{arm}__{args.task_set}"
        cset = load_completion_set(sub)
        csets[arm] = cset
        if problems is None:
            problems = cset.problem_set()

    valid_by_problem = {arm: _valid_by_problem(csets[arm]) for arm in ARMS}
    any_exact = {arm: _any_exact(csets[arm]) for arm in ARMS}

    rows = {}
    for arm in ARMS:
        cset = csets[arm]
        vbp = valid_by_problem[arm]
        battery = run_battery(problems, cset.completions_by_id(), k_values=args.k)
        rows[arm] = {
            "valid_rate": sum(vbp.values()) / len(vbp),
            "pass_at_k": {pk.k: pk.vanilla for pk in battery.pass_at_k},
            "any_exact_solved": sum(any_exact[arm].values()),
        }

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

    print(f"\n# Esme-214M-RL sampled decomposition (n={n_samples}, temp=1.0, held-out Countdown)\n")
    kcols = " | ".join(f"pass@{k}" for k in args.k)
    print(f"| Arm | valid-expr rate | {kcols} | any-exact solved |")
    print("| --- | ---: | " + " | ".join("---:" for _ in args.k) + " | ---: |")
    for arm in ARMS:
        r = rows[arm]
        ks = " | ".join(f"{r['pass_at_k'][k] * 100:.1f}%" for k in args.k)
        print(
            f"| {ARM_LABELS[arm]} | {r['valid_rate'] * 100:.1f}% | {ks} "
            f"| {r['any_exact_solved']}/{n_problems} |"
        )
    print()

    vr = valid_tests["random"]
    vb = valid_tests["base"]
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
        f"{exact_cmp.accuracy_b * 100:.1f}% vs {exact_cmp.accuracy_a * 100:.1f}%, "
        f"delta {exact_cmp.delta * 100:+.1f}pp, {exact_cmp.test} p={exact_cmp.p_value:.3f}, "
        f"n_discordant={exact_cmp.n_discordant}"
    )
    print()

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_set": args.task_set,
            "n_problems": n_problems,
            "n_samples": n_samples,
            "temperature": 1.0,
            "k_values": args.k,
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
        (args.out / "sampled_summary.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out / 'sampled_summary.json'}")


if __name__ == "__main__":
    main()
