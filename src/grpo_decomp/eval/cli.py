"""`grpo-decomp`: generate completions, score artifacts, and report controls.

Commands follow the artifact contract:

- ``generate`` — the only model-loading command (a generation backend required). One
  model (base or checkpoint) over one problem set -> a `CompletionSet` artifact.
- ``battery``  — a `CompletionSet` -> a `BatteryResult`. CPU-only.
- ``report-seeds`` / ``report-passk-seeds`` / ``report-mechanism`` /
  ``report-control-seeds`` — seed-level aggregate reports.
- ``heldout`` — accuracy curves over a run's validation split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from grpo_decomp.eval.battery import run_battery
from grpo_decomp.eval.completions import (
    CompletionSet,
    SamplingConfig,
    load_completion_set,
    write_completion_set,
)
from grpo_decomp.eval.heldout import run_heldout_curve
from grpo_decomp.plugins import load_plugins
from grpo_decomp.prompts import EVAL_MAX_NEW_TOKENS
from grpo_decomp.registries import (
    DEFAULT_PROMPT_STRATEGY,
    EVAL_SETS,
    PROBES,
    PROMPT_STRATEGIES,
)
from grpo_decomp.report.control_seeds import aggregate_control_rows
from grpo_decomp.report.inputs import (
    base_and_correct_seeds,
    greedy_pass1,
    seed_label,
    validate_aligned_artifacts,
)
from grpo_decomp.report.mechanism import build_mechanism
from grpo_decomp.report.passk_seeds import aggregate_passk_seeds
from grpo_decomp.report.render import (
    render_control_decomposition,
    render_mechanism,
    render_passk_multiseed,
    render_seed_placebo,
)
from grpo_decomp.report.seeds import aggregate_placebo_comparison
from grpo_decomp.schemas import Record, record_json
from grpo_decomp.splits import dev_slice
from grpo_decomp.stats.compare import Comparison, compare

_BACKEND_CHOICES = ("auto", "transformers", "vllm")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `grpo-decomp` console script."""
    # Load study plugins first so the registries (eval sets, task profiles) are populated
    # before the parser reads `--set` choices from them.
    load_plugins()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ImportError as exc:
        print(f"grpo-decomp: missing generation backend dependency: {exc}", file=sys.stderr)
        return 1
    except (ValueError, ValidationError, OSError) as exc:
        print(f"grpo-decomp: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo-decomp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate_parser(sub)
    _add_battery_parser(sub)
    _add_report_seeds_parser(sub)
    _add_report_passk_seeds_parser(sub)
    _add_report_mechanism_parser(sub)
    _add_report_control_seeds_parser(sub)
    _add_heldout_parser(sub)
    return parser


def _add_generate_parser(sub: argparse._SubParsersAction) -> None:
    gen = sub.add_parser("generate", help="sample completions from a model over a set")
    gen.add_argument("--model", required=True, help="model id or checkpoint path")
    gen.add_argument("--revision", default=None, help="model revision (HF only)")
    gen.add_argument("--set", required=True, choices=sorted(EVAL_SETS), dest="set_name")
    gen.add_argument("--backend", default="auto", choices=_BACKEND_CHOICES)
    gen.add_argument(
        "--prompt-strategy",
        default=DEFAULT_PROMPT_STRATEGY,
        choices=sorted(PROMPT_STRATEGIES),
        dest="prompt_strategy",
        help="prompt strategy (must match the arm's training strategy)",
    )
    gen.add_argument("--n", type=int, default=1, help="completions per problem")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    gen.add_argument(
        "--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS, dest="max_new_tokens"
    )
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--limit", type=int, default=None, help="subset to N problems")
    gen.add_argument("--out", required=True, type=Path, help="output dir for the CompletionSet")
    gen.set_defaults(func=_cmd_generate)


def _add_battery_parser(sub: argparse._SubParsersAction) -> None:
    bat = sub.add_parser("battery", help="score one CompletionSet into a BatteryResult")
    bat.add_argument("--completions", required=True, type=Path, help="a CompletionSet dir")
    bat.add_argument("--k", type=int, nargs="+", default=[1], help="pass@k values")
    bat.add_argument("--out", type=Path, default=None, help="write JSON here (else stdout)")
    bat.set_defaults(func=_cmd_battery)


def _add_report_seeds_parser(sub: argparse._SubParsersAction) -> None:
    rsd = sub.add_parser(
        "report-seeds", help="aggregate the seed-level placebo comparison across replicates"
    )
    rsd.add_argument(
        "--battery-dirs",
        required=True,
        nargs="+",
        type=Path,
        dest="battery_dirs",
        help="one battery dir per seed (each with correct__<set> + random__<set>)",
    )
    rsd.add_argument("--task-set", default="gsm8k-test", dest="task_set")
    rsd.add_argument(
        "--out", type=Path, default=None, help="SeedPlaceboComparison JSON path (else stdout)"
    )
    rsd.set_defaults(func=_cmd_report_seeds)


def _add_report_passk_seeds_parser(sub: argparse._SubParsersAction) -> None:
    rpk = sub.add_parser(
        "report-passk-seeds",
        help="aggregate multi-seed pass@k coverage (base anchor vs per-seed correct)",
    )
    rpk.add_argument(
        "--completions-dir",
        required=True,
        type=Path,
        dest="completions_dir",
        help="dir with base__<set> + correct-seed<N>__<set> sampled CompletionSets",
    )
    rpk.add_argument("--task-set", default="gsm8k-test", dest="task_set")
    rpk.add_argument("--k", type=int, default=8, help="pass@k coverage level (default 8)")
    rpk.add_argument(
        "--out", type=Path, default=None, help="PassKMultiSeed JSON path (else stdout)"
    )
    rpk.set_defaults(func=_cmd_report_passk_seeds)


def _add_report_mechanism_parser(sub: argparse._SubParsersAction) -> None:
    rmc = sub.add_parser(
        "report-mechanism", help="per-problem reliability migration + completion-length shift"
    )
    rmc.add_argument(
        "--completions-dir",
        required=True,
        type=Path,
        dest="completions_dir",
        help="dir with base__<set> + correct-seed<N>__<set> sampled CompletionSets",
    )
    rmc.add_argument("--task-set", default="gsm8k-test", dest="task_set")
    rmc.add_argument("--k", type=int, default=8, help="pass@k envelope level (default 8)")
    rmc.add_argument("--tau", type=float, default=0.5, help="reliability threshold (default 0.5)")
    rmc.add_argument(
        "--out", type=Path, default=None, help="MechanismReport JSON path (else stdout)"
    )
    rmc.set_defaults(func=_cmd_report_mechanism)


def _add_report_control_seeds_parser(sub: argparse._SubParsersAction) -> None:
    rcs = sub.add_parser(
        "report-control-seeds",
        help="multi-seed section-3 controls with Holm family-wise correction",
    )
    rcs.add_argument(
        "--battery-dirs",
        required=True,
        nargs="+",
        type=Path,
        dest="battery_dirs",
        help="one battery dir per seed; the first holds base__<control> (seed-independent)",
    )
    rcs.add_argument("--task-set", default="gsm8k-test", dest="task_set")
    rcs.add_argument(
        "--control-sets",
        nargs="+",
        default=list(PROBES),
        dest="control_sets",
    )
    rcs.add_argument(
        "--out", type=Path, default=None, help="ControlDecomposition JSON path (else stdout)"
    )
    rcs.set_defaults(func=_cmd_report_control_seeds)


def _add_heldout_parser(sub: argparse._SubParsersAction) -> None:
    hld = sub.add_parser("heldout", help="held-out accuracy curve over a run's checkpoints")
    hld.add_argument(
        "--run", required=True, type=Path, help="a training run dir (provenance + checkpoints)"
    )
    hld.add_argument("--backend", default="auto", choices=_BACKEND_CHOICES)
    hld.add_argument(
        "--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS, dest="max_new_tokens"
    )
    hld.add_argument(
        "--out", type=Path, default=None, help="write JSON here (else <run>/heldout.json)"
    )
    hld.set_defaults(func=_cmd_heldout)


def _cmd_generate(args: argparse.Namespace) -> int:
    # Lazy import: only `generate` needs a backend, so CPU analysis stays lightweight.
    from grpo_decomp.eval.generate import generate_completion_set

    config = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        n=args.n,
        seed=args.seed,
    )
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")
    problems = EVAL_SETS[args.set_name]()
    if args.limit is not None:
        problems = dev_slice(problems, n=args.limit, seed=config.seed)

    completion_set = generate_completion_set(
        args.model,
        problems,
        config,
        backend=args.backend,
        model_revision=args.revision,
        prompt_strategy=args.prompt_strategy,
    )
    out = write_completion_set(completion_set, args.out)
    print(f"wrote {len(completion_set.items)} problems x {config.n} samples to {out}")
    return 0


def _cmd_battery(args: argparse.Namespace) -> int:
    completion_set = load_completion_set(args.completions)
    result = run_battery(
        completion_set.problem_set(), completion_set.completions_by_id(), k_values=args.k
    )
    payload = record_json(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote battery result to {args.out}")
    else:
        sys.stdout.write(payload)
    return 0


def _cmd_report_seeds(args: argparse.Namespace) -> int:
    comparisons = []
    seeds = []
    if args.task_set not in EVAL_SETS:
        raise ValueError(
            f"unknown report set {args.task_set!r}; known sets are {tuple(sorted(EVAL_SETS))}"
        )
    artifacts: dict[str, CompletionSet] = {}
    for battery_dir in args.battery_dirs:
        battery_dir = Path(battery_dir)
        correct = load_completion_set(battery_dir / f"correct__{args.task_set}")
        random_arm = load_completion_set(battery_dir / f"random__{args.task_set}")
        artifacts[f"{battery_dir.name}:correct"] = correct
        artifacts[f"{battery_dir.name}:random"] = random_arm
        comparisons.append(
            compare(
                "random",
                greedy_pass1(random_arm, "lenient"),
                "correct",
                greedy_pass1(correct, "lenient"),
            )
        )
        seeds.append(seed_label(battery_dir))
    validate_aligned_artifacts(args.task_set, artifacts)

    placebo_comparison = aggregate_placebo_comparison(comparisons, seeds, task=args.task_set)
    _write_json(placebo_comparison, args.out, "seed-level placebo comparison")
    sys.stdout.write(render_seed_placebo(placebo_comparison))
    return 0


def _cmd_report_passk_seeds(args: argparse.Namespace) -> int:
    base, correct_by_seed = base_and_correct_seeds(args.completions_dir, args.task_set)

    panel = aggregate_passk_seeds(base, correct_by_seed, task=args.task_set, k=args.k)
    _write_json(panel, args.out, f"multi-seed pass@{panel.k} panel")
    sys.stdout.write(render_passk_multiseed(panel))
    return 0


def _cmd_report_mechanism(args: argparse.Namespace) -> int:
    base, correct_by_seed = base_and_correct_seeds(args.completions_dir, args.task_set)
    correct = [completion_set for _seed, completion_set in correct_by_seed]

    report = build_mechanism(base, correct, task=args.task_set, k=args.k, tau=args.tau)
    _write_json(report, args.out, "mechanism report")
    sys.stdout.write(render_mechanism(report))
    return 0


def _cmd_report_control_seeds(args: argparse.Namespace) -> int:
    battery_dirs = [Path(d) for d in args.battery_dirs]
    seed0 = battery_dirs[0]  # the seed-0 battery holds the seed-independent base arm
    seeds = [seed_label(d) for d in battery_dirs]
    rows: list[tuple[str, str, list[Comparison]]] = []
    for control in args.control_sets:
        if control not in EVAL_SETS:
            raise ValueError(
                f"unknown control set {control!r}; known sets are {tuple(sorted(EVAL_SETS))}"
            )
        base = load_completion_set(seed0 / f"base__{control}")
        artifacts = {"base": base}
        base_grade = greedy_pass1(base, "lenient")
        comparisons = []
        for battery_dir in battery_dirs:
            correct = load_completion_set(battery_dir / f"correct__{control}")
            artifacts[f"{battery_dir.name}:correct"] = correct
            comparisons.append(
                compare("base", base_grade, "correct", greedy_pass1(correct, "lenient"))
            )
        validate_aligned_artifacts(control, artifacts)
        rows.append((control, PROBES.get(control, control), comparisons))

    decomp = aggregate_control_rows(rows, seeds, task=args.task_set)
    _write_json(decomp, args.out, "multi-seed control decomposition")
    sys.stdout.write(render_control_decomposition(decomp))
    return 0


def _write_json(record: Record, out: Path | None, label: str) -> None:
    """Write a result record as deterministic JSON when ``--out`` was given."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record_json(record), encoding="utf-8")
    print(f"wrote {label} to {out}")


def _cmd_heldout(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    config = SamplingConfig(temperature=0.0, n=1, max_new_tokens=args.max_new_tokens, seed=0)
    curve = run_heldout_curve(run_dir, config, backend=args.backend)
    for point in curve.points:
        print(
            f"  {point.checkpoint}: held-out acc {point.accuracy:.3f} ({point.n_correct}/{point.n})"
        )

    out = Path(args.out) if args.out is not None else run_dir / "heldout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(record_json(curve), encoding="utf-8")
    print(f"wrote held-out curve ({len(curve.points)} checkpoints) to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
