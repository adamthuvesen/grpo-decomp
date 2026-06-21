"""`grpo-decomp`: generate completions, score a checkpoint, drive the decomposition.

Three subcommands, split along the phase contract:

- ``generate`` — the only model-loading command (a generation backend required). One
  model (base or checkpoint) over one problem set -> a `CompletionSet` artifact.
- ``battery``  — a `CompletionSet` -> a `BatteryResult` (the Phase-0 base-model smoke
  and the eval-battery end-to-end check). CPU-only.
- ``report``   — `CompletionSet`s across arms (base/correct/random) and sets ->
  the deterministic decomposition table + ``summary.json``. CPU-only, offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from grpo_gain_decomp.data import dev_slice
from grpo_gain_decomp.eval.battery import grade, run_battery
from grpo_gain_decomp.eval.completions import (
    SamplingConfig,
    load_completion_set,
    write_completion_set,
)
from grpo_gain_decomp.eval.heldout import (
    discover_checkpoints as _discover_checkpoints,
)
from grpo_gain_decomp.eval.heldout import (
    select_checkpoint,
)
from grpo_gain_decomp.eval.heldout import (
    validation_for_run as _validation_for_run,
)
from grpo_gain_decomp.eval.registry import ARMS, PROBES, SETS
from grpo_gain_decomp.eval.report_inputs import (
    base_and_correct_seeds as _base_and_correct_seeds,
)
from grpo_gain_decomp.eval.report_inputs import (
    control_row as _control_row,
)
from grpo_gain_decomp.eval.report_inputs import (
    discover_completion_sets as _discover_completion_sets,
)
from grpo_gain_decomp.eval.report_inputs import (
    elicitation_note as _elicitation_note,
)
from grpo_gain_decomp.eval.report_inputs import (
    greedy_pass1 as _pass1,
)
from grpo_gain_decomp.eval.report_inputs import (
    seed_label as _seed_label,
)
from grpo_gain_decomp.eval.report_inputs import (
    validate_report_artifacts as _validate_report_artifacts,
)
from grpo_gain_decomp.report.control_seeds import aggregate_control_rows
from grpo_gain_decomp.report.decomposition import DecompositionRow, build_decomposition
from grpo_gain_decomp.report.mechanism import build_mechanism
from grpo_gain_decomp.report.passk_seeds import aggregate_passk_seeds
from grpo_gain_decomp.report.render import render_table, write_summary
from grpo_gain_decomp.report.seeds import aggregate_placebo_comparison
from grpo_gain_decomp.schemas import Record
from grpo_gain_decomp.stats.compare import Comparison, compare
from grpo_gain_decomp.train.provenance import RunProvenance


def main(argv: list[str] | None = None) -> int:
    """Entry point for the `grpo-decomp` console script."""
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

    gen = sub.add_parser("generate", help="sample completions from a model over a set")
    gen.add_argument("--model", required=True, help="model id or checkpoint path")
    gen.add_argument("--revision", default=None, help="model revision (HF only)")
    gen.add_argument("--set", required=True, choices=sorted(SETS), dest="set_name")
    gen.add_argument("--backend", default="auto", choices=("auto", "transformers", "vllm"))
    gen.add_argument("--n", type=int, default=1, help="completions per problem")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    gen.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--limit", type=int, default=None, help="subset to N problems (smoke)")
    gen.add_argument("--out", required=True, type=Path, help="output dir for the CompletionSet")
    gen.set_defaults(func=_cmd_generate)

    bat = sub.add_parser("battery", help="score one CompletionSet into a BatteryResult")
    bat.add_argument("--completions", required=True, type=Path, help="a CompletionSet dir")
    bat.add_argument("--k", type=int, nargs="+", default=[1], help="pass@k values")
    bat.add_argument("--out", type=Path, default=None, help="write JSON here (else stdout)")
    bat.set_defaults(func=_cmd_battery)

    rep = sub.add_parser("report", help="decompose the gain across arms and sets")
    rep.add_argument("--completions-dir", required=True, type=Path, dest="completions_dir")
    rep.add_argument("--task-set", default="gsm8k-test", dest="task_set")
    rep.add_argument("--base-model", default=None, dest="base_model", help="override table label")
    rep.add_argument("--out", required=True, type=Path, help="output dir for summary.json + table")
    rep.set_defaults(func=_cmd_report)

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
        "--out", type=Path, default=None, help="Pass8MultiSeed JSON path (else stdout)"
    )
    rpk.set_defaults(func=_cmd_report_passk_seeds)

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
        default=["gsm-symbolic", "gsm-plus", "gsm8k-platinum"],
        dest="control_sets",
    )
    rcs.add_argument(
        "--out", type=Path, default=None, help="ControlDecomposition JSON path (else stdout)"
    )
    rcs.set_defaults(func=_cmd_report_control_seeds)

    hld = sub.add_parser("heldout", help="held-out accuracy curve over a run's checkpoints")
    hld.add_argument(
        "--run", required=True, type=Path, help="a training run dir (provenance + checkpoints)"
    )
    hld.add_argument("--backend", default="auto", choices=("auto", "transformers", "vllm"))
    hld.add_argument("--n", type=int, default=1, help="samples per problem (1 = greedy pass@1)")
    hld.add_argument("--temperature", type=float, default=0.0)
    hld.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    hld.add_argument("--seed", type=int, default=0)
    hld.add_argument(
        "--out", type=Path, default=None, help="write JSON here (else <run>/heldout.json)"
    )
    hld.set_defaults(func=_cmd_heldout)

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    # Lazy import: only `generate` needs a backend, so `battery`/`report` stay CPU-only.
    from grpo_gain_decomp.eval.generate import generate_completion_set

    config = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        n=args.n,
        seed=args.seed,
    )
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")
    problems = SETS[args.set_name]()
    if args.limit is not None:
        problems = dev_slice(problems, n=args.limit, seed=config.seed)

    completion_set = generate_completion_set(
        args.model, problems, config, backend=args.backend, model_revision=args.revision
    )
    out = write_completion_set(completion_set, args.out)
    print(f"wrote {len(completion_set.items)} problems x {config.n} samples to {out}")
    return 0


def _cmd_battery(args: argparse.Namespace) -> int:
    completion_set = load_completion_set(args.completions)
    result = run_battery(
        completion_set.problem_set(), completion_set.completions_by_id(), k_values=args.k
    )
    payload = json.dumps(result.model_dump(), sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote battery result to {args.out}")
    else:
        sys.stdout.write(payload)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    grouped = _discover_completion_sets(args.completions_dir)
    task = args.task_set
    if task not in grouped or not all(arm in grouped[task] for arm in ARMS):
        present = {s: sorted(a) for s, a in grouped.items()}
        raise ValueError(f"report needs base/correct/random for task set {task!r}; found {present}")
    _validate_report_artifacts(grouped)

    base = grouped[task]["base"]
    correct = grouped[task]["correct"]
    random_arm = grouped[task]["random"]

    base_lenient = _pass1(base, "lenient")
    correct_lenient = _pass1(correct, "lenient")

    raw_gain = DecompositionRow(
        control="raw gain",
        probes=f"correct vs base on {task}",
        comparison=compare("base", base_lenient, "correct", correct_lenient),
    )
    placebo = DecompositionRow(
        control="placebo (correct - random)",
        probes="non-correctness-driven gain",
        comparison=compare("random", _pass1(random_arm, "lenient"), "correct", correct_lenient),
    )
    format_row = DecompositionRow(
        control="format sensitivity",
        probes="lenient vs strict (same completions)",
        comparison=compare(
            "correct/strict", _pass1(correct, "strict"), "correct/lenient", correct_lenient
        ),
    )
    control_rows = [
        _control_row(slug, arms)
        for slug, arms in sorted(grouped.items())
        if slug != task and "base" in arms and "correct" in arms
    ]

    decomposition = build_decomposition(
        base_model=args.base_model or base.provenance.model,
        task=task,
        seeds=1,
        raw_gain=raw_gain,
        control_rows=control_rows,
        format_row=format_row,
        placebo=placebo,
        elicitation_note=_elicitation_note(base, correct),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(decomposition, out / "summary.json")
    table = render_table(decomposition)
    (out / "decomposition.md").write_text(table, encoding="utf-8")
    sys.stdout.write(table)
    return 0


def _cmd_report_seeds(args: argparse.Namespace) -> int:
    comparisons = []
    seeds = []
    for battery_dir in args.battery_dirs:
        battery_dir = Path(battery_dir)
        correct = load_completion_set(battery_dir / f"correct__{args.task_set}")
        random_arm = load_completion_set(battery_dir / f"random__{args.task_set}")
        comparisons.append(
            compare("random", _pass1(random_arm, "lenient"), "correct", _pass1(correct, "lenient"))
        )
        seeds.append(_seed_label(battery_dir))

    placebo_comparison = aggregate_placebo_comparison(comparisons, seeds, task=args.task_set)
    header = f"# Placebo comparison over {placebo_comparison.n_seeds} seed(s) - {args.task_set}"
    lines = [header, "", placebo_comparison.headline()]
    lines += ["", "| seed | random | correct | Δ (pp) |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {s} | {ra * 100:.1f}% | {ca * 100:.1f}% | {d * 100:+.1f} |"
        for s, ra, ca, d in zip(
            placebo_comparison.seeds,
            placebo_comparison.per_seed_random_acc,
            placebo_comparison.per_seed_correct_acc,
            placebo_comparison.per_seed_delta,
            strict=True,
        )
    ]
    _write_json(placebo_comparison, args.out, "seed-level placebo comparison")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_report_passk_seeds(args: argparse.Namespace) -> int:
    base, correct_by_seed = _base_and_correct_seeds(args.completions_dir, args.task_set)

    panel = aggregate_passk_seeds(base, correct_by_seed, task=args.task_set, k=args.k)
    lines = [
        f"# Multi-seed pass@{panel.k} coverage - {args.task_set}",
        "",
        panel.headline(),
        panel.cot_headline(),
        "",
        f"| seed | correct pass@1 | correct pass@{panel.k} |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {p1 * 100:.1f}% | {pk * 100:.1f}% |"
        for s, p1, pk in zip(
            panel.seeds, panel.per_seed_correct_pass1, panel.per_seed_correct_passk, strict=True
        )
    ]
    lines += [
        "",
        f"base pass@1 {panel.base_pass1 * 100:.1f}% · base pass@{panel.k} "
        f"{panel.base_passk * 100:.1f}% "
        f"[{panel.base_passk_ci_low * 100:.1f}, {panel.base_passk_ci_high * 100:.1f}]",
        f"base CoT-gated pass@{panel.k} {panel.base_cot_passk * 100:.1f}% "
        f"[{panel.base_cot_passk_ci_low * 100:.1f}, {panel.base_cot_passk_ci_high * 100:.1f}]",
    ]
    _write_json(panel, args.out, f"multi-seed pass@{panel.k} panel")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_report_mechanism(args: argparse.Namespace) -> int:
    base, correct_by_seed = _base_and_correct_seeds(args.completions_dir, args.task_set)
    correct = [completion_set for _seed, completion_set in correct_by_seed]

    report = build_mechanism(base, correct, task=args.task_set, k=args.k, tau=args.tau)
    lines = [
        f"# Mechanism - {args.task_set}",
        "",
        report.headline(),
        "",
        f"base already reliable {report.frac_base_already_reliable * 100:.1f}% · "
        f"migrated {report.frac_migrated_to_reliable * 100:.1f}% · "
        f"new {report.frac_new_capability * 100:.1f}% · "
        f"still hard {report.frac_still_hard * 100:.1f}%",
    ]
    _write_json(report, args.out, "mechanism report")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_report_control_seeds(args: argparse.Namespace) -> int:
    battery_dirs = [Path(d) for d in args.battery_dirs]
    seed0 = battery_dirs[0]  # the seed-0 battery holds the seed-independent base arm
    seeds = [_seed_label(d) for d in battery_dirs]
    rows: list[tuple[str, str, list[Comparison]]] = []
    for control in args.control_sets:
        base_grade = _pass1(load_completion_set(seed0 / f"base__{control}"), "lenient")
        comparisons = []
        for battery_dir in battery_dirs:
            correct = load_completion_set(battery_dir / f"correct__{control}")
            comparisons.append(compare("base", base_grade, "correct", _pass1(correct, "lenient")))
        rows.append((control, PROBES.get(control, control), comparisons))

    decomp = aggregate_control_rows(rows, seeds, task=args.task_set)
    lines = [
        f"# Multi-seed controls - {args.task_set}",
        "",
        decomp.headline(),
        "",
        "| control | probes | Δ (pp) | 95% CI | p (raw) | p (Holm) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.control} | {r.probes} | {r.mean_delta * 100:+.1f} | "
        f"[{r.ci_low * 100:.1f}, {r.ci_high * 100:.1f}] | {r.p_value:.3g} | "
        f"{r.p_value_holm:.3g}{' *' if r.significant else ''} |"
        for r in decomp.rows
    ]
    _write_json(decomp, args.out, "multi-seed control decomposition")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _write_json(record: Record, out: Path | None, label: str) -> None:
    """Write a result record as deterministic JSON when ``--out`` was given."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(record.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {label} to {out}")


def _cmd_heldout(args: argparse.Namespace) -> int:
    # Lazy import: only this command loads a model, so it stays off the CPU-only path.
    from grpo_gain_decomp.eval.generate import generate

    run_dir = Path(args.run)
    provenance = RunProvenance.model_validate_json(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    validation = _validation_for_run(provenance)
    config = SamplingConfig(
        temperature=args.temperature, n=args.n, max_new_tokens=args.max_new_tokens, seed=args.seed
    )

    points = []
    for checkpoint in _discover_checkpoints(run_dir):
        samples = generate(str(checkpoint), validation, config, backend=args.backend)
        graded = grade(validation, {pid: s[0] for pid, s in samples.items()}, policy="lenient")
        n_correct = sum(graded.values())
        accuracy = n_correct / len(graded)
        step = None if checkpoint.name == "final" else int(checkpoint.name.split("-", 1)[1])
        points.append(
            {
                "checkpoint": checkpoint.name,
                "step": step,
                "accuracy": accuracy,
                "n_correct": n_correct,
                "n": len(graded),
            }
        )
        print(f"  {checkpoint.name}: held-out acc {accuracy:.3f} ({n_correct}/{len(graded)})")

    payload = {
        "run": str(run_dir),
        "validation_size": len(validation),
        "policy": "lenient",
        "points": points,
    }
    out = Path(args.out) if args.out is not None else run_dir / "heldout.json"
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"wrote held-out curve ({len(points)} checkpoints) to {out}")

    # Realize the pre-registered checkpoint-selection rule and record it in provenance,
    # so the decomposition provably uses the checkpoint the rule actually chose.
    name, step = select_checkpoint(
        points, provenance.checkpoint_selection, provenance.grpo.max_steps
    )
    selected = provenance.model_copy(update={"selected_step": step, "selected_checkpoint": name})
    (run_dir / "provenance.json").write_text(
        json.dumps(selected.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"selected {name} (step {step}) per rule '{provenance.checkpoint_selection}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
