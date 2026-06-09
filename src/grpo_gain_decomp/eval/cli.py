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
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from grpo_gain_decomp.data import (
    dev_slice,
    load_countdown,
    load_gsm8k,
    load_gsm8k_platinum,
    load_gsm_plus,
    load_gsm_symbolic,
    validation_split,
)
from grpo_gain_decomp.eval.battery import BatteryResult, grade, run_battery
from grpo_gain_decomp.eval.completions import (
    CompletionSet,
    SamplingConfig,
    load_completion_set,
    write_completion_set,
)
from grpo_gain_decomp.report.control_seeds import aggregate_control_rows
from grpo_gain_decomp.report.decomposition import DecompositionRow, build_decomposition
from grpo_gain_decomp.report.mechanism import build_mechanism
from grpo_gain_decomp.report.passk_seeds import aggregate_passk_seeds
from grpo_gain_decomp.report.render import render_table, write_summary
from grpo_gain_decomp.report.seeds import aggregate_placebo_comparison
from grpo_gain_decomp.schemas import ProblemSet, Record
from grpo_gain_decomp.stats.compare import Comparison, compare
from grpo_gain_decomp.train.provenance import RunProvenance

#: The named problem sets `generate --set` can target.
SETS: dict[str, Callable[[], ProblemSet]] = {
    "gsm8k-test": lambda: load_gsm8k("test"),
    "gsm8k-train": lambda: load_gsm8k("train"),
    "dev": lambda: dev_slice(load_gsm8k("test")),
    "gsm-symbolic": lambda: load_gsm_symbolic("main"),
    "gsm-plus": lambda: load_gsm_plus("test"),
    "gsm8k-platinum": load_gsm8k_platinum,
    "countdown-test": lambda: load_countdown("test"),
    "countdown-dev": lambda: load_countdown("dev"),
}

#: The arms a decomposition compares; `correct` is compared against `base` and `random`.
ARMS = ("base", "correct", "random")

#: What each control set actually probes (the report's controlled row labels).
PROBES = {
    "gsm-symbolic": "memorization (templated renumbering)",
    "gsm8k-platinum": "label noise (cleaned labels)",
    "gsm-plus": "robustness (adversarial perturbation)",
}


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


def _base_and_correct_seeds(
    root: Path, task_set: str
) -> tuple[CompletionSet, list[tuple[int | str, CompletionSet]]]:
    """Load ``base__<set>`` plus every ``correct-seed<N>__<set>`` under `root`.

    The seed-replicate layout both `report-passk-seeds` and `report-mechanism` consume.
    Numeric seeds sort first (in order), any non-numeric labels after — never int vs str.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"completions dir {root} does not exist")
    suffix = f"__{task_set}"
    base = load_completion_set(root / f"base{suffix}")
    correct_by_seed: list[tuple[int | str, CompletionSet]] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("correct-seed") and sub.name.endswith(suffix):
            label = sub.name[: -len(suffix)].partition("-seed")[2]
            seed: int | str = int(label) if label.isdigit() else label
            correct_by_seed.append((seed, load_completion_set(sub)))
    if not correct_by_seed:
        raise ValueError(f"no 'correct-seed<N>{suffix}' dirs under {root}")
    correct_by_seed.sort(key=lambda pair: (isinstance(pair[0], str), pair[0]))
    return base, correct_by_seed


def _seed_label(battery_dir: Path) -> int | str:
    """Recover a seed label from a battery dir name (``battery`` -> 0, ``battery-seed2`` -> 2)."""
    name = Path(battery_dir).name
    if name == "battery":
        return 0
    head, _, tail = name.rpartition("-seed")
    return int(tail) if head and tail.isdigit() else name


def _discover_completion_sets(root: Path) -> dict[str, dict[str, CompletionSet]]:
    """Load ``<arm>__<set>`` subdirectories into ``{set: {arm: CompletionSet}}``."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"completions dir {root} does not exist")
    grouped: dict[str, dict[str, CompletionSet]] = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or "__" not in sub.name:
            continue
        arm, _, slug = sub.name.partition("__")
        if arm not in ARMS:
            continue
        grouped.setdefault(slug, {})[arm] = load_completion_set(sub)
    if not grouped:
        raise ValueError(f"no '<arm>__<set>' completion dirs found under {root}")
    return grouped


def _validate_report_artifacts(grouped: dict[str, dict[str, CompletionSet]]) -> None:
    """Require report artifacts to match the registered eval sets they claim to be."""
    for slug, arms in grouped.items():
        if slug not in SETS:
            raise ValueError(f"unknown report set {slug!r}; known sets are {tuple(sorted(SETS))}")
        expected = SETS[slug]()
        for arm, completion_set in arms.items():
            _validate_completion_set(slug, arm, completion_set, expected)


def _validate_completion_set(
    slug: str, arm: str, completion_set: CompletionSet, expected: ProblemSet
) -> None:
    if completion_set.provenance.dataset != expected.source:
        raise ValueError(
            f"{arm}__{slug}: dataset metadata does not match registered set "
            f"{expected.source.model_dump()}"
        )
    expected_ids = tuple(problem.id for problem in expected)
    actual_ids = tuple(item.problem.id for item in completion_set.items)
    if len(actual_ids) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError(
            f"{arm}__{slug}: problem ids do not match registered set "
            f"(expected {len(expected_ids)}, got {len(actual_ids)})"
        )


def _pass1(completion_set: CompletionSet, policy: str) -> dict[str, bool]:
    """Grade the first sample per problem (pass@1) under `policy`."""
    sampling = completion_set.provenance.sampling
    if sampling.n != 1 or sampling.temperature != 0.0:
        raise ValueError(
            "greedy pass@1 artifacts must have sampling.n=1 and temperature=0.0; "
            f"got n={sampling.n}, temperature={sampling.temperature}"
        )
    first = {item.problem.id: item.samples[0] for item in completion_set.items}
    return grade(completion_set.problem_set(), first, policy=policy)


def _control_row(slug: str, arms: dict[str, CompletionSet]) -> DecompositionRow:
    comparison = compare(
        "base", _pass1(arms["base"], "lenient"), "correct", _pass1(arms["correct"], "lenient")
    )
    return DecompositionRow(
        control=f"control ({slug})", probes=PROBES.get(slug, slug), comparison=comparison
    )


def _battery_at(completion_set: CompletionSet, k_values: list[int]) -> BatteryResult:
    return run_battery(
        completion_set.problem_set(), completion_set.completions_by_id(), k_values=k_values
    )


def _vanilla_at(battery: BatteryResult, k: int) -> float:
    """The vanilla pass@k at exactly `k` from a battery result."""
    for entry in battery.pass_at_k:
        if entry.k == k:
            return entry.vanilla
    raise ValueError(f"pass@{k} was not computed")


def _elicitation_note(base: CompletionSet, correct: CompletionSet) -> str:
    """The elicitation / capability-expansion panel line.

    When both arms are sampled at n>1, report the pass@k curve (base vs correct at
    matched k) — the certified-expansion readout: higher pass@k coverage means
    new capability, not just improved pass@1 reliability. Otherwise fall back to
    the base pass@k vs correct pass@1 elicitation line, or a plain pass@1 line when n==1.
    """
    n_base = base.provenance.sampling.n
    n_correct = correct.provenance.sampling.n
    if n_base > 1 and n_correct > 1:
        k = min(n_base, n_correct)
        base_battery = _battery_at(base, sorted({1, k}))
        correct_battery = _battery_at(correct, sorted({1, k}))
        base_k = _vanilla_at(base_battery, k)
        correct_k = _vanilla_at(correct_battery, k)
        return (
            f"pass@k curve: base pass@{k}={base_k:.2f} vs correct pass@{k}={correct_k:.2f} "
            f"(Δ={correct_k - base_k:+.2f}); pass@1 base={base_battery.lenient_accuracy:.2f}, "
            f"correct={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    base_battery = _battery_at(base, sorted({1, n_base}))
    correct_battery = _battery_at(correct, [1])
    if n_base > 1:
        return (
            f"base pass@{n_base}={base_battery.pass_at_k[-1].vanilla:.2f} vs "
            f"correct pass@1={correct_battery.lenient_accuracy:.2f} "
            f"(code-reasoning base={base_battery.code_reasoning_frequency:.2f}, "
            f"correct={correct_battery.code_reasoning_frequency:.2f})"
        )
    return (
        f"pass@1: base={base_battery.lenient_accuracy:.2f}, "
        f"correct={correct_battery.lenient_accuracy:.2f}; "
        "high-n pass@k coverage deferred to a Phase-2 sampling run"
    )


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


def select_checkpoint(points: list[dict], rule: str, final_step: int) -> tuple[str, int]:
    """Realize a checkpoint-selection rule over a held-out curve → (checkpoint dir, step).

    `final` always takes the end-of-training checkpoint; `best_on_validation` takes the
    highest held-out accuracy, breaking ties toward the later (more-trained) step.
    """
    if rule == "final":
        if not any(point["checkpoint"] == "final" for point in points):
            raise ValueError("final checkpoint selection needs a discovered final checkpoint")
        return "final", final_step
    if rule == "best_on_validation":
        if not points:
            raise ValueError("best_on_validation needs a non-empty held-out curve")
        best = max(
            points,
            key=lambda p: (p["accuracy"], p["step"] if p["step"] is not None else final_step),
        )
        return best["checkpoint"], best["step"] if best["step"] is not None else final_step
    raise ValueError(f"unknown checkpoint_selection rule {rule!r}")


def _validation_for_run(provenance: RunProvenance) -> ProblemSet:
    """Reconstruct the exact held-out split a run used (deterministic from the source)."""
    ref = provenance.dataset
    if ref.name == "countdown":
        # Countdown ships a dedicated, seed-independent validation split; regenerate it.
        return load_countdown("validation")
    if ref.name == "openai/gsm8k":
        train = load_gsm8k(ref.split, revision=ref.revision)
        _, validation = validation_split(train, n=provenance.validation_size, seed=provenance.seed)
        return validation
    raise ValueError(f"held-out reconstruction supports gsm8k train or countdown, got {ref.name!r}")


def _discover_checkpoints(run_dir: Path) -> list[Path]:
    """Saved checkpoints under ``run_dir/checkpoints``, sorted by step with ``final`` last."""
    root = Path(run_dir) / "checkpoints"
    if not root.is_dir():
        raise ValueError(f"no checkpoints/ directory under {run_dir}")
    numbered: list[Path] = []
    final: Path | None = None
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "final":
            final = sub
        elif sub.name.startswith("checkpoint-") and sub.name.split("-", 1)[1].isdigit():
            numbered.append(sub)
    numbered.sort(key=lambda path: int(path.name.split("-", 1)[1]))
    if final is not None:
        numbered.append(final)
    if not numbered:
        raise ValueError(f"no 'checkpoint-<step>' or 'final' dirs under {root}")
    return numbered


if __name__ == "__main__":
    sys.exit(main())
