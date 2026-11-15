"""Inventory and reproduce the reference study's published artifacts on CPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import JsonValue, TypeAdapter

from grpo_decomp.eval.completions import load_completion_set
from grpo_decomp.schemas import Record

_SEEDS = tuple(str(seed) for seed in range(6))
_COUNTDOWN_SEEDS = ("0", "1", "2")
_ESME_SEEDS = tuple(str(seed) for seed in range(214, 220))
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class StudyWorkflow(Record):
    """One published artifact, its recorded seed axis, and its reproduction prerequisites."""

    name: str
    task_set: str | None
    seeds: tuple[str, ...]
    output: Path
    reproduction: Literal["committed-completions", "external-completions", "published-diagnostic"]
    module: str | None
    arguments: tuple[str, ...] = ()
    completion_sets: tuple[Path, ...] = ()
    output_directory: bool = False
    comparison: Literal["exact", "semantic"] = "semantic"

    def command(self, output: Path) -> list[str]:
        """Build the CPU command that writes the requested temporary output."""
        if self.module is None:
            raise ValueError(
                f"{self.name}: no reproduction command exists for the source diagnostic"
            )
        target = output.parent if self.output_directory else output
        return [sys.executable, "-m", self.module, *self.arguments, "--out", str(target)]


def _sampled(
    name: str, command: str, root: str, task: str, output: str, seeds: tuple[str, ...]
) -> StudyWorkflow:
    paths = (
        Path(root) / f"base__{task}",
        *(Path(root) / f"correct-seed{seed}__{task}" for seed in seeds),
    )
    return StudyWorkflow(
        name=name,
        task_set=task,
        seeds=seeds,
        output=Path(output),
        reproduction="external-completions",
        module="llm_grpo_gains.cli",
        arguments=(command, "--completions-dir", root, "--task-set", task),
        completion_sets=paths,
    )


def _battery(
    name: str,
    command: str,
    root: str,
    task: str,
    output: str,
    seeds: tuple[str, ...],
    *,
    controls: tuple[str, ...] = (),
    arm: str | None = None,
) -> StudyWorkflow:
    directories = tuple(Path(root if seed == "0" else f"{root}-seed{seed}") for seed in seeds)
    if controls:
        paths = tuple(directories[0] / f"base__{control}" for control in controls) + tuple(
            directory / f"correct__{control}" for directory in directories for control in controls
        )
    elif arm is not None:
        paths = (
            directories[0] / f"base__{task}",
            *(directory / f"{arm}__{task}" for directory in directories),
        )
    else:
        paths = tuple(
            directory / f"{role}__{task}"
            for directory in directories
            for role in ("correct", "random")
        )
    arguments = (
        command,
        "--battery-dirs",
        *(str(path) for path in directories),
        "--task-set",
        task,
    )
    if controls:
        arguments += ("--control-sets", *controls)
    if arm is not None:
        arguments += ("--arm", arm)
    return StudyWorkflow(
        name=name,
        task_set=task,
        seeds=seeds,
        output=Path(output),
        reproduction="external-completions",
        module="llm_grpo_gains.cli",
        arguments=arguments,
        completion_sets=paths,
    )


def study_workflows() -> tuple[StudyWorkflow, ...]:
    """Return the study manifest. Seed labels follow each listed published result."""
    workflows = [
        _battery(
            "gsm8k-placebo",
            "report-seeds",
            "runs/battery",
            "gsm8k-test",
            "results/seed-placebo-comparison.json",
            _SEEDS,
        ),
        _battery(
            "countdown-placebo",
            "report-seeds",
            "runs/battery-countdown",
            "countdown-test",
            "results/countdown/seed-placebo-comparison.json",
            _COUNTDOWN_SEEDS,
        ),
        _battery(
            "gsm8k-controls",
            "report-control-seeds",
            "runs/battery",
            "gsm8k-test",
            "results/decomposition-multiseed.json",
            _SEEDS,
            controls=("gsm-symbolic", "gsm-plus", "gsm8k-platinum"),
        ),
        _battery(
            "offpolicy-placebo",
            "report-arm-vs-base",
            "runs/offpolicy/battery",
            "gsm8k-test",
            "results/offpolicy-placebo.json",
            _COUNTDOWN_SEEDS,
            arm="random",
        ),
    ]
    for name, root, task, directory, seeds in (
        ("gsm8k", "runs/passk-multiseed", "gsm8k-test", "results", _SEEDS),
        (
            "countdown",
            "runs/passk-multiseed-countdown",
            "countdown-test",
            "results/countdown",
            _COUNTDOWN_SEEDS,
        ),
        (
            "countdown-math",
            "runs/passk-multiseed-countdown-math",
            "countdown-test",
            "results/countdown-math",
            _COUNTDOWN_SEEDS,
        ),
    ):
        for suffix, command, filename in (
            ("coverage", "report-passk-seeds", "pass8-multiseed.json"),
            ("mechanism", "report-mechanism", "mechanism.json"),
        ):
            workflows.append(
                _sampled(f"{name}-{suffix}", command, root, task, f"{directory}/{filename}", seeds)
            )
    for task, filename in (
        ("gsm-symbolic", "pass8-symbolic.json"),
        ("gsm8k-platinum", "pass8-platinum.json"),
    ):
        workflows.append(
            _sampled(
                task,
                "report-passk-seeds",
                "runs/passk-multiseed",
                task,
                f"results/decontam/{filename}",
                _SEEDS,
            )
        )
    esme_root = Path("results/esme-countdown/completions")
    base = esme_root / "base__esme-countdown"
    seed_args = tuple(
        argument
        for seed in _ESME_SEEDS
        for argument in ("--seed", f"{seed}={esme_root}/seed{seed}")
    )
    workflows.append(
        StudyWorkflow(
            name="esme",
            task_set="esme-countdown",
            seeds=_ESME_SEEDS,
            output=Path("results/esme-countdown/sampled_multiseed_summary.json"),
            reproduction="committed-completions",
            module="llm_grpo_gains.esme_analysis",
            arguments=(*seed_args, "--base-completions-dir", str(base), "--legacy"),
            completion_sets=(
                base,
                *(
                    esme_root / f"seed{seed}" / f"{role}__esme-countdown"
                    for seed in _ESME_SEEDS
                    for role in ("correct", "random")
                ),
            ),
            output_directory=True,
            comparison="exact",
        )
    )
    workflows.append(
        workflows[-1].model_copy(
            update={
                "name": "esme-corrected",
                "output": Path("results/esme-countdown/sampled_multiseed_summary.corrected.json"),
                "arguments": (*seed_args, "--base-completions-dir", str(base)),
                "comparison": "semantic",
            }
        )
    )
    for name, output, seeds in (
        ("single-seed-controls", "results/single-seed-controls.json", ("0",)),
        ("offpolicy-clip-ratios", "results/offpolicy-clip-ratios.json", _COUNTDOWN_SEEDS),
    ):
        workflows.append(
            StudyWorkflow(
                name=name,
                task_set="gsm8k-test",
                seeds=seeds,
                output=Path(output),
                reproduction="published-diagnostic",
                module=None,
            )
        )
    return tuple(workflows)


def prerequisites(workflow: StudyWorkflow, root: Path) -> tuple[str, ...]:
    """Report every missing or malformed prerequisite before any report command runs."""
    errors = []
    if workflow.module is None:
        errors.append(f"{workflow.name}: original diagnostic source is not committed")
    if not (root / workflow.output).is_file():
        errors.append(f"{workflow.name}: missing published artifact {workflow.output}")
    for relative in workflow.completion_sets:
        path = root / relative
        missing = [
            name for name in ("provenance.json", "completions.jsonl") if not (path / name).is_file()
        ]
        if missing:
            errors.extend(f"{workflow.name}: missing {relative / name}" for name in missing)
            continue
        try:
            load_completion_set(path)
        except (ValueError, OSError) as exc:
            errors.append(f"{workflow.name}: invalid {relative}: {exc}")
    return tuple(errors)


def _scientific_payload(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _scientific_payload(item)
            for key, item in value.items()
            if key not in {"analysis", "analyses"} and item is not None
        }
    if isinstance(value, list):
        return [_scientific_payload(item) for item in value]
    return value


def _keeps_published(expected: JsonValue, actual: JsonValue) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _keeps_published(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_keeps_published(a, b) for a, b in zip(expected, actual, strict=True))
        )
    return expected == actual


def compare_outputs(expected: Path, actual: Path, comparison: str) -> bool:
    """Compare bytes, or require every published scientific field to survive unchanged.

    Semantic mode ignores analysis provenance and allows fields a newer report adds,
    so a published result that predates a field still reproduces.
    """
    if comparison == "exact":
        return expected.read_bytes() == actual.read_bytes()
    if comparison != "semantic":
        raise ValueError(f"unknown comparison {comparison!r}")
    return _keeps_published(
        _scientific_payload(_JSON.validate_json(expected.read_text())),
        _scientific_payload(_JSON.validate_json(actual.read_text())),
    )


def reproduce(
    workflows: Sequence[StudyWorkflow], root: Path, comparison: str | None = None
) -> None:
    """Validate all inputs, then reproduce and compare reports in a temporary directory."""
    errors = [error for workflow in workflows for error in prerequisites(workflow, root)]
    if errors:
        raise ValueError("\n".join(errors))
    with TemporaryDirectory(prefix="grpo-study-") as temporary:
        for workflow in workflows:
            output = Path(temporary) / workflow.output
            output.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                workflow.command(output), cwd=root, capture_output=True, text=True, check=False
            )
            if result.returncode:
                raise ValueError(f"{workflow.name}: reproduction failed: {result.stderr.strip()}")
            if not output.is_file():
                raise ValueError(f"{workflow.name}: command did not write {output.name}")
            mode = comparison or workflow.comparison
            if not compare_outputs(root / workflow.output, output, mode):
                raise ValueError(
                    f"{workflow.name}: {mode} comparison differs from {workflow.output}"
                )
            print(f"{workflow.name}: {mode} comparison matches {workflow.output}")


def aggregate(workflows: Sequence[StudyWorkflow], root: Path) -> None:
    """Regenerate explicitly requested published aggregates after all inputs pass validation."""
    errors = [error for workflow in workflows for error in prerequisites(workflow, root)]
    if errors:
        raise ValueError("\n".join(errors))
    for workflow in workflows:
        result = subprocess.run(
            workflow.command(root / workflow.output),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"{workflow.name}: aggregation failed: {result.stderr.strip()}")
        print(f"{workflow.name}: wrote {workflow.output}")


def check_figures(root: Path) -> None:
    """Regenerate figures in a temporary directory and compare rendered content."""
    from PIL import Image

    from llm_grpo_gains.figures import render_figures

    # The published PNGs are rendered on macOS, and other platforms lay out text
    # differently, so PNGs are compared exactly only there. Elsewhere every figure must
    # still render; `reproduce` checks the numbers exactly.
    exact_pixels = sys.platform == "darwin"
    with TemporaryDirectory(prefix="grpo-figures-") as temporary:
        output = Path(temporary)
        differences = []
        for actual in render_figures(root / "results", output):
            relative = actual.relative_to(output)
            expected = root / "results" / relative
            if not expected.is_file():
                differences.append(f"missing published figure {relative}")
            elif actual.suffix == ".png":
                with Image.open(expected) as original, Image.open(actual) as generated:
                    # Text metrics change the tight bounding box too, so off macOS a PNG
                    # is only required to render.
                    if not exact_pixels:
                        continue
                    if original.size != generated.size:
                        differences.append(f"rendered size differs: {relative}")
                    elif original.convert("RGBA").tobytes() != (
                        generated.convert("RGBA").tobytes()
                    ):
                        differences.append(f"rendered pixels differ: {relative}")
            elif actual.read_bytes() != expected.read_bytes():
                differences.append(f"figure differs: {relative}")
        if differences:
            raise ValueError("\n".join(differences))
        if not exact_pixels:
            print(
                f"PNG pixels not compared on {sys.platform}; the published figures are rendered on macOS"
            )
        print("all generated figures match published content")


def main(argv: list[str] | None = None) -> int:
    """Plan, validate, or reproduce selected study workflows without changing published files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "validate", "reproduce", "aggregate", "figures-check")
    )
    parser.add_argument(
        "--workflow", action="append", choices=[item.name for item in study_workflows()]
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--comparison", choices=("exact", "semantic"))
    args = parser.parse_args(argv)
    selected = tuple(
        item for item in study_workflows() if not args.workflow or item.name in args.workflow
    )
    if args.command == "aggregate" and not args.workflow:
        selected = tuple(item for item in selected if item.reproduction == "external-completions")
    try:
        if args.command == "figures-check":
            check_figures(args.root.resolve())
        elif args.command == "aggregate":
            aggregate(selected, args.root.resolve())
        elif args.command == "plan":
            print(json.dumps([workflow.model_dump(mode="json") for workflow in selected], indent=2))
        elif args.command == "validate":
            errors = [
                error for workflow in selected for error in prerequisites(workflow, args.root)
            ]
            if errors:
                raise ValueError("\n".join(errors))
            print(f"validated {len(selected)} workflow(s)")
        else:
            reproduce(selected, args.root.resolve(), args.comparison)
    except ImportError as exc:
        print(f"grpo-study: missing study dependency: {exc}. Run make install.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"grpo-study: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
