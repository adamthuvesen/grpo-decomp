"""Typed, offline decomposition of the Esme Countdown-Lite completion artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictFloat, StrictInt, model_validator

from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.grading import extract_lenient
from grpo_decomp.report.evidence import (
    AnalysisProvenance,
    ArmInput,
    ComparisonInputs,
    seed_labels,
    unique_input_paths,
)
from grpo_decomp.report.status import MIN_HEADLINE_SEEDS
from grpo_decomp.report.validation import Effect, Rate, interval, mean, require_close
from grpo_decomp.schemas import Record
from grpo_decomp.stats.bootstrap import seed_problem_bootstrap_ci
from grpo_decomp.stats.significance import holm_correction, sign_flip_p_value
from llm_grpo_gains.data.countdown import parse_countdown_key
from llm_grpo_gains.esme_countdown import (
    esme_countdown_is_correct,
    is_wellformed_esme_countdown,
)
from llm_grpo_gains.registration import register

EffectMetric = Literal["valid_rate", "exact_rate", "any_exact"]
ArmRole = Literal["base", "correct", "random"]
EFFECT_METRICS: tuple[EffectMetric, ...] = ("valid_rate", "exact_rate", "any_exact")
SUMMARY_NAME = "sampled_multiseed_summary.corrected.json"
LEGACY_SUMMARY_NAME = "sampled_multiseed_summary.json"


class ProblemMetrics(Record):
    valid_rate: Rate
    sum_template_rate: Rate
    exact_rate: Rate
    any_exact: Rate


class ArmRates(Record):
    model: str | None = None
    valid_rate: Rate
    sum_template_rate: Rate
    exact_rate: Rate
    any_exact_rate: Rate
    solved_sum_target: StrictInt | StrictFloat = Field(ge=0)
    solved_other: StrictInt | StrictFloat = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ArmRates:
        if self.sum_template_rate > self.valid_rate or self.exact_rate > self.valid_rate:
            raise ValueError("sum-template and exact rates cannot exceed the valid-expression rate")
        if self.exact_rate > self.any_exact_rate:
            raise ValueError("per-sample exact rate cannot exceed the task-solved rate")
        return self


class SeedRates(Record):
    seed: str
    correct: ArmRates
    random: ArmRates


class SampledEffect(Record):
    mean_delta: Effect
    ci_low: float
    ci_high: float
    ci_kind: str
    sign_flip_p: Rate
    sign_flip_p_holm: Rate | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SampledEffect:
        interval("sampled effect", self.ci_low, self.ci_high)
        if self.sign_flip_p_holm is not None and self.sign_flip_p_holm < self.sign_flip_p:
            raise ValueError("Holm-adjusted p cannot be below the raw p-value")
        return self


class LegacySampledSummary(Record):
    task_set: str
    dataset_revision: str
    n_seeds: StrictInt = Field(gt=0)
    seeds: tuple[str, ...]
    preliminary: bool
    n_problems: StrictInt = Field(gt=0)
    n_sum_target_problems: StrictInt = Field(ge=0)
    n_samples: StrictInt = Field(gt=0)
    temperature: float = Field(ge=0)
    max_new_tokens: StrictInt = Field(gt=0)
    arms: dict[ArmRole, ArmRates]
    per_seed: tuple[SeedRates, ...]
    effects: dict[EffectMetric, SampledEffect]

    @model_validator(mode="after")
    def _consistent(self) -> LegacySampledSummary:
        labels = seed_labels(self.seeds)
        if len(labels) != self.n_seeds or tuple(row.seed for row in self.per_seed) != labels:
            raise ValueError("seed count, labels, and per-seed rows must agree")
        if self.preliminary != (self.n_seeds < MIN_HEADLINE_SEEDS):
            raise ValueError("preliminary flag disagrees with the number of seeds")
        if set(self.arms) != {"base", "correct", "random"}:
            raise ValueError("summary requires base, correct, and random arms")
        if set(self.effects) != set(EFFECT_METRICS):
            raise ValueError("summary requires the three sampled effect metrics")
        if self.n_sum_target_problems > self.n_problems:
            raise ValueError("sum-target count exceeds the problem count")
        rows = [
            *self.arms.values(),
            *(row.correct for row in self.per_seed),
            *(row.random for row in self.per_seed),
        ]
        for row in rows:
            if (
                row.solved_sum_target > self.n_sum_target_problems
                or row.solved_other > self.n_problems - self.n_sum_target_problems
            ):
                raise ValueError("solved-task count exceeds its problem subset")
            require_close(
                "any_exact_rate",
                row.any_exact_rate,
                (row.solved_sum_target + row.solved_other) / self.n_problems,
            )
        roles: tuple[ArmRole, ...] = ("correct", "random")
        for role in roles:
            rates = [getattr(row, role) for row in self.per_seed]
            for field in (
                "valid_rate",
                "sum_template_rate",
                "exact_rate",
                "any_exact_rate",
                "solved_sum_target",
                "solved_other",
            ):
                require_close(
                    field,
                    getattr(self.arms[role], field),
                    mean([getattr(row, field) for row in rates]),
                )
        for metric in EFFECT_METRICS:
            field = "any_exact_rate" if metric == "any_exact" else metric
            effects = [
                getattr(row.correct, field) - getattr(row.random, field) for row in self.per_seed
            ]
            require_close(
                "sign_flip_p", self.effects[metric].sign_flip_p, sign_flip_p_value(effects)
            )
            require_close(
                "mean_delta",
                self.effects[metric].mean_delta,
                getattr(self.arms["correct"], field) - getattr(self.arms["random"], field),
            )
        return self


class InferenceFamily(Record):
    scope: Literal["retrospective exploratory"] = "retrospective exploratory"
    hypotheses: tuple[EffectMetric, ...] = EFFECT_METRICS
    multiplicity_method: Literal["Holm"] = "Holm"
    confidence_interval_scope: Literal["marginal, not simultaneous"] = "marginal, not simultaneous"
    primary_endpoint: None = None

    @model_validator(mode="after")
    def _consistent(self) -> InferenceFamily:
        if self.hypotheses != EFFECT_METRICS:
            raise ValueError("the exploratory family must contain the three reported outcomes")
        return self


class SampledDecomposition(LegacySampledSummary):
    schema_version: Literal[1] = 1
    inference: InferenceFamily = Field(default_factory=InferenceFamily)
    analysis: AnalysisProvenance

    @model_validator(mode="after")
    def _adjustment_consistent(self) -> SampledDecomposition:
        adjusted = holm_correction([self.effects[metric].sign_flip_p for metric in EFFECT_METRICS])
        for metric, expected in zip(EFFECT_METRICS, adjusted, strict=True):
            actual = self.effects[metric].sign_flip_p_holm
            if actual is None:
                raise ValueError("corrected summaries require Holm-adjusted p-values")
            require_close("sign_flip_p_holm", actual, expected)
        return self


def _problem_metrics(cset: CompletionSet) -> dict[str, ProblemMetrics]:
    metrics = {}
    for item in cset.items:
        numbers, _ = parse_countdown_key(item.problem.gold_answer)
        valid = sum_template = exact = 0
        for sample in item.samples:
            expression = extract_lenient(sample)
            if expression is None or not is_wellformed_esme_countdown(expression, numbers):
                continue
            valid += 1
            sum_template += not any(op in expression for op in "-*")
            exact += esme_countdown_is_correct(expression, item.problem.gold_answer)
        n = len(item.samples)
        metrics[item.problem.id] = ProblemMetrics(
            valid_rate=valid / n,
            sum_template_rate=sum_template / n,
            exact_rate=exact / n,
            any_exact=float(exact > 0),
        )
    return metrics


def _arm_rates(
    cset: CompletionSet, metrics: dict[str, ProblemMetrics], sum_target_ids: set[str]
) -> ArmRates:
    solved = {pid for pid, row in metrics.items() if row.any_exact}
    return ArmRates(
        model=cset.provenance.model,
        valid_rate=mean([row.valid_rate for row in metrics.values()]),
        sum_template_rate=mean([row.sum_template_rate for row in metrics.values()]),
        exact_rate=mean([row.exact_rate for row in metrics.values()]),
        any_exact_rate=mean([row.any_exact for row in metrics.values()]),
        solved_sum_target=len(solved & sum_target_ids),
        solved_other=len(solved - sum_target_ids),
    )


def _mean_rates(rows: Sequence[ArmRates]) -> ArmRates:
    return ArmRates(
        valid_rate=mean([row.valid_rate for row in rows]),
        sum_template_rate=mean([row.sum_template_rate for row in rows]),
        exact_rate=mean([row.exact_rate for row in rows]),
        any_exact_rate=mean([row.any_exact_rate for row in rows]),
        solved_sum_target=mean([row.solved_sum_target for row in rows]),
        solved_other=mean([row.solved_other for row in rows]),
    )


def analyze(
    *, seeds: Sequence[tuple[str, Path]], task_set: str, base_completions_dir: Path
) -> SampledDecomposition:
    """Validate every input before grading and compute the three exploratory effects."""
    labels = seed_labels([label for label, _ in seeds])
    unique_input_paths([directory for _, directory in seeds])
    register()
    base = load_completion_set(base_completions_dir)
    arms = [ArmInput("base", None, base)]
    for label, (_, directory) in zip(labels, seeds, strict=True):
        for role in ("correct", "random"):
            arms.append(
                ArmInput(role, label, load_completion_set(directory / f"{role}__{task_set}"))
            )
    inputs = ComparisonInputs(tuple(arms))
    # This decomposition reports the same sampled opportunity for every arm.
    if any(arm.completions.provenance.sampling.n != base.provenance.sampling.n for arm in arms):
        raise ValueError("Esme sampled decomposition requires the same n across all arms")
    problem_ids = sorted(item.problem.id for item in base.items)
    sum_target_ids = set()
    for item in base.items:
        numbers, target = parse_countdown_key(item.problem.gold_answer)
        if target == sum(numbers):
            sum_target_ids.add(item.problem.id)
    per_seed = []
    deltas: dict[EffectMetric, list[list[float]]] = {metric: [] for metric in EFFECT_METRICS}
    for index, label in enumerate(labels):
        correct_cset, random_cset = arms[1 + index * 2].completions, arms[2 + index * 2].completions
        correct, random = _problem_metrics(correct_cset), _problem_metrics(random_cset)
        for metric in EFFECT_METRICS:
            deltas[metric].append(
                [
                    getattr(correct[pid], metric) - getattr(random[pid], metric)
                    for pid in problem_ids
                ]
            )
        per_seed.append(
            SeedRates(
                seed=label,
                correct=_arm_rates(correct_cset, correct, sum_target_ids),
                random=_arm_rates(random_cset, random, sum_target_ids),
            )
        )
    effects: dict[EffectMetric, SampledEffect] = {}
    raw_p = [sign_flip_p_value([mean(row) for row in deltas[metric]]) for metric in EFFECT_METRICS]
    adjusted_p = holm_correction(raw_p)
    for metric, raw, adjusted in zip(EFFECT_METRICS, raw_p, adjusted_p, strict=True):
        rows = deltas[metric]
        effect_mean, low, high = seed_problem_bootstrap_ci(rows)
        effects[metric] = SampledEffect(
            mean_delta=effect_mean,
            ci_low=low,
            ci_high=high,
            ci_kind=f"seed x problem bootstrap ({len(rows)} seeds x {len(rows[0])} problems)",
            sign_flip_p=raw,
            sign_flip_p_holm=adjusted,
        )
    provenance = base.provenance
    return SampledDecomposition(
        task_set=task_set,
        dataset_revision=provenance.dataset.revision,
        n_seeds=len(labels),
        seeds=labels,
        preliminary=len(labels) < MIN_HEADLINE_SEEDS,
        n_problems=provenance.n_problems,
        n_sum_target_problems=len(sum_target_ids),
        n_samples=provenance.sampling.n,
        temperature=provenance.sampling.temperature,
        max_new_tokens=provenance.sampling.max_new_tokens,
        arms={
            "base": _arm_rates(base, _problem_metrics(base), sum_target_ids),
            "correct": _mean_rates([row.correct for row in per_seed]),
            "random": _mean_rates([row.random for row in per_seed]),
        },
        per_seed=tuple(per_seed),
        effects=effects,
        analysis=inputs.provenance("esme-sampled-decomposition-v1"),
    )


def legacy_json(result: SampledDecomposition) -> str:
    """Serialize the historical schema in its original field order and whitespace."""
    payload = result.model_dump(
        exclude_none=True,
        exclude={
            "schema_version": True,
            "inference": True,
            "analysis": True,
            "effects": {"__all__": {"sign_flip_p_holm"}},
        },
    )
    return json.dumps(payload, indent=2) + "\n"


def render_tables(result: LegacySampledSummary) -> str:
    """Render the report tables from the same typed results as the JSON artifact."""
    labels = {
        "valid_rate": "Valid expressions",
        "exact_rate": "Exact solves, per sample",
        "any_exact": "Tasks solved at least once",
    }
    lines = [
        "| Measure | Real reward | Placebo | Real - placebo, marginal 95% CI | Raw p | Holm p |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for metric in EFFECT_METRICS:
        effect = result.effects[metric]
        field = "any_exact_rate" if metric == "any_exact" else metric
        real, placebo = (
            getattr(result.arms["correct"], field),
            getattr(result.arms["random"], field),
        )
        adjusted = (
            "unavailable" if effect.sign_flip_p_holm is None else f"{effect.sign_flip_p_holm:.5f}"
        )
        lines.append(
            f"| {labels[metric]} | {real * 100:.1f}% | {placebo * 100:.2f}% | "
            f"{effect.mean_delta * 100:+.1f} pp [{effect.ci_low * 100:+.1f}, "
            f"{effect.ci_high * 100:+.1f}] | {effect.sign_flip_p:.5f} | {adjusted} |"
        )
    lines.extend(
        [
            "",
            "| Arm | Valid | Sum of the numbers | Sum-target tasks solved | Other tasks solved |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    arm_labels: tuple[tuple[ArmRole, str], ...] = (
        ("base", "base"),
        ("random", "placebo"),
        ("correct", "real reward"),
    )
    for role, label in arm_labels:
        row = result.arms[role]
        lines.append(
            f"| {label} | {row.valid_rate * 100:.1f}% | {row.sum_template_rate * 100:.1f}% | "
            f"{row.solved_sum_target:.1f} | {row.solved_other:.1f} |"
        )
    lines.extend(
        [
            "",
            "| Seed | real valid | real sum of the numbers | real solved | "
            "placebo valid | placebo solved |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed_row in result.per_seed:
        lines.append(
            f"| {seed_row.seed} | {seed_row.correct.valid_rate * 100:.1f}% | "
            f"{seed_row.correct.sum_template_rate * 100:.1f}% | "
            f"{seed_row.correct.solved_sum_target:g} + {seed_row.correct.solved_other:g} | "
            f"{seed_row.random.valid_rate * 100:.1f}% | "
            f"{seed_row.random.solved_sum_target:g} + {seed_row.random.solved_other:g} |"
        )
    return "\n".join(lines) + "\n"


def render_report(result: SampledDecomposition) -> str:
    real = result.arms["correct"]
    return (
        "# Esme-214M findings: the reward teaches the sum of the numbers\n\n"
        f"{result.n_seeds} real-reward seeds and {result.n_seeds} random-reward placebo seeds. "
        f"{result.n_problems} held-out Countdown-Lite tasks. "
        f"{result.n_samples} samples per task at temperature {result.temperature:g}, "
        f"with {result.max_new_tokens} new tokens.\n\n"
        f"The real-reward model writes the sum template in {real.sum_template_rate * 100:.1f}% "
        "of samples. The descriptive pattern supports a sum-template interpretation. "
        "The model solves almost only tasks whose target equals the sum of the numbers.\n\n"
        "The three outcomes form a retrospective exploratory family. "
        "No primary endpoint was specified. Raw sign-flip p-values remain visible. "
        "Holm correction covers all three outcomes. The 95% confidence intervals are marginal, "
        "not simultaneous. These exploratory tests do not establish a confirmatory result.\n\n"
        "## Results\n\n"
        + render_tables(result)
        + "\n![Esme sum-template decomposition](fig-sum-template.svg)\n\n"
        "## Metrics and method\n\n"
        "A valid expression uses every supplied number exactly once with binary `+ - *`. "
        "A sum template is a valid expression that uses only `+`. "
        "An exact solve reaches the target. "
        "A task is solved when any sampled expression reaches it.\n\n"
        f"{result.n_sum_target_problems} of the {result.n_problems} held-out tasks have "
        "a target equal to the sum of their numbers. The sum template solves that subset. "
        "Other solves test whether the behavior extends beyond the template.\n\n"
        "The real-reward and placebo arms use the same recipe and budget. "
        "The committed runs use one fixed checkpoint at training step 240. "
        "No checkpoint is selected by training reward. Each comparison pairs seeds and problems. "
        "The bootstrap resamples seeds and problems together. The exact sign-flip test uses "
        "the seed-level effects. The reported intervals do not adjust for multiplicity.\n\n"
        "The training reward gives partial credit for well-formed and valid expressions, "
        "plus closeness to the target. The sum template earns that credit on many tasks. "
        "The [esme-posttrain research report]"
        "(https://github.com/adamthuvesen/esme-posttrain/blob/main/docs/rlvr-research-report.md) "
        "reports reward ablations. This analysis does not recompute those ablations.\n\n"
        "## Limitations\n\n"
        "The findings cover one task and one model size. The held-out tasks use the training "
        "generator's distribution. In the committed set, 199 of 200 tasks have three numbers. "
        "Six seeds limit the smallest two-sided sign-flip p-value to 0.03125. "
        "Output form identifies the template. It does not reveal internal representations. "
        "The retrospective family limits confirmatory claims.\n\n"
        + "\n".join(f"- {limitation}" for limitation in result.analysis.limitations)
        + "\n\nReproduce the historical JSON with `make reproduce-esme`. "
        "The explicit `--legacy` mode preserves `sampled_multiseed_summary.json`. "
        "Default analysis writes `sampled_multiseed_summary.corrected.json` and this report.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)

    def parse_seed(value: str) -> tuple[str, Path]:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise argparse.ArgumentTypeError("--seed must be LABEL=DIR")
        return label, Path(raw_path)

    parser.add_argument(
        "--seed", action="append", required=True, type=parse_seed, metavar="LABEL=DIR"
    )
    parser.add_argument("--base-completions-dir", type=Path, required=True)
    parser.add_argument("--task-set", default="esme-countdown")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--legacy", action="store_true", help="Write the historical unversioned JSON."
    )
    args = parser.parse_args(argv)
    try:
        result = analyze(
            seeds=args.seed, task_set=args.task_set, base_completions_dir=args.base_completions_dir
        )
        table = render_tables(result)
        name = LEGACY_SUMMARY_NAME if args.legacy else SUMMARY_NAME
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            payload = (
                legacy_json(result)
                if args.legacy
                else json.dumps(result.model_dump(), indent=2) + "\n"
            )
            (args.out / name).write_text(payload, encoding="utf-8")
            if not args.legacy:
                (args.out / "sampled_decomposition.md").write_text(
                    render_report(result), encoding="utf-8"
                )
    except (OSError, ValueError) as exc:
        print(f"esme sampled decomposition: {exc}", file=sys.stderr)
        return 2
    print(f"Esme-214M-RL sampled multiseed decomposition ({result.n_seeds} seeds)")
    print(table)
    if args.out:
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
