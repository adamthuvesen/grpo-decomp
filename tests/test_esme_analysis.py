"""Esme input comparability and historical reproduction contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    write_completion_set,
)
from grpo_decomp.schemas import Problem
from llm_grpo_gains import esme_analysis
from llm_grpo_gains.esme_countdown import load_esme_countdown


def _artifact(model: str) -> CompletionSet:
    problems = (
        Problem(id="sum", question="Reach 10 with 1 and 9", gold_answer="target=10;numbers=1,9"),
        Problem(id="other", question="Reach 9 with 1 and 9", gold_answer="target=9;numbers=1,9"),
    )
    return CompletionSet(
        provenance=GenerationProvenance(
            model=model,
            model_revision=None,
            backend="historical",
            prompt_strategy="esme_countdown_chat",
            sampling=SamplingConfig(temperature=1.0, n=2),
            dataset=load_esme_countdown().source,
            n_problems=2,
            commit="commit",
            python_version="3.13",
            package_versions={},
        ),
        items=tuple(
            ProblemCompletions(problem=problem, samples=("1 + 9", "1 + 9")) for problem in problems
        ),
    )


def _inputs(
    root: Path,
    change: Callable[[CompletionSet], CompletionSet] | None = None,
) -> tuple[list[tuple[str, Path]], Path]:
    base = write_completion_set(_artifact("base"), root / "base")
    seeds = []
    for seed in ("1", "2", "3"):
        directory = root / f"seed{seed}"
        seeds.append((seed, directory))
        for role in ("correct", "random"):
            artifact = _artifact(f"{role}-{seed}")
            if change is not None and seed == "1" and role == "random":
                artifact = change(artifact)
            write_completion_set(artifact, directory / f"{role}__esme-countdown")
    return seeds, base


@pytest.mark.parametrize("field", ["problem", "prompt", "top_p", "n"])
def test_alignment_errors_precede_all_grading(tmp_path, monkeypatch, field) -> None:
    def change(artifact: CompletionSet) -> CompletionSet:
        if field == "problem":
            item = artifact.items[0]
            changed = item.model_copy(
                update={"problem": item.problem.model_copy(update={"question": "changed"})}
            )
            return artifact.model_copy(update={"items": (changed, artifact.items[1])})
        provenance = artifact.provenance
        if field == "prompt":
            provenance = provenance.model_copy(update={"prompt_strategy": "different"})
        else:
            sampling = provenance.sampling.model_copy(
                update={field: 0.5 if field == "top_p" else 1}
            )
            provenance = provenance.model_copy(update={"sampling": sampling})
        items = artifact.items
        if field == "n":
            items = tuple(item.model_copy(update={"samples": item.samples[:1]}) for item in items)
        return artifact.model_copy(update={"provenance": provenance, "items": items})

    seeds, base = _inputs(tmp_path, change)

    def unexpected_grading(*args, **kwargs):
        pytest.fail("comparison inputs must be validated before grading")

    monkeypatch.setattr(esme_analysis, "_problem_metrics", unexpected_grading)
    with pytest.raises(
        ValueError, match=r"problem records|prompt strategy|sampling\.top_p|share n"
    ):
        esme_analysis.analyze(seeds=seeds, task_set="esme-countdown", base_completions_dir=base)


def test_duplicate_seed_labels_fail_before_artifact_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate seed labels"):
        esme_analysis.analyze(
            seeds=[("1", tmp_path / "one"), ("01", tmp_path / "two")],
            task_set="esme-countdown",
            base_completions_dir=tmp_path / "missing",
        )


def test_duplicate_directories_fail_before_artifact_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicate input directory"):
        esme_analysis.analyze(
            seeds=[("1", tmp_path), ("2", tmp_path / ".")],
            task_set="esme-countdown",
            base_completions_dir=tmp_path / "missing",
        )


def test_generation_seeds_may_differ_but_training_labels_remain_unverified(tmp_path) -> None:
    def change(artifact: CompletionSet) -> CompletionSet:
        sampling = artifact.provenance.sampling.model_copy(update={"seed": 999})
        return artifact.model_copy(
            update={"provenance": artifact.provenance.model_copy(update={"sampling": sampling})}
        )

    seeds, base = _inputs(tmp_path, change)
    result = esme_analysis.analyze(
        seeds=seeds, task_set="esme-countdown", base_completions_dir=base
    )
    assert len(result.analysis.inputs) == 7
    assert any("not verified" in text for text in result.analysis.limitations)
    assert all(effect.sign_flip_p_holm == 1 for effect in result.effects.values())


def test_committed_legacy_summary_is_byte_identical() -> None:
    root = Path(__file__).resolve().parents[1] / "results/esme-countdown"
    completions = root / "completions"
    result = esme_analysis.analyze(
        seeds=[(str(seed), completions / f"seed{seed}") for seed in range(214, 220)],
        task_set="esme-countdown",
        base_completions_dir=completions / "base__esme-countdown",
    )
    assert (
        esme_analysis.legacy_json(result).encode()
        == (root / "sampled_multiseed_summary.json").read_bytes()
    )
    assert all(effect.sign_flip_p == 0.03125 for effect in result.effects.values())
    assert all(effect.sign_flip_p_holm == 0.09375 for effect in result.effects.values())
    assert esme_analysis.render_tables(result) in esme_analysis.render_report(result)


@pytest.mark.parametrize("field", ["adjustment", "family", "arm_mean", "fabricated_p"])
def test_corrected_summary_rejects_inconsistent_persisted_evidence(tmp_path, field) -> None:
    seeds, base = _inputs(tmp_path)
    result = esme_analysis.analyze(
        seeds=seeds, task_set="esme-countdown", base_completions_dir=base
    )
    payload = result.model_dump()
    if field == "adjustment":
        payload["effects"]["valid_rate"]["sign_flip_p"] = 0.1
    elif field == "fabricated_p":
        for effect in payload["effects"].values():
            effect["sign_flip_p"] = 0.001
            effect["sign_flip_p_holm"] = 0.003
    elif field == "family":
        payload["inference"]["hypotheses"] = ("valid_rate",)
    else:
        payload["arms"]["correct"]["valid_rate"] = 0.5
    with pytest.raises(ValueError):
        esme_analysis.SampledDecomposition.model_validate(payload)


def test_plugin_failure_is_a_clean_cli_error(monkeypatch, capsys, tmp_path) -> None:
    from grpo_decomp.plugins import PluginLoadError

    def broken():
        raise PluginLoadError("plugin 'broken' failed")

    monkeypatch.setattr(esme_analysis, "load_plugins", broken)
    seeds, base = _inputs(tmp_path)
    args = ["--base-completions-dir", str(base), "--out", str(tmp_path / "out")]
    for seed, path in seeds:
        args.extend(["--seed", f"{seed}={path}"])
    assert esme_analysis.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plugin 'broken' failed" in captured.err
