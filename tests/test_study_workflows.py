"""The study manifest must explain every published JSON and fail before partial execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_grpo_gains.workflows import (
    StudyWorkflow,
    compare_outputs,
    main,
    prerequisites,
    reproduce,
    study_workflows,
)

_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_inventories_every_published_result_and_its_recorded_seeds() -> None:
    workflows = study_workflows()
    published = {
        path.relative_to(_ROOT)
        for path in (_ROOT / "results").rglob("*.json")
        if path.name != "provenance.json"
    }
    assert {item.output for item in workflows} == published
    assert len({item.name for item in workflows}) == len(workflows)
    for workflow in workflows:
        if workflow.reproduction == "published-diagnostic":
            continue
        result = json.loads((_ROOT / workflow.output).read_text())
        if "n_seeds" in result:
            assert result["n_seeds"] == len(workflow.seeds)
        if "seeds" in result:
            assert tuple(result["seeds"]) == workflow.seeds
    for name in ("gsm8k-placebo", "countdown-placebo"):
        workflow = next(item for item in workflows if item.name == name)
        assert workflow.arguments[0] == "report-seeds"


def test_manifest_lists_every_missing_input_before_running_commands(tmp_path, monkeypatch) -> None:
    workflow = StudyWorkflow(
        name="missing-inputs",
        task_set="fixture",
        seeds=("0",),
        output=Path("published.json"),
        reproduction="external-completions",
        module="unused",
        completion_sets=(Path("first"), Path("second")),
    )
    (tmp_path / "published.json").write_text("{}")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("must validate all inputs before executing a report")

    monkeypatch.setattr("llm_grpo_gains.workflows.subprocess.run", unexpected_run)
    errors = prerequisites(workflow, tmp_path)
    assert len(errors) == 4
    with pytest.raises(ValueError) as caught:
        reproduce((workflow,), tmp_path)
    for path in (
        "first/provenance.json",
        "first/completions.jsonl",
        "second/provenance.json",
        "second/completions.jsonl",
    ):
        assert path in str(caught.value)


def test_semantic_comparison_allows_analysis_provenance_but_rejects_metric_changes(
    tmp_path,
) -> None:
    original, generated = tmp_path / "old.json", tmp_path / "new.json"
    original.write_text('{"accuracy": 0.5}')
    generated.write_text('{"accuracy":0.5,"analysis":{"commit":"new"},"optional":null}')
    assert compare_outputs(original, generated, "semantic")
    assert not compare_outputs(original, generated, "exact")
    generated.write_text('{"accuracy":0.6,"analysis":{"commit":"new"}}')
    assert not compare_outputs(original, generated, "semantic")


def test_semantic_comparison_allows_additive_fields_but_not_missing_or_changed_ones(
    tmp_path,
) -> None:
    original, generated = tmp_path / "old.json", tmp_path / "new.json"
    original.write_text('{"k": 8, "rows": [{"gain": 0.1}]}')
    generated.write_text(
        '{"k": 8, "frac_lost_reliability": 0.2, "rows": [{"gain": 0.1, "ci": [0, 1]}]}'
    )
    assert compare_outputs(original, generated, "semantic")
    generated.write_text('{"frac_lost_reliability": 0.2, "rows": [{"gain": 0.1}]}')
    assert not compare_outputs(original, generated, "semantic")
    generated.write_text('{"k": 8, "rows": [{"gain": 0.1}, {"gain": 0.2}]}')
    assert not compare_outputs(original, generated, "semantic")
    generated.write_text('{"k": 8, "rows": {"gain": 0.1}}')
    assert not compare_outputs(original, generated, "semantic")


def test_controls_workflow_forwards_its_control_order_to_the_report() -> None:
    workflow = next(item for item in study_workflows() if item.name == "gsm8k-controls")
    published = json.loads((_ROOT / workflow.output).read_text())
    start = workflow.arguments.index("--control-sets") + 1
    forwarded = workflow.arguments[start : start + len(published["rows"])]
    assert forwarded == tuple(row["control"] for row in published["rows"])


def test_plan_is_machine_readable_and_validation_reports_missing_prerequisites(
    tmp_path, capsys
) -> None:
    assert main(["plan", "--workflow", "gsm8k-placebo"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan[0]["name"] == "gsm8k-placebo"
    assert plan[0]["reproduction"] == "external-completions"
    assert main(["validate", "--workflow", "gsm8k-placebo", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "correct__gsm8k-test/provenance.json" in captured.err
    assert "battery-seed5/random__gsm8k-test/completions.jsonl" in captured.err
    assert not captured.out


def test_committed_esme_workflow_has_all_prerequisites() -> None:
    esme = next(item for item in study_workflows() if item.name == "esme")
    assert prerequisites(esme, _ROOT) == ()
    assert "--legacy" in esme.command(Path("/tmp/unused.json"))


def test_missing_figure_dependency_has_actionable_error(monkeypatch, capsys) -> None:
    from llm_grpo_gains import workflows

    def missing(root):
        raise ImportError("No module named matplotlib")

    monkeypatch.setattr(workflows, "check_figures", missing)
    assert workflows.main(["figures-check"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "matplotlib" in captured.err and "make install" in captured.err
    assert "Traceback" not in captured.err
