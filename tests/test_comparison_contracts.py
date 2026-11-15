"""Failure cases for the shared comparison boundary and persisted reports."""

from __future__ import annotations

import json

import pytest
from completion_set_fixtures import write_completion_set_dir
from pydantic import ValidationError

from grpo_decomp.cli import main
from grpo_decomp.eval.completions import CompletionSet, load_completion_set
from grpo_decomp.report.evidence import anchored_inputs
from grpo_decomp.report.passk_seeds import PassKMultiSeed, aggregate_passk_seeds


@pytest.fixture
def arms(tmp_path):
    for name, answer in (("base", "0"), ("trained", "4")):
        write_completion_set_dir(tmp_path / name, model=name, boxed=answer, n=2, temperature=0.7)
    return tuple(load_completion_set(tmp_path / name) for name in ("base", "trained"))


def with_provenance(cs: CompletionSet, **updates: object) -> CompletionSet:
    return cs.model_copy(update={"provenance": cs.provenance.model_copy(update=updates)})


@pytest.mark.parametrize(
    "field,value", [("temperature", 0.9), ("top_p", 0.8), ("max_new_tokens", 17)]
)
def test_decoding_drift_fails_before_statistics(arms, field, value):
    base, trained = arms
    changed = with_provenance(
        trained, sampling=trained.provenance.sampling.model_copy(update={field: value})
    )
    with pytest.raises(ValueError, match=rf"sampling.{field}"):
        aggregate_passk_seeds(base, [(0, changed)], task="mini", k=1)


@pytest.mark.parametrize("field,value", [("gold_answer", "99"), ("question", "different")])
def test_same_id_different_problem_is_rejected(arms, field, value):
    base, trained = arms
    changed = trained.items[0].model_copy(
        update={"problem": trained.items[0].problem.model_copy(update={field: value})}
    )
    trained = trained.model_copy(update={"items": (changed, *trained.items[1:])})
    with pytest.raises(ValueError, match="problem records"):
        anchored_inputs(base, [(0, trained)])


def test_copied_artifact_does_not_become_an_independent_seed(arms):
    base, trained = arms
    copy = CompletionSet.model_validate_json(trained.model_dump_json())
    with pytest.raises(ValueError, match="duplicate completion artifact"):
        anchored_inputs(base, [(0, trained), (1, copy)])


def test_retry_is_not_a_new_training_seed(arms):
    base, trained = arms
    first = with_provenance(trained, training_seed=0, training_run_id="attempt-a")
    second = with_provenance(trained, training_seed=0, training_run_id="attempt-b")
    with pytest.raises(ValueError, match="duplicate training_seed"):
        anchored_inputs(base, [(0, first), (1, second)])


def test_labels_cannot_override_recorded_identity(arms):
    base, trained = arms
    trained = with_provenance(trained, training_seed=7)
    with pytest.raises(ValueError, match="disagrees with training_seed"):
        anchored_inputs(base, [(0, trained)])


def test_different_training_protocol_is_not_seed_variance(arms):
    base, trained = arms
    first = with_provenance(trained, training_seed=0, training_protocol_digest="first")
    second = with_provenance(trained, training_seed=1, training_protocol_digest="second")
    with pytest.raises(ValueError, match="training_protocol_digest"):
        anchored_inputs(base, [(0, first), (1, second)])


def test_legacy_evidence_has_explicit_limitations(arms):
    base, trained = arms
    provenance = anchored_inputs(base, [(0, trained)]).provenance("test")
    assert len(provenance.limitations) == 3
    assert provenance.inputs[1].training_seed is None
    assert len(provenance.inputs[1].content_digest) == 64


@pytest.mark.parametrize(
    "updates",
    [
        {"delta": float("nan")},
        {"base_passk": -0.1},
        {"mean_correct_passk": 0.3},
        {"preliminary": False},
        {"n_seeds": 3},
        {"delta_ci_low": 2.0, "delta_ci_high": 1.0},
        {"base_counts": [True, 0, 0]},
    ],
)
def test_persisted_report_cannot_contradict_observations(arms, updates):
    base, trained = arms
    panel = aggregate_passk_seeds(base, [(0, trained)], task="mini", k=1)
    with pytest.raises(ValidationError):
        PassKMultiSeed.model_validate({**panel.model_dump(), **updates})


def test_duplicate_cli_directory_fails_without_stdout(tmp_path, capsys):
    path = str(tmp_path / "battery")
    assert main(["report-seeds", "--battery-dirs", path, path, path]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate input directory" in captured.err


def test_report_json_stdout_is_machine_readable(tmp_path, capsys):
    battery = tmp_path / "battery"
    write_completion_set_dir(battery / "correct__gsm8k-test", model="correct", boxed="4")
    write_completion_set_dir(battery / "random__gsm8k-test", model="random", boxed="0")
    assert main(["report-seeds", "--battery-dirs", str(battery), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_seeds"] == 1
    assert data["analysis"]["inputs"][0]["content_digest"]


def test_regression_headline_does_not_claim_elicitation(arms):
    trained, base = arms
    headline = aggregate_passk_seeds(base, [(0, trained)], task="mini", k=1).headline()
    assert "coverage decreased" in headline
    assert "elicitation" not in headline


def test_resolved_base_revision_is_authoritative(arms):
    base, trained = arms
    base = with_provenance(base, model_revision="main", resolved_model_revision="resolved-a")
    trained = with_provenance(trained, training_base_model_revision="resolved-a")
    anchored_inputs(base, [(0, trained)])
    changed = with_provenance(trained, training_base_model_revision="resolved-b")
    with pytest.raises(ValueError, match="training base revision"):
        anchored_inputs(base, [(0, changed)])


def test_control_p_values_must_match_recorded_seed_effects():
    from pathlib import Path

    from grpo_decomp.report.control_seeds import ControlDecomposition

    path = Path(__file__).resolve().parents[1] / "results/decomposition-multiseed.json"
    payload = json.loads(path.read_text())
    for row in payload["rows"]:
        row.update(p_value=0.001, p_value_holm=0.003, significant=True)
    with pytest.raises(ValueError, match="p_value"):
        ControlDecomposition.model_validate(payload)


def test_unknown_anchor_does_not_hide_conflicting_known_prompts(arms):
    base, trained = arms
    first = with_provenance(trained, prompt_digest="prompt-a", training_seed=0)
    second = with_provenance(trained, prompt_digest="prompt-b", training_seed=1)
    with pytest.raises(ValueError, match="prompt_digest"):
        anchored_inputs(base, [(0, first), (1, second)])


def test_paired_roles_require_matching_known_training_recipe(arms):
    from grpo_decomp.report.evidence import ArmInput, ComparisonInputs

    _, trained = arms
    correct = with_provenance(
        trained, training_protocol_digest="recipe-a", training_reward="correct"
    )
    random = with_provenance(trained, training_protocol_digest="recipe-b", training_reward="random")
    with pytest.raises(ValueError, match="training_protocol_digest"):
        ComparisonInputs((ArmInput("correct", "0", correct), ArmInput("random", "0", random)))
