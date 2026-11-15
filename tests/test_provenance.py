"""Unit tests for run provenance capture (no network, no GPU)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from grpo_decomp.provenance import load_run_provenance, package_versions
from grpo_decomp.schemas import DatasetRef
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.provenance import capture_provenance


def _arm() -> ArmConfig:
    return ArmConfig(name="correct", base_model="Qwen/Qwen2.5-Math-1.5B", reward="correct", seed=7)


def _dataset() -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split="train", revision="abc123")


def test_package_versions_marks_missing_as_absent() -> None:
    versions = package_versions(["grpo-decomp", "definitely-not-a-real-package-xyz"])
    assert versions["grpo-decomp"]  # an installed version string
    assert versions["definitely-not-a-real-package-xyz"] == "absent"


def test_capture_records_run_identity() -> None:
    prov = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    assert prov.arm == "correct"
    assert prov.reward == "correct"
    assert prov.seed == 7
    assert prov.train_size == 7217
    assert prov.validation_size == 256
    assert prov.dataset.revision == "abc123"
    assert prov.grpo.beta == 0.0


def test_capture_records_environment() -> None:
    prov = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    assert prov.python_version.count(".") == 2  # e.g. "3.12.7"
    assert prov.commit == "unknown" or len(prov.commit) == 40  # git SHA or sentinel
    assert prov.package_versions["grpo-decomp"]
    # TRL is not installed in the CPU dev env -> recorded as absent, not crashing.
    assert prov.package_versions["trl"] == "absent"
    assert isinstance(prov.dirty, bool)  # worktree cleanliness recorded


def test_capture_uses_explicit_commit_and_dirty_overrides() -> None:
    # On Modal the container has no .git, so the local entrypoint passes these in.
    prov = capture_provenance(
        _arm(), _dataset(), train_size=7217, validation_size=256, commit="deadbeef", dirty=True
    )
    assert prov.commit == "deadbeef"
    assert prov.dirty is True


def test_load_accepts_retired_checkpoint_fields(tmp_path) -> None:
    provenance = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    payload = provenance.model_dump()
    payload.update(
        checkpoint_selection="best_on_validation",
        selected_checkpoint="checkpoint-300",
        selected_step=300,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_run_provenance(run_dir)

    assert loaded == provenance
    assert set(payload) - set(type(loaded).model_fields) == {
        "checkpoint_selection",
        "selected_checkpoint",
        "selected_step",
    }


def test_load_still_rejects_other_unknown_fields(tmp_path) -> None:
    provenance = capture_provenance(_arm(), _dataset(), train_size=7217, validation_size=256)
    payload = {**provenance.model_dump(), "unexpected": True}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "provenance.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="unexpected"):
        load_run_provenance(run_dir)


def test_legacy_checkpoint_stays_unverified_after_failed_retry(tmp_path) -> None:
    from grpo_decomp.provenance import completed_checkpoint
    from grpo_decomp.schemas import Problem, ProblemSet
    from grpo_decomp.train.launcher import prepare_run
    from grpo_decomp.train.provenance import finish_run

    arm = _arm()
    alias = tmp_path / f"{arm.name}-seed{arm.seed}"
    alias.mkdir()
    legacy = capture_provenance(arm, _dataset(), train_size=1, validation_size=1)
    legacy = legacy.model_copy(update={"config_digest": None})
    (alias / "provenance.json").write_text(json.dumps(legacy.model_dump()))
    final = alias / "checkpoints" / "final"
    final.mkdir(parents=True)
    problems = ProblemSet(
        source=_dataset(), problems=(Problem(id="p", question="q", gold_answer="4"),)
    )
    attempt = prepare_run(arm, problems, validation_size=1, output_root=tmp_path)
    finish_run(attempt, error="setup failed")
    assert completed_checkpoint(alias) == final
    assert load_run_provenance(alias).attempt_id is None


def test_completed_attempt_survives_stale_alias_and_failed_concurrent_attempt(tmp_path) -> None:
    from grpo_decomp.provenance import completed_checkpoint
    from grpo_decomp.schemas import Problem, ProblemSet
    from grpo_decomp.train.launcher import prepare_run
    from grpo_decomp.train.provenance import finish_run

    problems = ProblemSet(
        source=_dataset(), problems=(Problem(id="p", question="q", gold_answer="4"),)
    )
    first = prepare_run(_arm(), problems, validation_size=1, output_root=tmp_path)
    second = prepare_run(_arm(), problems, validation_size=1, output_root=tmp_path)
    final = first / "checkpoints" / "final"
    final.mkdir(parents=True)
    finish_run(first)
    finish_run(second, error="setup failed")
    alias = first.parent.parent
    (alias / "run.json").write_text('{"completed_attempt": null}')
    assert completed_checkpoint(alias) == final


def test_multiple_successful_attempts_require_explicit_selection(tmp_path) -> None:
    from grpo_decomp.provenance import completed_checkpoint
    from grpo_decomp.schemas import Problem, ProblemSet
    from grpo_decomp.train.launcher import prepare_run
    from grpo_decomp.train.provenance import finish_run

    problems = ProblemSet(
        source=_dataset(), problems=(Problem(id="p", question="q", gold_answer="4"),)
    )
    attempts = [
        prepare_run(_arm(), problems, validation_size=1, output_root=tmp_path) for _ in range(2)
    ]
    for attempt in attempts:
        (attempt / "checkpoints" / "final").mkdir(parents=True)
        finish_run(attempt)
    with pytest.raises(ValueError, match=r"multiple completed attempts.*select one explicitly"):
        completed_checkpoint(attempts[0].parent.parent)
    for attempt in attempts:
        assert completed_checkpoint(attempt) == attempt / "checkpoints" / "final"
