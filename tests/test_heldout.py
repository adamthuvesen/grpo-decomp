"""Held-out reconstruction must match the split recorded by the training attempt."""

from __future__ import annotations

import pytest

from grpo_decomp.eval.heldout import validation_for_run
from grpo_decomp.provenance import problem_set_digest
from grpo_decomp.registries import VALIDATION_RECONSTRUCTORS
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.provenance import capture_provenance


def test_reconstruction_rejects_same_size_different_validation(monkeypatch) -> None:
    source = DatasetRef(name="heldout-test", config=None, split="validation", revision="r")
    expected = ProblemSet(
        source=source, problems=(Problem(id="v1", question="q", gold_answer="4"),)
    )
    different = ProblemSet(
        source=source, problems=(Problem(id="v2", question="q", gold_answer="4"),)
    )
    arm = ArmConfig(name="arm", base_model="m", reward="random", seed=0)
    provenance = capture_provenance(
        arm,
        source,
        train_size=1,
        validation_size=1,
        validation_identity=problem_set_digest(expected),
    )
    monkeypatch.setitem(VALIDATION_RECONSTRUCTORS, source.name, lambda p: different)
    with pytest.raises(ValueError, match="validation identity"):
        validation_for_run(provenance)
    monkeypatch.setitem(VALIDATION_RECONSTRUCTORS, source.name, lambda p: expected)
    assert validation_for_run(provenance) == expected


def test_reconstruction_rejects_wrong_legacy_size(monkeypatch) -> None:
    source = DatasetRef(name="heldout-size", config=None, split="validation", revision="r")
    validation = ProblemSet(
        source=source, problems=(Problem(id="v", question="q", gold_answer="4"),)
    )
    arm = ArmConfig(name="arm", base_model="m", reward="random", seed=0)
    provenance = capture_provenance(arm, source, train_size=1, validation_size=2)
    monkeypatch.setitem(VALIDATION_RECONSTRUCTORS, source.name, lambda p: validation)
    with pytest.raises(ValueError, match="validation size"):
        validation_for_run(provenance)
