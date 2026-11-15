"""Unit tests for the GPU-independent parts of the launcher (no trl, no GPU)."""

from __future__ import annotations

import json

import pytest

from grpo_decomp.registries import TRAIN_DATASETS, TrainDataset, get_prompt_strategy
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.launcher import (
    _load_train_and_validation,
    prepare_run,
    smoke_overrides,
    to_dataset,
)

_REF = DatasetRef(name="openai/gsm8k", config="main", split="train", revision="rev")


def _problems() -> ProblemSet:
    return ProblemSet(
        source=_REF, problems=(Problem(id="p0", question="What is 2+2?", gold_answer="4"),)
    )


def test_to_dataset_exposes_prompt_and_forwarded_gold() -> None:
    dataset = to_dataset(_problems(), get_prompt_strategy("r1_zero"))
    assert dataset.column_names == ["prompt", "gold_answer"]
    assert dataset[0]["gold_answer"] == "4"
    assert "What is 2+2?" in dataset[0]["prompt"]


def test_prepare_run_creates_dir_and_writes_provenance(tmp_path) -> None:
    arm = ArmConfig(name="correct", base_model="Qwen/Qwen2.5-Math-1.5B", reward="correct", seed=3)
    run_dir = prepare_run(arm, _problems(), validation_size=256, output_root=tmp_path)

    assert run_dir.parent == tmp_path / "correct-seed3" / "attempts"
    assert len(run_dir.name) == 32
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["arm"] == "correct"
    assert provenance["reward"] == "correct"
    assert provenance["seed"] == 3
    assert provenance["train_size"] == 1
    assert provenance["validation_size"] == 256
    assert provenance["dataset"]["revision"] == "rev"
    assert provenance["grpo"]["beta"] == 0.0


def test_prepare_run_threads_explicit_commit_and_dirty(tmp_path) -> None:
    # The path Modal uses: launch() passes the locally-computed git state through.
    arm = ArmConfig(name="correct", base_model="Qwen/Qwen2.5-Math-1.5B", reward="correct", seed=3)
    run_dir = prepare_run(
        arm, _problems(), validation_size=256, output_root=tmp_path, commit="cafe", dirty=True
    )
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["commit"] == "cafe"
    assert provenance["dirty"] is True


def test_smoke_overrides_caps_steps_and_logs_each_step() -> None:
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    smoke = smoke_overrides(arm, 5)
    assert smoke.grpo.max_steps == 5
    assert smoke.grpo.logging_steps == 1  # per-step so clipped_ratio is visible in a short run
    assert smoke.grpo.save_steps == 5
    assert arm.grpo.max_steps == 500  # original is frozen and untouched


def test_smoke_overrides_is_a_noop_without_max_steps() -> None:
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    assert smoke_overrides(arm, None) is arm


def test_load_train_and_validation_uses_registered_dataset(monkeypatch) -> None:
    # The launcher dispatches on ArmConfig.dataset via the train-dataset registry; how the
    # (train, validation) split is derived is the registered dataset's concern.
    train_ref = DatasetRef(name="dummy", config="x", split="train", revision="r")
    val_ref = DatasetRef(name="dummy", config="x", split="validation", revision="r")
    train_set = ProblemSet(
        source=train_ref, problems=(Problem(id="t0", question="q", gold_answer="k"),)
    )
    val_set = ProblemSet(
        source=val_ref, problems=(Problem(id="v0", question="q", gold_answer="k"),)
    )
    seen: list[int] = []

    def _load(seed: int) -> tuple[ProblemSet, ProblemSet]:
        seen.append(seed)
        return train_set, val_set

    monkeypatch.setitem(
        TRAIN_DATASETS, "__launcher_test__", TrainDataset(name="__launcher_test__", load=_load)
    )
    arm = ArmConfig(
        name="cd", base_model="m", reward="correct", dataset="__launcher_test__", seed=7
    )
    train, validation = _load_train_and_validation(arm)
    assert seen == [7]
    assert train is train_set
    assert validation is val_set


def test_load_train_and_validation_rejects_unknown_dataset() -> None:
    arm = ArmConfig(
        name="x", base_model="m", reward="correct", dataset="nope-not-registered", seed=0
    )
    with pytest.raises(ValueError, match="unknown dataset"):
        _load_train_and_validation(arm)


@pytest.mark.parametrize("max_steps", [0, -1, True])
def test_smoke_overrides_revalidates_invalid_steps(max_steps) -> None:
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    with pytest.raises(ValueError):
        smoke_overrides(arm, max_steps)


def test_prepare_run_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "correct-seed0").symlink_to(outside, target_is_directory=True)
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    with pytest.raises(ValueError, match="escapes output root"):
        prepare_run(arm, _problems(), validation_size=1, output_root=root)
    assert list(outside.iterdir()) == []


def test_repeated_attempt_preserves_completed_checkpoint(tmp_path) -> None:
    from grpo_decomp.provenance import completed_checkpoint, load_run_provenance
    from grpo_decomp.train.provenance import finish_run

    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    first = prepare_run(arm, _problems(), validation_size=1, output_root=tmp_path)
    first_provenance = (first / "provenance.json").read_bytes()
    final = first / "checkpoints" / "final"
    final.mkdir(parents=True)
    with pytest.raises(ValueError, match="not completed"):
        completed_checkpoint(first)
    finish_run(first)
    second = prepare_run(
        smoke_overrides(arm, 3), _problems(), validation_size=1, output_root=tmp_path
    )
    assert first != second
    alias = tmp_path / "correct-seed0"
    assert completed_checkpoint(alias) == final
    assert load_run_provenance(alias).grpo.max_steps == 500
    finish_run(second, error="setup failed")
    assert completed_checkpoint(alias) == final
    assert (first / "provenance.json").read_bytes() == first_provenance


@pytest.mark.parametrize("failure", ["tokenizer", "constructor", "train", "save", None])
def test_launch_tracks_setup_and_training_failures(tmp_path, monkeypatch, failure) -> None:
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import grpo_decomp.train.launcher as launcher
    from grpo_decomp.provenance import completed_checkpoint, load_run_provenance

    cleanup = []
    monkeypatch.setattr(launcher, "_shutdown_distributed", lambda: cleanup.append(True))
    monkeypatch.setattr(
        launcher, "_load_train_and_validation", lambda arm: (_problems(), _problems())
    )

    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if failure == "tokenizer":
                raise RuntimeError("tokenizer failed")
            return SimpleNamespace(pad_token=None, eos_token="eos", padding_side="right")

    class Trainer:
        def __init__(self, **kwargs):
            if failure == "constructor":
                raise RuntimeError("constructor failed")

        def train(self):
            if failure == "train":
                raise RuntimeError("train failed")

        def save_model(self, path):
            Path(path).mkdir(parents=True)
            if failure == "save":
                raise RuntimeError("save failed")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=Tokenizer, PreTrainedTokenizerBase=type("TokenBase", (), {})),
    )
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(GRPOConfig=lambda **kw: kw, GRPOTrainer=Trainer)
    )
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            launcher.launch(arm, output_root=tmp_path, max_steps=2)
        run = next((tmp_path / "correct-seed0" / "attempts").iterdir())
        status = json.loads((run / "status.json").read_text())
        assert status["state"] == "failed"
        with pytest.raises(ValueError, match="no completed"):
            completed_checkpoint(tmp_path / "correct-seed0")
    else:
        run = launcher.launch(arm, output_root=tmp_path, max_steps=2)
        assert completed_checkpoint(tmp_path / "correct-seed0") == run / "checkpoints" / "final"
        assert load_run_provenance(run).validation_identity is not None
    assert cleanup == [True]


@pytest.mark.parametrize("kwargs", [{"smoke_problems": 0}, {"max_steps": 0}])
def test_invalid_launch_stops_before_dataset_or_output(tmp_path, monkeypatch, kwargs) -> None:
    import grpo_decomp.train.launcher as launcher

    monkeypatch.setattr(
        launcher, "_load_train_and_validation", lambda arm: pytest.fail("loaded data")
    )
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    with pytest.raises(ValueError):
        launcher.launch(arm, output_root=tmp_path, **kwargs)
    assert list(tmp_path.iterdir()) == []
