"""Unit tests for Modal entrypoint helpers without requiring the Modal package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from grpo_decomp.train.checkpoints import (
    final_or_selected_checkpoint_path,
    require_selected_checkpoint_path,
)


class _FakeImage:
    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def add_local_file(self, *args, **kwargs):
        return self

    def add_local_dir(self, *args, **kwargs):
        return self

    def workdir(self, *args, **kwargs):
        return self

    def run_commands(self, *args, **kwargs):
        return self


class _FakeVolume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def commit(self) -> None:
        return None


class _FakeSecret:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()


class _FakeApp:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def function(self, *args, **kwargs):
        return lambda fn: fn

    def local_entrypoint(self, *args, **kwargs):
        return lambda fn: fn


class _FakeModal:
    App = _FakeApp
    Image = _FakeImage
    Secret = _FakeSecret
    Volume = _FakeVolume


def _modal_app(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", _FakeModal)
    sys.modules.pop("modal_app", None)
    path = Path(__file__).parents[1] / "modal_app.py"
    spec = importlib.util.spec_from_file_location("modal_app", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_app"] = module
    spec.loader.exec_module(module)
    return module


def test_selected_checkpoint_path_requires_heldout(monkeypatch) -> None:
    _modal_app(monkeypatch)
    run_dir = Path("/runs/correct-seed0")

    with pytest.raises(ValueError, match="heldout"):
        require_selected_checkpoint_path(run_dir, None)

    assert (
        require_selected_checkpoint_path(run_dir, "final")
        == "/runs/correct-seed0/checkpoints/final"
    )


def test_final_or_selected_checkpoint_realizes_the_final_rule(monkeypatch) -> None:
    _modal_app(monkeypatch)
    run_dir = Path("/runs/correct-seed3")

    # Recorded selection wins when present.
    assert (
        final_or_selected_checkpoint_path(run_dir, "checkpoint-400", "best_on_validation")
        == "/runs/correct-seed3/checkpoints/checkpoint-400"
    )
    # Unset selection + 'final' rule resolves to final without a held-out curve (the
    # replicate-seed case: rule 'final', selection None — what the published artifacts used).
    assert (
        final_or_selected_checkpoint_path(run_dir, None, "final")
        == "/runs/correct-seed3/checkpoints/final"
    )
    # Unset selection under any other rule is an explicit error.
    with pytest.raises(ValueError, match="heldout"):
        final_or_selected_checkpoint_path(run_dir, None, "best_on_validation")


def test_elicitation_multiseed_rejects_unknown_set(monkeypatch) -> None:
    # The decontam override validates the set against the registry before any model load,
    # so a typo fails fast instead of after a paid generation. (`@app.function` is a no-op
    # under the fake modal, so the raw function runs on CPU up to the first generate().)
    modal_app = _modal_app(monkeypatch)
    with pytest.raises(ValueError, match="unknown set"):
        modal_app.elicitation_multiseed(task="gsm8k", set_name="not-a-real-set")


def test_elicitation_multiseed_rejects_nonpositive_limit(monkeypatch) -> None:
    modal_app = _modal_app(monkeypatch)
    with pytest.raises(ValueError, match="limit must be"):
        modal_app.elicitation_multiseed(task="gsm8k", limit=0)
