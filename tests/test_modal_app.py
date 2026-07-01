"""Unit tests for Modal entrypoint helpers without requiring the Modal package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
