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


def test_passk_multiseed_rejects_unknown_set(monkeypatch) -> None:
    # The decontam override validates the set against the registry before any model load,
    # so a typo fails fast instead of after a paid generation. (`@app.function` is a no-op
    # under the fake modal, so the raw function runs on CPU up to the first generate().)
    modal_app = _modal_app(monkeypatch)
    with pytest.raises(ValueError, match="unknown set"):
        modal_app.passk_multiseed(task="gsm8k", set_name="not-a-real-set")


def test_passk_multiseed_rejects_nonpositive_limit(monkeypatch) -> None:
    modal_app = _modal_app(monkeypatch)
    with pytest.raises(ValueError, match="limit must be"):
        modal_app.passk_multiseed(task="gsm8k", limit=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command": "train", "max_steps": 0},
        {"command": "train", "smoke_problems": 0},
        {"command": "battery", "scope": "typo"},
        {"command": "passk-multiseed", "n_correct": 0},
        {"command": "passk-multiseed", "correct_seeds": "0,0"},
        {"command": "passk-multiseed", "correct_seeds": "0,,1"},
    ],
)
def test_local_entrypoint_rejects_before_dispatch(monkeypatch, kwargs) -> None:
    modal_app = _modal_app(monkeypatch)
    # Fake Modal returns plain functions. A dispatch attempt would fail with AttributeError.
    with pytest.raises(ValueError):
        modal_app.main(**kwargs)


def test_all_outputs_preflight_before_first_generation(monkeypatch, tmp_path) -> None:
    from grpo_decomp.eval.completions import SamplingConfig
    from grpo_decomp.jobs import CompletionJob

    modal_app = _modal_app(monkeypatch)
    (tmp_path / "random__dev").mkdir()
    jobs = [
        CompletionJob(role, "m", None, "dev", SamplingConfig()) for role in ("correct", "random")
    ]
    monkeypatch.setattr(
        modal_app, "_generate_completion_set_to_volume", lambda **kw: pytest.fail("generated")
    )
    with pytest.raises(ValueError, match="already exists"):
        modal_app._run_completion_jobs(jobs, out_root=tmp_path, commit=None, dirty=None)


def test_plans_use_completed_attempts_and_preserve_final_semantics(monkeypatch, tmp_path) -> None:
    from grpo_decomp.jobs import plan_battery
    from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
    from grpo_decomp.train.config import ArmConfig
    from grpo_decomp.train.launcher import prepare_run
    from grpo_decomp.train.provenance import finish_run

    _modal_app(monkeypatch)
    problems = ProblemSet(
        source=DatasetRef(name="t", config=None, split="train", revision="r"),
        problems=(Problem(id="p", question="q", gold_answer="4"),),
    )
    finals = []
    for role in ("correct", "random"):
        arm = ArmConfig(name=role, base_model="m", reward=role, seed=0)
        attempt = prepare_run(arm, problems, validation_size=1, output_root=tmp_path)
        final = attempt / "checkpoints" / "final"
        final.mkdir(parents=True)
        finish_run(attempt)
        finals.append(str(final))
        # A later failed run must not redirect planning to its stale/incomplete checkpoints.
        failed = prepare_run(arm, problems, validation_size=1, output_root=tmp_path)
        finish_run(failed, error="failed")
    out_root, jobs = plan_battery(tmp_path, scope="placebo")
    assert out_root == tmp_path / "battery"
    assert [job.model_ref for job in jobs] == finals
    assert [job.arm for job in jobs] == ["correct", "random"]


def test_training_failure_commits_failed_attempt_records(monkeypatch) -> None:
    import grpo_decomp.train.launcher as launcher

    modal_app = _modal_app(monkeypatch)
    commits = []
    monkeypatch.setattr(modal_app.runs, "commit", lambda: commits.append(True))

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    monkeypatch.setattr(launcher, "launch", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        modal_app.train_arm("configs/correct.yaml")
    assert commits == [True]


def test_dispatch_sends_exact_validated_effective_config(monkeypatch) -> None:
    from types import SimpleNamespace

    from grpo_decomp.train.config import ArmConfig

    modal_app = _modal_app(monkeypatch)
    calls = []

    def spawn(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(object_id="fake-call")

    monkeypatch.setattr(modal_app, "train_arm", SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(modal_app.runs, "name", "fake-volume", raising=False)
    modal_app.main(max_steps=3)
    effective = ArmConfig.model_validate_json(calls[0]["arm_config_json"])
    assert effective.grpo.max_steps == 3
    assert effective.grpo.logging_steps == 1


def test_remote_train_accepts_validated_payload_without_local_yaml(monkeypatch) -> None:
    from pathlib import Path

    import grpo_decomp.train.launcher as launcher
    from grpo_decomp.train.config import ArmConfig

    modal_app = _modal_app(monkeypatch)
    arm = ArmConfig(name="payload", base_model="m", reward="random", seed=5)
    seen = []

    def launch(config, **kwargs):
        seen.append(config)
        return Path("/runs/payload")

    monkeypatch.setattr(launcher, "launch", launch)
    modal_app.train_arm("/local/only/missing.yaml", arm_config_json=arm.model_dump_json())
    assert seen == [arm]


def test_heldout_output_stays_with_the_scored_attempt(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    import grpo_decomp.eval.heldout as heldout

    modal_app = _modal_app(monkeypatch)
    attempt_a = tmp_path / "a"
    attempt_b = tmp_path / "b"
    attempt_a.mkdir()
    attempt_b.mkdir()
    selected = [attempt_a]
    monkeypatch.setattr(modal_app, "resolve_run_dir", lambda path: selected[0])
    monkeypatch.setattr(modal_app, "record_json", lambda curve: curve.payload)

    def score(path, **kwargs):
        assert path == attempt_a
        selected[0] = attempt_b
        return SimpleNamespace(payload="curve from a")

    monkeypatch.setattr(heldout, "run_heldout_curve", score)
    assert modal_app.heldout_arm("configs/correct.yaml") == str(attempt_a / "heldout.json")
    assert (attempt_a / "heldout.json").read_text() == "curve from a"
    assert not (attempt_b / "heldout.json").exists()
