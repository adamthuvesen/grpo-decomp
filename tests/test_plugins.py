"""The real entry-point discovery path (not the direct register() conftest uses).

conftest registers the study directly, so this is the one place the packaging entry point
and the loader are exercised together — a broken `[project.entry-points."grpo_decomp.plugins"]`
or a loader regression would fail here even though everything else passes.
"""

from __future__ import annotations

from grpo_decomp.plugins import ENTRY_POINT_GROUP, load_plugins
from grpo_decomp.registries import EVAL_SETS, REWARDS, TASKS


def test_load_plugins_discovers_the_study_via_entry_point() -> None:
    names = load_plugins(force=True)
    assert "gsm8k" in names  # the study's grpo_decomp.plugins entry-point name (see pyproject)


def test_load_plugins_actually_populates_the_registries() -> None:
    load_plugins(force=True)
    assert "gsm8k-test" in EVAL_SETS
    assert {"correct", "countdown"} <= set(REWARDS)
    assert {"gsm8k", "countdown"} <= set(TASKS)


def test_entry_point_group_name_is_stable() -> None:
    assert ENTRY_POINT_GROUP == "grpo_decomp.plugins"


def test_failed_plugin_startup_restores_all_registry_changes(monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from grpo_decomp import plugins, registries

    eval_before = EVAL_SETS.copy()
    owners_before = registries._OWNERS.copy()
    monkeypatch.setattr(plugins, "_loaded", False)

    def first() -> None:
        registries.register_eval_set("temporary-first", lambda: None)

    def broken() -> None:
        registries.register_control_set("temporary-control", "probe")
        raise ValueError("broken registration")

    entries = [
        SimpleNamespace(name="first", value="first:register", load=lambda: first),
        SimpleNamespace(name="broken", value="broken:register", load=lambda: broken),
    ]
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: entries)
    with pytest.raises(RuntimeError, match="plugin 'broken' failed: broken registration"):
        plugins.load_plugins()
    assert eval_before == EVAL_SETS
    assert "temporary-control" not in registries.PROBES
    assert owners_before == registries._OWNERS
    assert not plugins._loaded

    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    assert plugins.load_plugins() == []
    assert plugins._loaded


def test_plugin_task_references_are_validated_after_all_plugins(monkeypatch) -> None:
    from types import SimpleNamespace

    import pytest

    from grpo_decomp import plugins, registries

    tasks_before = TASKS.copy()

    def invalid() -> None:
        registries.register_task(
            "temporary-task",
            registries.EvalTaskProfile(
                base_config="fixture.yaml",
                task_set="missing-task-set",
                control_sets=(),
                run_prefix="",
                battery_root="battery",
                passk_multiseed_root="passk",
            ),
        )

    entry = SimpleNamespace(name="invalid", value="invalid:register", load=lambda: invalid)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(ValueError, match=r"temporary-task.*invalid.*missing-task-set"):
        plugins.load_plugins(force=True)
    assert tasks_before == TASKS
