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
