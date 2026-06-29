"""Discover and load study plugins from the ``grpo_decomp.plugins`` entry-point group.

A study (or a new RL task) ships a zero-argument ``register()`` callable that fills the
registries in :mod:`grpo_decomp.registries`, and declares it as a ``grpo_decomp.plugins``
entry point in its packaging metadata, e.g.::

    [project.entry-points."grpo_decomp.plugins"]
    gsm8k = "llm_grpo_gains.registration:register"

The CLI and Modal entrypoints call :func:`load_plugins` once at startup so the registries
are populated before they are read (e.g. before the argument parser lists ``--set``
choices). The harness keeps no static import of any study, preserving the one-way
dependency — discovery is by metadata, not by name.
"""

from __future__ import annotations

from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "grpo_decomp.plugins"

_loaded = False


def load_plugins(*, force: bool = False) -> list[str]:
    """Load every registered plugin (idempotent); return the loaded plugin names.

    Registration overwrites registry keys, so a repeated call is harmless; the guard just
    avoids redundant imports. Pass ``force=True`` to re-run regardless.
    """
    global _loaded
    if _loaded and not force:
        return []
    names: list[str] = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        register = entry_point.load()
        register()
        names.append(entry_point.name)
    _loaded = True
    return names
