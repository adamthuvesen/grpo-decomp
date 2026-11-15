"""Each owner of a banned module keeps to its own modules.

Ruff TID251 bans the modules in pyproject.toml. A per-file TID251 ignore lifts every ban for
that file, so this test pins which files hold an ignore and which banned modules each may use.
"""

from __future__ import annotations

import ast
import tomllib
from typing import Any

from test_strict_mode import ROOT, checkout_python_files

SKILL = ".agents/skills/verify-grpo-decomp"
OWNERS: dict[str, set[str]] = {
    "src/grpo_decomp/provenance.py": {"subprocess"},
    "src/grpo_decomp/train/launcher.py": {"datasets"},
    "src/llm_grpo_gains/data/gsm8k.py": {"datasets"},
    "src/llm_grpo_gains/data/gsm8k_platinum.py": {"datasets"},
    "src/llm_grpo_gains/data/gsm_plus.py": {"datasets"},
    "src/llm_grpo_gains/data/gsm_symbolic.py": {"datasets"},
    "src/llm_grpo_gains/workflows.py": {"subprocess"},
    "modal_app.py": {"modal"},
    "tests/test_architecture.py": {"subprocess"},
    "tests/test_esme_countdown_decomp.py": {"subprocess"},
    "tests/test_packaging.py": {"subprocess"},
    "tests/test_strict_mode.py": {"subprocess"},
    f"{SKILL}/drives.py": {"subprocess"},
    f"{SKILL}/launch_plan.py": {"modal"},
    f"{SKILL}/verify": {"subprocess"},
}


def pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]


def banned_modules() -> set[str]:
    return set(pyproject()["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"])


def banned_names_used(source: str, banned: set[str]) -> set[str]:
    """Banned names a module imports or reaches as `os.system`-style attributes."""
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            names.add(f"{aliases[node.value.id]}.{node.attr}")
    return {
        key for key in banned if any(name == key or name.startswith(f"{key}.") for name in names)
    }


def test_only_named_owners_hold_a_tid251_ignore() -> None:
    ignores = pyproject()["ruff"]["lint"]["per-file-ignores"]
    holders = {path for path, rules in ignores.items() if "TID251" in rules}
    assert holders == set(OWNERS), (
        "TID251 ignores and OWNERS disagree. A new owner needs an entry in both: "
        f"only in pyproject {sorted(holders - set(OWNERS))}, only in OWNERS {sorted(set(OWNERS) - holders)}"
    )


def test_every_file_uses_only_the_banned_modules_it_owns() -> None:
    banned = banned_modules()
    violations: list[str] = []
    for path in checkout_python_files():
        name = path.relative_to(ROOT).as_posix()
        used = banned_names_used(path.read_text(encoding="utf-8"), banned)
        extra = {key for key in used if key not in OWNERS.get(name, set())}
        if extra:
            violations.append(f"{name}: {sorted(extra)}")
    assert not violations, (
        "banned modules used outside their owners; move the call into an owner "
        "or add the owner to OWNERS and pyproject per-file-ignores: " + "; ".join(violations)
    )


def test_every_owner_still_uses_what_it_owns() -> None:
    banned = banned_modules()
    stale = [
        name
        for name, owned in OWNERS.items()
        if owned != banned_names_used((ROOT / name).read_text(encoding="utf-8"), banned)
    ]
    assert not stale, (
        f"OWNERS in tests/test_banned_modules.py lists modules these files no longer use or miss: {stale}. "
        "Set each entry to the banned modules the file uses now, or remove the entry and its TID251 per-file-ignore."
    )


def test_detector_finds_known_cases() -> None:
    banned = {"subprocess", "os.system", "http.client", "asyncio.subprocess"}
    assert banned_names_used("import subprocess as sp", banned) == {"subprocess"}
    assert banned_names_used("import os\nos.system('x')", banned) == {"os.system"}
    assert banned_names_used("from http import client", banned) == {"http.client"}
    assert banned_names_used("from http import HTTPStatus", banned) == set()
    assert banned_names_used("from http.client import HTTPConnection", banned) == {"http.client"}
    assert banned_names_used("import os\nos.getcwd()", banned) == set()
