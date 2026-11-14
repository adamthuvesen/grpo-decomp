"""Every row of the AGENTS.md Enforced Rules table cites enforcers that exist."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _rows() -> list[tuple[str, list[str]]]:
    text = (_REPO / "AGENTS.md").read_text()
    assert "\n## Enforced Rules\n" in text, "AGENTS.md must keep a `## Enforced Rules` section"
    table = text.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [line.split("|") for line in lines[1:]]
    return [(cell[1].strip(), re.findall(r"`([^`]+)`", cell[2])) for cell in cells]


def _defines(path: Path, name: str) -> bool:
    if path.suffix == ".toml":
        node: object = tomllib.loads(path.read_text())
        for key in name.split("."):
            if not isinstance(node, dict) or key not in node:
                return False
            node = node[key]
        return True
    escaped = re.escape(name)
    if path.name == "Makefile":
        pattern = rf"^{escaped}:"
    else:
        pattern = rf"^\s*(def|class) {escaped}\b|^{escaped}\s*[:=]"
    return re.search(pattern, path.read_text(), re.MULTILINE) is not None


def test_enforced_rules_cite_existing_definitions() -> None:
    rows = _rows()
    assert rows, "AGENTS.md ## Enforced Rules has no rows"
    uncited = [rule for rule, refs in rows if not refs]
    assert not uncited, f"Enforced Rules rows cite no enforcer: {uncited}"
    missing = []
    for _, refs in rows:
        for ref in refs:
            path, _, name = ref.partition("::")
            file = _REPO / path
            if not file.is_file() or (name and not _defines(file, name)):
                missing.append(ref)
    assert not missing, (
        f"AGENTS.md Enforced Rules cites missing enforcers {missing}. Cite `path` or "
        "`path::name` where the file defines name (def, class, assignment, make target, toml key)."
    )


def test_defines_finds_a_known_definition_and_rejects_a_missing_one() -> None:
    assert _defines(_REPO / "Makefile", "check")
    assert not _defines(_REPO / "Makefile", "chec")
    assert _defines(_REPO / "pyproject.toml", "tool.basedpyright")
    assert not _defines(_REPO / "modal_app.py", "_require_approval_missing")
    assert _defines(_REPO / "modal_app.py", "_require_approval")
