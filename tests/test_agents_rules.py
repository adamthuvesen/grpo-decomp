"""Every enforcer cited in the AGENTS.md Enforced Rules table must exist."""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _cited_enforcers() -> list[str]:
    text = (_REPO / "AGENTS.md").read_text()
    table = text.split("## Enforced Rules", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [row.split("|")[2] for row in rows[1:]]
    return [ref for cell in cells for ref in re.findall(r"`([^`]+)`", cell)]


def test_enforced_rules_cite_existing_paths_and_symbols() -> None:
    cited = _cited_enforcers()
    assert cited, "AGENTS.md ## Enforced Rules lists no enforcers"
    missing = []
    for ref in cited:
        path, _, symbol = ref.partition("::")
        file = _REPO / path
        needle = f"def {symbol}(" if symbol.startswith("test_") else symbol
        if not file.is_file() or needle not in file.read_text():
            missing.append(ref)
    assert not missing, (
        f"AGENTS.md Enforced Rules cites missing enforcers {missing}. "
        "Cite `path` or `path::symbol` for a file, test, or config key that exists."
    )
