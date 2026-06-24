"""Deterministic rendering and serialization of a `Decomposition`.

Both outputs are pure functions of the decomposition, so regenerating from the
same result artifacts is byte-identical (sorted JSON keys, fixed table layout).
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_grpo_gains.report.decomposition import Decomposition, DecompositionRow
from llm_grpo_gains.report.status import preliminary_suffix


def _row_cells(row: DecompositionRow) -> str:
    c = row.comparison
    return (
        f"| {row.control} | {row.probes} | {c.delta * 100:+.1f} | "
        f"[{c.ci_low * 100:+.1f}, {c.ci_high * 100:+.1f}] | {c.p_value:.3g} | {c.n} |"
    )


def render_table(decomposition: Decomposition) -> str:
    """Render the decomposition as a deterministic Markdown table + panels."""
    status = preliminary_suffix(decomposition.preliminary)
    lines = [
        f"# Decomposition — {decomposition.base_model} on {decomposition.task}{status}",
        "",
        f"Artifact scope: {decomposition.artifact_scope}",
        "",
        f"Confirmatory comparison: {decomposition.confirmatory_comparison.comparison.headline()}",
        "",
        "| Control | Probes | Δ (pp) | 95% CI (pp) | McNemar p | n |",
        "| --- | --- | --- | --- | --- | --- |",
        *(_row_cells(row) for row in decomposition.rows),
        "",
        f"Elicitation (separate panel): {decomposition.elicitation_note}",
        "",
        "Caveats:",
        *(f"- {caveat}" for caveat in decomposition.caveats),
    ]
    return "\n".join(lines) + "\n"


def to_summary_json(decomposition: Decomposition) -> str:
    """Serialize to deterministic JSON (sorted keys) — byte-identical for equal inputs."""
    return json.dumps(decomposition.model_dump(), sort_keys=True, indent=2) + "\n"


def write_summary(decomposition: Decomposition, path: Path) -> None:
    """Write `results/summary.json` (the committed result artifact)."""
    path.write_text(to_summary_json(decomposition), encoding="utf-8")
