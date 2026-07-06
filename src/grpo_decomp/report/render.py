"""Deterministic rendering and serialization of a `Decomposition`.

Both outputs are pure functions of the decomposition, so regenerating from the
same result artifacts is byte-identical (sorted JSON keys, fixed table layout).
"""

from __future__ import annotations

from pathlib import Path

from grpo_decomp.report.control_seeds import ControlDecomposition
from grpo_decomp.report.decomposition import Decomposition, DecompositionRow
from grpo_decomp.report.mechanism import MechanismReport
from grpo_decomp.report.passk_seeds import PassKMultiSeed
from grpo_decomp.report.seeds import SeedPlaceboComparison
from grpo_decomp.report.status import preliminary_suffix
from grpo_decomp.schemas import record_json


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
        f"# Decomposition: {decomposition.base_model} on {decomposition.task}{status}",
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
    """Serialize to deterministic JSON (sorted keys); equal inputs produce equal bytes."""
    return record_json(decomposition)


def write_summary(decomposition: Decomposition, path: Path) -> None:
    """Write `results/summary.json` (the committed result artifact)."""
    path.write_text(to_summary_json(decomposition), encoding="utf-8")


def render_seed_placebo(comparison: SeedPlaceboComparison) -> str:
    """Render the seed-level placebo comparison as Markdown."""
    lines = [
        f"# Placebo comparison over {comparison.n_seeds} seed(s) - {comparison.task}",
        "",
        comparison.headline(),
        "",
        "| seed | random | correct | Δ (pp) |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {ra * 100:.1f}% | {ca * 100:.1f}% | {d * 100:+.1f} |"
        for s, ra, ca, d in zip(
            comparison.seeds,
            comparison.per_seed_random_acc,
            comparison.per_seed_correct_acc,
            comparison.per_seed_delta,
            strict=True,
        )
    ]
    return "\n".join(lines) + "\n"


def render_passk_multiseed(panel: PassKMultiSeed) -> str:
    """Render the multi-seed pass@k coverage panel as Markdown."""
    lines = [
        f"# Multi-seed pass@{panel.k} coverage - {panel.task}",
        "",
        panel.headline(),
        panel.cot_headline(),
        "",
        f"| seed | correct pass@1 | correct pass@{panel.k} |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {s} | {p1 * 100:.1f}% | {pk * 100:.1f}% |"
        for s, p1, pk in zip(
            panel.seeds, panel.per_seed_correct_pass1, panel.per_seed_correct_passk, strict=True
        )
    ]
    lines += [
        "",
        f"base pass@1 {panel.base_pass1 * 100:.1f}% · base pass@{panel.k} "
        f"{panel.base_passk * 100:.1f}% "
        f"[{panel.base_passk_ci_low * 100:.1f}, {panel.base_passk_ci_high * 100:.1f}]",
        f"base CoT-gated pass@{panel.k} {panel.base_cot_passk * 100:.1f}% "
        f"[{panel.base_cot_passk_ci_low * 100:.1f}, {panel.base_cot_passk_ci_high * 100:.1f}]",
    ]
    return "\n".join(lines) + "\n"


def render_mechanism(report: MechanismReport) -> str:
    """Render the mechanism report as Markdown."""
    lines = [
        f"# Mechanism - {report.task}",
        "",
        report.headline(),
        "",
        f"base already reliable {report.frac_base_already_reliable * 100:.1f}% · "
        f"migrated {report.frac_migrated_to_reliable * 100:.1f}% · "
        f"new {report.frac_new_capability * 100:.1f}% · "
        f"still hard {report.frac_still_hard * 100:.1f}%",
    ]
    return "\n".join(lines) + "\n"


def render_control_decomposition(decomposition: ControlDecomposition) -> str:
    """Render multi-seed control rows as Markdown."""
    lines = [
        f"# Multi-seed controls - {decomposition.task}",
        "",
        decomposition.headline(),
        "",
        "| control | probes | Δ (pp) | 95% CI | p (raw) | p (Holm) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {row.control} | {row.probes} | {row.mean_delta * 100:+.1f} | "
        f"[{row.ci_low * 100:.1f}, {row.ci_high * 100:.1f}] | {row.p_value:.3g} | "
        f"{row.p_value_holm:.3g}{' *' if row.significant else ''} |"
        for row in decomposition.rows
    ]
    return "\n".join(lines) + "\n"
