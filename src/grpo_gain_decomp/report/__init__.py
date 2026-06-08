"""The decomposition report: assemble, render, and serialize the headline table."""

from grpo_gain_decomp.report.decomposition import (
    MIN_SEEDS,
    Decomposition,
    DecompositionRow,
    build_decomposition,
)
from grpo_gain_decomp.report.render import render_table, to_summary_json, write_summary

__all__ = [
    "MIN_SEEDS",
    "Decomposition",
    "DecompositionRow",
    "build_decomposition",
    "render_table",
    "to_summary_json",
    "write_summary",
]
