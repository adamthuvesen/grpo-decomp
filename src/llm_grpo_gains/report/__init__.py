"""The decomposition report: assemble, render, and serialize the headline table."""

from llm_grpo_gains.report.decomposition import (
    MIN_SEEDS,
    Decomposition,
    DecompositionRow,
    build_decomposition,
)
from llm_grpo_gains.report.render import render_table, to_summary_json, write_summary

__all__ = [
    "MIN_SEEDS",
    "Decomposition",
    "DecompositionRow",
    "build_decomposition",
    "render_table",
    "to_summary_json",
    "write_summary",
]
