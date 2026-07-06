"""Claim-status guardrails shared by report construction and rendering.

Single-seed decomposition artifacts are useful diagnostics, but headline claims
come from seed-level aggregators. Keep that policy in one place so rendering,
schemas, and tests do not each spell it slightly differently.
"""

from __future__ import annotations

#: Minimum seeds before a result is a headline claim rather than preliminary.
MIN_HEADLINE_SEEDS = 3


def is_preliminary_seed_count(seeds: int) -> bool:
    """True when `seeds` is below the headline-claim threshold."""
    if seeds < 1:
        raise ValueError(f"seeds must be >= 1, got {seeds}")
    return seeds < MIN_HEADLINE_SEEDS


def artifact_scope_for(seeds: int) -> str:
    """Reader-facing scope warning for a decomposition artifact."""
    if is_preliminary_seed_count(seeds):
        return (
            f"Single-seed descriptive decomposition ({seeds} seed). Treat the comparison below as "
            "a per-seed diagnostic; the headline claim must come from seed-level aggregation "
            "such as `seed-placebo-comparison.json`."
        )
    return (
        f"Seed-aggregated decomposition ({seeds} seeds). The placebo comparison is "
        "eligible as a headline claim because it includes run-to-run seed variance."
    )


def preliminary_suffix(preliminary: bool) -> str:
    """Markdown title suffix for preliminary decomposition artifacts."""
    return " [PRELIMINARY]" if preliminary else ""


def preliminary_caveat_for(seeds: int) -> str | None:
    """Extra caveat for non-headline seed counts, if any."""
    if not is_preliminary_seed_count(seeds):
        return None
    return (
        f"PRELIMINARY: aggregated over {seeds} seed(s) (< {MIN_HEADLINE_SEEDS}). CIs reflect "
        "eval-sampling noise only; they do not include run-to-run seed variance, so this is "
        "not a headline claim."
    )
