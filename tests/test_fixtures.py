"""End-to-end: the multi-seed aggregators run on the committed frozen fixture (no volume).

Proves the on-disk CompletionSet -> aggregator -> schema path on a tiny committed sample, and
(unlike the real Qwen completions, which emit no <<>> chains) exercises the CoT-gated path with
nonzero coverage. Regenerate the fixture with `uv run python tests/fixtures/_generate.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_grpo_gains.eval.completions import load_completion_set
from llm_grpo_gains.report.mechanism import build_mechanism
from llm_grpo_gains.report.passk_seeds import aggregate_passk_seeds

_MINI = Path(__file__).parent / "fixtures" / "mini"


def _correct_sets() -> list:
    return [load_completion_set(_MINI / f"correct-seed{s}__mini") for s in (0, 1)]


def test_mini_fixture_passk_seeds_end_to_end() -> None:
    base = load_completion_set(_MINI / "base__mini")
    panel = aggregate_passk_seeds(
        base, [(0, _correct_sets()[0]), (1, _correct_sets()[1])], task="mini", k=2
    )
    assert panel.n_seeds == 2 and panel.preliminary  # below MIN_SEEDS
    assert panel.base_passk == pytest.approx(0.5)  # 8/12 problems reachable at pass@2
    # CoT-gating is a subset of vanilla, and these fixture chains make it nonzero (real data = 0).
    assert panel.base_cot_passk <= panel.base_passk
    assert panel.mean_correct_cot_passk <= panel.mean_correct_passk
    assert panel.base_chain_coverage > 0.0
    assert panel.base_cot_passk > 0.0


def test_mini_fixture_mechanism_end_to_end() -> None:
    base = load_completion_set(_MINI / "base__mini")
    report = build_mechanism(base, _correct_sets(), task="mini", k=2)
    partition = (
        report.frac_base_already_reliable
        + report.frac_migrated_to_reliable
        + report.frac_new_capability
        + report.frac_still_hard
    )
    assert partition == pytest.approx(1.0)
    assert report.frac_migrated_to_reliable > 0.0  # the base->trained migration is exercised
    assert report.migration_share_of_gain == pytest.approx(1.0)
    assert report.base_mean_words > 0.0 and report.correct_mean_words > 0.0
