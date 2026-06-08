"""Integration tests: load the control sets from HuggingFace at pinned revisions.

Network-bound, so marked `integration` and deselected by default. Run with:
    make test-integration
"""

from __future__ import annotations

import re

import pytest

from grpo_gain_decomp.data import load_gsm8k_platinum, load_gsm_plus, load_gsm_symbolic

pytestmark = pytest.mark.integration

# GSM-Plus golds are int/decimal OR fraction (e.g. '3/4'). Loader output is
# comma-stripped, so the numeric branch is intentionally comma-free here.
_NUMERIC_OR_FRACTION = re.compile(r"\A-?\d+(?:\.\d+)?\Z|\A-?\d+/\d+\Z")


def test_gsm_symbolic_main_conforms() -> None:
    ps = load_gsm_symbolic("main")
    assert len(ps) == 5000
    assert ps[0].id == "gsm-symbolic/main/test/0"
    for problem in ps:
        float(problem.gold_answer)  # marker golds are plain numbers


def test_gsm8k_platinum_conforms() -> None:
    ps = load_gsm8k_platinum()
    assert len(ps) == 1209
    assert ps[0].id == "gsm8k-platinum/main/test/0"
    for problem in ps:
        float(problem.gold_answer)


def test_gsm_plus_excludes_unanswerable_and_keeps_fractions() -> None:
    ps = load_gsm_plus("test")
    # 10552 total - 1319 'critical thinking' unanswerable = 9233 kept.
    assert len(ps) == 9233
    ids = [p.id for p in ps]
    assert len(set(ids)) == len(ids), "ids must stay unique after exclusion"
    for problem in ps:
        assert _NUMERIC_OR_FRACTION.match(problem.gold_answer), problem.gold_answer
    # The fraction golds survive (they are answerable).
    assert any("/" in p.gold_answer for p in ps)
