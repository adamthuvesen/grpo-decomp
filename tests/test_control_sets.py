"""Unit tests for the control-set loaders (no network)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from llm_grpo_gains.data import gsm8k_platinum, gsm_plus, gsm_symbolic
from llm_grpo_gains.data._common import GoldAnswerError


def test_platinum_assembles_with_marker_gold() -> None:
    rows = [{"question": "Q", "answer": "work\n#### 9"}]
    with patch.object(gsm8k_platinum, "load_dataset", return_value=rows) as mock_load:
        ps = gsm8k_platinum.load_gsm8k_platinum(revision="rev1")

    mock_load.assert_called_once_with(
        gsm8k_platinum.NAME, gsm8k_platinum.CONFIG, split="test", revision="rev1"
    )
    assert ps[0].id == "gsm8k-platinum/main/test/0"
    assert ps[0].gold_answer == "9"
    assert ps.source.revision == "rev1"


def test_symbolic_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="configs"):
        gsm_symbolic.load_gsm_symbolic("p9")


def test_symbolic_assembles_per_config() -> None:
    rows = [{"question": "Q", "answer": "= 20%\n\n#### 20"}]
    with patch.object(gsm_symbolic, "load_dataset", return_value=rows) as mock_load:
        ps = gsm_symbolic.load_gsm_symbolic("p1", revision="rev2")

    mock_load.assert_called_once_with(gsm_symbolic.NAME, "p1", split="test", revision="rev2")
    assert ps[0].id == "gsm-symbolic/p1/test/0"
    assert ps[0].gold_answer == "20"


def test_plus_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="splits"):
        gsm_plus.load_gsm_plus("train")


def test_plus_keeps_fractions_excludes_unanswerable(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {"question": "Q0", "answer": "27"},
        {"question": "Q1", "answer": "3/4"},  # legitimate fraction gold, kept verbatim
        {"question": "Q2", "answer": "None"},  # critical-thinking unanswerable, excluded
        {"question": "Q3", "answer": "14.4"},
    ]
    with (
        caplog.at_level(logging.WARNING),
        patch.object(gsm_plus, "load_dataset", return_value=rows) as mock_load,
    ):
        ps = gsm_plus.load_gsm_plus("test", revision="rev3")

    # GSM-Plus has no config: the call is positional name + split/revision kwargs.
    mock_load.assert_called_once_with(gsm_plus.NAME, split="test", revision="rev3")
    # The unanswerable row is dropped; surviving ids keep their ORIGINAL index.
    assert [p.id for p in ps] == [
        "gsm-plus/test/0",
        "gsm-plus/test/1",
        "gsm-plus/test/3",
    ]
    assert [p.gold_answer for p in ps] == ["27", "3/4", "14.4"]
    assert ps.source.config is None
    assert "excluded 1 unanswerable" in caplog.text


def test_plus_raises_on_truly_malformed_gold() -> None:
    rows = [{"question": "Q0", "answer": "not-a-number"}]
    with (
        patch.object(gsm_plus, "load_dataset", return_value=rows),
        pytest.raises(GoldAnswerError, match="gsm-plus/test/0"),
    ):
        gsm_plus.load_gsm_plus("test")
