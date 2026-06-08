"""Unit tests for the GSM8K loader assembly (no network)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from grpo_gain_decomp.data import gsm8k
from grpo_gain_decomp.data._common import GoldAnswerError


def test_load_gsm8k_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="splits"):
        gsm8k.load_gsm8k("validation")


def test_load_gsm8k_assembles_problems_and_records_provenance() -> None:
    rows = [
        {"question": "Q0", "answer": "steps\n#### 1,200"},
        {"question": "Q1", "answer": "steps\n#### 7"},
    ]
    with patch.object(gsm8k, "load_dataset", return_value=rows) as mock_load:
        problem_set = gsm8k.load_gsm8k("train", revision="deadbeef")

    mock_load.assert_called_once_with(gsm8k.NAME, gsm8k.CONFIG, split="train", revision="deadbeef")
    assert [p.id for p in problem_set] == ["gsm8k/main/train/0", "gsm8k/main/train/1"]
    assert [p.gold_answer for p in problem_set] == ["1200", "7"]
    assert problem_set.source.revision == "deadbeef"
    assert problem_set.source.split == "train"


def test_load_gsm8k_propagates_gold_error_with_synthesized_id() -> None:
    rows = [{"question": "Q0", "answer": "no #### marker number"}]
    with (
        patch.object(gsm8k, "load_dataset", return_value=rows),
        pytest.raises(GoldAnswerError, match="gsm8k/main/test/0"),
    ):
        gsm8k.load_gsm8k("test")
