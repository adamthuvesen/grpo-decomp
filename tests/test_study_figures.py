"""Figure rendering consumes validated records and keeps sources unchanged."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from llm_grpo_gains import figures


@pytest.mark.slow
def test_figures_render_into_requested_output_directory(tmp_path) -> None:
    results = Path(__file__).resolve().parents[1] / "results"
    source_json = tuple(results.rglob("*.json"))
    before = {path: path.read_bytes() for path in source_json}
    paths = figures.render_figures(results, tmp_path / "figures")
    assert tuple(path.relative_to(tmp_path / "figures") for path in paths) == figures.FIGURE_PATHS
    for path in paths:
        content = path.read_bytes()
        if path.suffix == ".png":
            assert content[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack("!II", content[16:24])
            assert width > 0 and height > 0
        else:
            assert content == (results / path.relative_to(tmp_path / "figures")).read_bytes()
    assert {path: path.read_bytes() for path in source_json} == before


def test_invalid_result_fails_before_rendering_or_creating_output(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    original = Path(__file__).resolve().parents[1] / "results/seed-placebo-comparison.json"
    shutil.copyfile(original, results / original.name)
    data = json.loads((results / original.name).read_text())
    data["mean_delta"] = float("nan")
    (results / original.name).write_text(json.dumps(data))

    def unexpected_figure(*args, **kwargs):
        pytest.fail("invalid results must fail before rendering")

    monkeypatch.setattr(figures, "_placebo_comparison_figure", unexpected_figure)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        figures.render_figures(results, out)
    assert not out.exists()
