"""Unit tests for the `grpo-decomp` CLI — fixture CompletionSets, no model load."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from completion_set_fixtures import (
    dataset_ref as _ref,
)
from completion_set_fixtures import (
    problem_set as _problem_set,
)
from completion_set_fixtures import (
    write_completion_set_dir as _write_cs,
)

from grpo_decomp.eval.cli import main
from grpo_decomp.eval.heldout import discover_checkpoints
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet
from grpo_decomp.train.config import ArmConfig
from grpo_decomp.train.provenance import capture_provenance

#: The real generate submodule (sys.modules), separate from the imported CLI function.
_GENERATE_MODULE = importlib.import_module("grpo_decomp.eval.generate")
_CLI_MODULE = importlib.import_module("grpo_decomp.eval.cli")


def _patch_report_sets(monkeypatch, **sets: ProblemSet) -> None:
    for slug, problems in sets.items():
        monkeypatch.setitem(_CLI_MODULE.EVAL_SETS, slug, lambda problems=problems: problems)


def test_battery_emits_result_json(tmp_path, capsys) -> None:
    _write_cs(tmp_path / "cs", model="base", boxed="4")
    assert main(["battery", "--completions", str(tmp_path / "cs"), "--k", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["n_problems"] == 3
    assert result["lenient_accuracy"] == 1.0
    assert result["strict_accuracy"] == 1.0


def test_battery_writes_to_out_file(tmp_path) -> None:
    _write_cs(tmp_path / "cs", model="base", boxed="0")
    out = tmp_path / "nested" / "result.json"
    assert main(["battery", "--completions", str(tmp_path / "cs"), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["lenient_accuracy"] == 0.0


def test_battery_k_above_n_is_clean_error(tmp_path, capsys) -> None:
    _write_cs(tmp_path / "cs", model="base", boxed="4", n=1)
    assert main(["battery", "--completions", str(tmp_path / "cs"), "--k", "2"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("grpo-decomp:")


def test_report_seeds_rejects_non_greedy_artifact(tmp_path, capsys, monkeypatch) -> None:
    _patch_report_sets(monkeypatch, **{"gsm8k-test": _problem_set()})
    battery = tmp_path / "battery"
    _write_cs(battery / "correct__gsm8k-test", model="correct", boxed="4", n=2, temperature=0.7)
    _write_cs(battery / "random__gsm8k-test", model="random", boxed="0")

    assert main(["report-seeds", "--battery-dirs", str(battery)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "greedy pass@1" in captured.err


def test_report_seeds_unknown_task_set_is_clean_error(tmp_path, capsys) -> None:
    code = main(
        ["report-seeds", "--task-set", "not-a-set", "--battery-dirs", str(tmp_path / "battery")]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "unknown report set" in captured.err
    assert "Traceback" not in captured.err


def test_report_passk_seeds_aggregates_panel(tmp_path, capsys, monkeypatch) -> None:
    _patch_report_sets(monkeypatch, **{"gsm8k-test": _problem_set()})
    # Sampled (n>1) base anchor + two correct seeds — the multi-seed pass@k layout.
    root = tmp_path / "passk"
    _write_cs(root / "base__gsm8k-test", model="base", boxed="0", n=2, temperature=0.7)
    _write_cs(root / "correct-seed0__gsm8k-test", model="c0", boxed="4", n=2, temperature=0.7)
    _write_cs(root / "correct-seed1__gsm8k-test", model="c1", boxed="4", n=2, temperature=0.7)
    out = tmp_path / "pass8.json"

    code = main(
        ["report-passk-seeds", "--completions-dir", str(root), "--k", "1", "--out", str(out)]
    )
    assert code == 0
    assert "Multi-seed pass@1 coverage" in capsys.readouterr().out

    panel = json.loads(out.read_text(encoding="utf-8"))
    assert panel["task"] == "gsm8k-test"
    assert panel["k"] == 1
    assert panel["n_seeds"] == 2 and panel["seeds"] == ["0", "1"]
    assert panel["n_base"] == 2 and panel["n_correct"] == 2
    assert panel["base_passk"] == 0.0  # base boxed='0' (gold=4): all wrong
    assert panel["mean_correct_passk"] == 1.0  # both correct seeds boxed='4': all right
    assert panel["delta"] == 1.0
    assert panel["preliminary"] is True  # 2 < MIN_SEEDS


def test_report_seeds_happy_path(tmp_path, capsys, monkeypatch) -> None:
    _patch_report_sets(monkeypatch, **{"gsm8k-test": _problem_set()})
    battery = tmp_path / "battery"
    _write_cs(battery / "correct__gsm8k-test", model="correct", boxed="4")
    _write_cs(battery / "random__gsm8k-test", model="random", boxed="0")
    out = tmp_path / "placebo.json"

    assert main(["report-seeds", "--battery-dirs", str(battery), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 1
    assert "Placebo comparison" in capsys.readouterr().out


def test_report_mechanism_happy_path(tmp_path, capsys, monkeypatch) -> None:
    _patch_report_sets(monkeypatch, **{"gsm8k-test": _problem_set()})
    root = tmp_path / "passk"
    _write_cs(root / "base__gsm8k-test", model="base", boxed="0", n=2, temperature=0.7)
    _write_cs(root / "correct-seed0__gsm8k-test", model="c0", boxed="4", n=2, temperature=0.7)
    out = tmp_path / "mechanism.json"

    assert (
        main(
            [
                "report-mechanism",
                "--completions-dir",
                str(root),
                "--k",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["task"] == "gsm8k-test"
    assert "Mechanism" in capsys.readouterr().out


def test_report_control_seeds_happy_path(tmp_path, capsys, monkeypatch) -> None:
    battery = tmp_path / "battery"
    ref = _ref(revision="sym")
    _patch_report_sets(monkeypatch, **{"gsm-symbolic": _problem_set(ref=ref)})
    _write_cs(battery / "base__gsm-symbolic", model="base", boxed="4", ref=ref)
    _write_cs(battery / "correct__gsm-symbolic", model="correct", boxed="4", ref=ref)
    out = tmp_path / "controls.json"

    assert (
        main(
            [
                "report-control-seeds",
                "--battery-dirs",
                str(battery),
                "--control-sets",
                "gsm-symbolic",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rows"][0]["control"] == "gsm-symbolic"
    assert "Multi-seed controls" in capsys.readouterr().out


def test_report_passk_seeds_requires_correct_seed_dirs(tmp_path, capsys, monkeypatch) -> None:
    _patch_report_sets(monkeypatch, **{"gsm8k-test": _problem_set()})
    root = tmp_path / "passk"
    _write_cs(root / "base__gsm8k-test", model="base", boxed="0", n=2, temperature=0.7)

    assert main(["report-passk-seeds", "--completions-dir", str(root), "--k", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "correct-seed" in captured.err


def test_generate_missing_backend_dependency_is_clean_error(tmp_path, capsys, monkeypatch) -> None:
    def missing_backend(*args, **kwargs):
        raise ImportError("No module named 'vllm'")

    monkeypatch.setattr(_GENERATE_MODULE, "generate", missing_backend)
    monkeypatch.setattr(_GENERATE_MODULE, "resolve_backend", lambda backend: "vllm")
    _patch_report_sets(monkeypatch, dev=_problem_set())

    assert (
        main(
            [
                "generate",
                "--model",
                "m",
                "--set",
                "dev",
                "--backend",
                "vllm",
                "--out",
                str(tmp_path),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing generation backend dependency" in captured.err


def test_unknown_backend_rejected_by_argparse(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["generate", "--model", "m", "--set", "dev", "--backend", "x", "--out", str(tmp_path)])


def test_invalid_sampling_is_clean_error(tmp_path, capsys) -> None:
    # n=0 fails SamplingConfig validation before any dataset load (no network).
    assert (
        main(["generate", "--model", "m", "--set", "dev", "--n", "0", "--out", str(tmp_path)]) == 1
    )
    assert capsys.readouterr().err.startswith("grpo-decomp:")


def test_missing_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_discover_checkpoints_orders_by_step_with_final_last(tmp_path) -> None:
    for name in ("checkpoint-100", "checkpoint-50", "final", "not-a-checkpoint"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    found = [p.name for p in discover_checkpoints(tmp_path)]
    assert found == ["checkpoint-50", "checkpoint-100", "final"]


def test_discover_checkpoints_errors_when_empty(tmp_path) -> None:
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(ValueError, match="no 'checkpoint"):
        discover_checkpoints(tmp_path)


@pytest.mark.parametrize(
    ("flag", "value"), [("--n", "2"), ("--temperature", "0.7"), ("--seed", "1")]
)
def test_heldout_rejects_sampling_flags(tmp_path, flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        main(["heldout", "--run", str(tmp_path), flag, value])


def _write_run_provenance(run_dir: Path) -> None:
    arm = ArmConfig(name="correct", base_model="m", reward="correct", seed=0)
    ref = DatasetRef(name="openai/gsm8k", config="main", split="train", revision="r")
    provenance = capture_provenance(arm, ref, train_size=100, validation_size=2)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance.model_dump(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _fake_generate_correct_on(needle: str):
    """Fake generate: a checkpoint whose path contains `needle` answers correctly (gold=4)."""

    def fake(
        model, problems, config, *, backend="auto", model_revision=None, prompt_strategy="r1_zero"
    ):
        boxed = "4" if needle in str(model) else "9"
        return {p.id: [f"\\boxed{{{boxed}}}"] for p in problems}

    return fake


def _val_problems() -> ProblemSet:
    return ProblemSet(
        source=DatasetRef(name="openai/gsm8k", config="main", split="train", revision="r"),
        problems=(
            Problem(id="v0", question="q", gold_answer="4"),
            Problem(id="v1", question="q", gold_answer="4"),
        ),
    )


def test_heldout_writes_curve_for_all_checkpoints(tmp_path, monkeypatch, capsys) -> None:
    run = tmp_path / "correct-seed0"
    _write_run_provenance(run)
    for name in ("checkpoint-50", "checkpoint-100", "final"):
        (run / "checkpoints" / name).mkdir(parents=True)
    monkeypatch.setattr(
        "grpo_decomp.eval.heldout.validation_for_run", lambda provenance: _val_problems()
    )
    monkeypatch.setattr(_GENERATE_MODULE, "generate", _fake_generate_correct_on("final"))

    assert main(["heldout", "--run", str(run), "--backend", "transformers"]) == 0

    payload = json.loads((run / "heldout.json").read_text(encoding="utf-8"))
    assert payload["validation_size"] == 2
    curve = [(pt["checkpoint"], pt["step"], pt["accuracy"]) for pt in payload["points"]]
    assert curve == [
        ("checkpoint-50", 50, 0.0),
        ("checkpoint-100", 100, 0.0),
        ("final", None, 1.0),
    ]
    assert "held-out acc" in capsys.readouterr().out


def test_generate_rejects_nonpositive_limit(tmp_path, capsys) -> None:
    code = main(
        ["generate", "--model", "m", "--set", "dev", "--limit", "0", "--out", str(tmp_path)]
    )
    assert code == 1
    assert capsys.readouterr().err.startswith("grpo-decomp:")
