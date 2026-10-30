"""Unit tests for the completion artifact: schema, validation, IO (no model, no network)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grpo_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    load_completion_set,
    write_completion_set,
)
from grpo_decomp.schemas import DatasetRef, Problem


def _ref() -> DatasetRef:
    return DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")


def _provenance(*, n: int, n_problems: int) -> GenerationProvenance:
    return GenerationProvenance(
        model="Qwen/Qwen2.5-Math-1.5B",
        model_revision=None,
        backend="transformers",
        sampling=SamplingConfig(temperature=0.0, n=n),
        dataset=_ref(),
        n_problems=n_problems,
        commit="c" * 40,
        python_version="3.11.0",
        package_versions={"grpo-decomp": "0.1.0"},
    )


def _completion_set(*, n: int = 2, ids: tuple[str, ...] = ("a", "b")) -> CompletionSet:
    items = tuple(
        ProblemCompletions(
            problem=Problem(id=pid, question=f"q{pid}", gold_answer="4"),
            samples=tuple(f"{pid}-s{j}" for j in range(n)),
        )
        for pid in ids
    )
    return CompletionSet(provenance=_provenance(n=n, n_problems=len(ids)), items=items)


def test_round_trips_through_disk(tmp_path) -> None:
    original = _completion_set()
    write_completion_set(original, tmp_path / "cs")
    assert load_completion_set(tmp_path / "cs") == original


def test_round_trip_preserves_problem_order_not_lexicographic_id_order(tmp_path) -> None:
    original = _completion_set(ids=("p1", "p2", "p10"))
    write_completion_set(original, tmp_path / "cs")
    loaded = load_completion_set(tmp_path / "cs")
    assert tuple(item.problem.id for item in loaded.items) == ("p1", "p2", "p10")


def test_write_is_byte_identical_for_equal_inputs(tmp_path) -> None:
    write_completion_set(_completion_set(), tmp_path / "one")
    write_completion_set(_completion_set(), tmp_path / "two")
    for name in ("provenance.json", "completions.jsonl"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_non_uniform_sample_counts_are_loud() -> None:
    items = (
        ProblemCompletions(
            problem=Problem(id="a", question="q", gold_answer="4"), samples=("s0", "s1")
        ),
        ProblemCompletions(problem=Problem(id="b", question="q", gold_answer="4"), samples=("s0",)),
    )
    with pytest.raises(ValidationError, match="non-uniform"):
        CompletionSet(provenance=_provenance(n=2, n_problems=2), items=items)


def test_n_problems_mismatch_is_loud() -> None:
    with pytest.raises(ValidationError, match="n_problems"):
        CompletionSet(
            provenance=_provenance(n=1, n_problems=5),
            items=(
                ProblemCompletions(
                    problem=Problem(id="a", question="q", gold_answer="4"), samples=("s0",)
                ),
            ),
        )


def test_empty_completion_set_is_loud() -> None:
    with pytest.raises(ValidationError, match="empty"):
        CompletionSet(provenance=_provenance(n=1, n_problems=0), items=())


def test_duplicate_ids_are_loud() -> None:
    item = ProblemCompletions(
        problem=Problem(id="dup", question="q", gold_answer="4"), samples=("s0",)
    )
    with pytest.raises(ValidationError, match="duplicate"):
        CompletionSet(provenance=_provenance(n=1, n_problems=2), items=(item, item))


def test_problem_set_and_completions_by_id_round_trip() -> None:
    completion_set = _completion_set(n=2, ids=("x", "y"))
    assert {p.id for p in completion_set.problem_set()} == {"x", "y"}
    by_id = completion_set.completions_by_id()
    assert by_id["x"] == ("x-s0", "x-s1")


@pytest.mark.parametrize(
    "kwargs", [{"n": 0}, {"temperature": -0.1}, {"top_p": 1.5}, {"top_p": 0.0}]
)
def test_sampling_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": "0.8"},
        {"temperature": True},
        {"top_p": "1"},
        {"n": True},
        {"n": "2"},
        {"n": 2.0},
        {"max_new_tokens": 2.5},
        {"seed": False},
    ],
)
def test_sampling_config_rejects_nonfinite_and_coerced_numbers(kwargs) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig(**kwargs)


def test_manifest_records_version_and_content_hashes(tmp_path) -> None:
    import hashlib
    import json

    from grpo_decomp.provenance import problem_set_digest

    original = _completion_set()
    path = write_completion_set(original, tmp_path / "cs")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert (
        manifest["provenance_sha256"]
        == hashlib.sha256((path / "provenance.json").read_bytes()).hexdigest()
    )
    assert (
        manifest["completions_sha256"]
        == hashlib.sha256((path / "completions.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["dataset_digest"] == problem_set_digest(original.problem_set())


def test_legacy_artifact_load_and_retry_preserve_all_bytes(tmp_path) -> None:
    import json

    original = _completion_set()
    path = write_completion_set(original, tmp_path / "cs")
    (path / "manifest.json").unlink()
    provenance_path = path / "provenance.json"
    data = json.loads(provenance_path.read_text())
    for key in tuple(data):
        if data[key] is None:
            del data[key]
    # model_revision was required in the original schema.
    data["model_revision"] = None
    provenance_path.write_text(json.dumps(data))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    assert load_completion_set(path) == original
    write_completion_set(original, path)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_conflicting_existing_artifact_is_never_replaced(tmp_path) -> None:
    path = write_completion_set(_completion_set(), tmp_path / "cs")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match="conflicting CompletionSet"):
        write_completion_set(_completion_set(ids=("different",)), path)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


def test_failed_staged_write_cleans_up_and_does_not_publish(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    original_write = Path.write_bytes

    def fail_completions(path, data):
        if path.name == "completions.jsonl":
            raise OSError("disk full")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_completions)
    with pytest.raises(OSError, match="disk full"):
        write_completion_set(_completion_set(), tmp_path / "cs")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["provenance.json", "completions.jsonl"])
def test_corrupted_versioned_artifact_fails_integrity_check(tmp_path, name) -> None:
    path = write_completion_set(_completion_set(), tmp_path / "cs")
    (path / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match=rf"{name}: SHA256"):
        load_completion_set(path)


def test_unsupported_manifest_version_is_loud(tmp_path) -> None:
    import json

    path = write_completion_set(_completion_set(), tmp_path / "cs")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError, match=r"(?s)manifest\.json.*unsupported CompletionSet schema version"
    ):
        load_completion_set(path)


@pytest.mark.parametrize("bad_line", [b"not json", b"\xff", b'{"problem":null}'])
def test_legacy_jsonl_error_names_file_and_line(tmp_path, bad_line) -> None:
    path = write_completion_set(_completion_set(), tmp_path / "cs")
    (path / "manifest.json").unlink()
    completions_path = path / "completions.jsonl"
    first = completions_path.read_bytes().splitlines()[0]
    completions_path.write_bytes(first + b"\n\n" + bad_line + b"\n")
    with pytest.raises(ValueError, match=r"completions\.jsonl:3:"):
        load_completion_set(path)


def test_problem_digest_distinguishes_questions_with_same_dataset_ref() -> None:
    from grpo_decomp.provenance import problem_set_digest
    from grpo_decomp.schemas import ProblemSet

    original = _completion_set().problem_set()
    changed = ProblemSet(
        source=original.source,
        problems=(original[0].model_copy(update={"question": "new question"}), *original[1:]),
    )
    assert problem_set_digest(original) != problem_set_digest(changed)


def test_existing_empty_directory_is_not_published_over(tmp_path) -> None:
    path = tmp_path / "cs"
    path.mkdir()
    with pytest.raises(FileNotFoundError):
        write_completion_set(_completion_set(), path)
    assert list(path.iterdir()) == []
