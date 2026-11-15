"""Unit tests for the backend-agnostic sampler — fake backend, no model, no CUDA."""

from __future__ import annotations

import importlib
from typing import Literal

import pytest

from grpo_decomp.eval.completions import SamplingConfig
from grpo_decomp.eval.generate import generate, resolve_backend
from grpo_decomp.schemas import DatasetRef, Problem, ProblemSet

#: The real submodule (via sys.modules), separate from the imported `generate` function.
_GENERATE_MODULE = importlib.import_module("grpo_decomp.eval.generate")


def _problems() -> ProblemSet:
    ref = DatasetRef(name="openai/gsm8k", config="main", split="test", revision="rev")
    problems = (
        Problem(id="p0", question="What is 2+2?", gold_answer="4"),
        Problem(id="p1", question="What is 3+3?", gold_answer="6"),
    )
    return ProblemSet(source=ref, problems=problems)


def _fake_backend(samples_per_prompt: int):
    def backend(model, prompts, config, *, revision, strategy=None):
        return _GENERATE_MODULE._GeneratedSamples(
            groups=[
                [f"{model}:{i}:s{j}" for j in range(samples_per_prompt)]
                for i in range(len(prompts))
            ],
            backend="transformers",
            device="cpu",
        )

    return backend


def test_resolve_backend_explicit() -> None:
    assert resolve_backend("transformers") == "transformers"
    assert resolve_backend("vllm") == "vllm"


def test_resolve_backend_auto_without_cuda_is_transformers(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_cuda_available", lambda: False)
    assert resolve_backend("auto") == "transformers"


def test_resolve_backend_auto_with_cuda_is_vllm(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_cuda_available", lambda: True)
    assert resolve_backend("auto") == "vllm"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        resolve_backend("jax")


def test_generate_keys_by_problem_id(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(1))
    out = generate("m", _problems(), SamplingConfig(temperature=0.0, n=1), backend="transformers")
    assert set(out) == {"p0", "p1"}
    assert all(len(samples) == 1 for samples in out.values())


def test_generate_respects_sample_count(monkeypatch) -> None:
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(4))
    out = generate("m", _problems(), SamplingConfig(temperature=0.8, n=4), backend="transformers")
    assert all(len(samples) == 4 for samples in out.values())


def test_generate_greedy_with_multiple_samples_is_loud() -> None:
    with pytest.raises(ValueError, match="greedy"):
        generate("m", _problems(), SamplingConfig(temperature=0.0, n=2), backend="transformers")


def test_generate_detects_backend_count_mismatch(monkeypatch) -> None:
    # Backend hands back 2 samples though the config asked for 1: explicit error.
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(2))
    with pytest.raises(ValueError, match="expected n=1"):
        generate("m", _problems(), SamplingConfig(temperature=0.0, n=1), backend="transformers")


def test_generation_captures_lineage_and_actual_backend_metadata(monkeypatch) -> None:
    from grpo_decomp.eval.generate import generate_completion_set
    from grpo_decomp.provenance import content_digest, problem_set_digest
    from grpo_decomp.registries import get_prompt_strategy, verifier_identity

    def backend(model, prompts, config, *, revision, strategy):
        return _GENERATE_MODULE._GeneratedSamples(
            groups=[["sample"] for _ in prompts],
            backend="transformers",
            device="cpu",
            resolved_model_revision="resolved-commit",
        )

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", backend)
    problems = _problems()
    artifact = generate_completion_set(
        "model",
        problems,
        SamplingConfig(seed=7),
        backend="transformers",
        model_revision="requested-tag",
        training_run_id="run-id",
        training_seed=42,
        training_config_digest="config",
        checkpoint_digest="checkpoint",
        training_base_model="base",
        training_base_model_revision="base-rev",
        training_reward="reward",
        training_protocol_digest="protocol",
    )
    provenance = artifact.provenance
    assert provenance.device == "cpu"
    assert provenance.model_revision == "requested-tag"
    assert provenance.resolved_model_revision == "resolved-commit"
    assert provenance.training_run_id == "run-id"
    assert provenance.training_seed == 42
    assert provenance.sampling.seed == 7
    assert provenance.training_config_digest == "config"
    assert provenance.checkpoint_digest == "checkpoint"
    assert provenance.training_base_model == "base"
    assert provenance.training_base_model_revision == "base-rev"
    assert provenance.training_reward == "reward"
    assert provenance.training_protocol_digest == "protocol"
    strategy = get_prompt_strategy("r1_zero")
    assert provenance.prompt_digest == content_digest(
        [strategy.build_prompt(problem.question) for problem in problems]
    )
    assert provenance.dataset_digest == problem_set_digest(problems)
    assert provenance.grader_identity == verifier_identity(problems.source)


def test_generation_does_not_manufacture_resolved_revision(monkeypatch) -> None:
    from grpo_decomp.eval.generate import generate_completion_set

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(1))
    provenance = generate_completion_set(
        "local-checkpoint",
        _problems(),
        SamplingConfig(),
        backend="transformers",
        model_revision="requested-tag",
    ).provenance
    assert provenance.resolved_model_revision is None
    assert provenance.training_run_id is None
    assert provenance.training_seed is None


def test_empty_problem_set_fails_before_loading_model(monkeypatch) -> None:
    def unexpected_backend(*args, **kwargs):
        pytest.fail("backend must not load for empty data")

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", unexpected_backend)
    empty = ProblemSet(source=_problems().source, problems=())
    with pytest.raises(ValueError, match="empty problem set"):
        generate("m", empty, SamplingConfig(), backend="transformers")


def test_vllm_records_loaded_config_metadata(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    engine = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(_commit_hash="resolved")),
        vllm_config=SimpleNamespace(device_config=SimpleNamespace(device="cuda:0")),
    )
    llm = SimpleNamespace(
        llm_engine=engine,
        generate=lambda prompts, params: [
            SimpleNamespace(outputs=[SimpleNamespace(text="answer")]) for _ in prompts
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=lambda **kwargs: llm,
            SamplingParams=lambda **kwargs: kwargs,
        ),
    )
    output = _GENERATE_MODULE._generate_vllm(
        "m", ["prompt"], SamplingConfig(), revision="requested"
    )
    assert output.groups == [["answer"]]
    assert output.device == "cuda:0"
    assert output.resolved_model_revision == "resolved"


def _local_checkpoint(
    tmp_path,
    *,
    attempt_id="a" * 32,
    state: Literal["running", "completed", "failed"] = "completed",
    seed=42,
):
    from grpo_decomp.provenance import RunProvenance, RunStatus
    from grpo_decomp.schemas import GRPOSettings, record_json

    run_dir = tmp_path / "arm-seed42" / "attempts" / attempt_id
    checkpoint = run_dir / "checkpoints" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model-weights")
    provenance = RunProvenance(
        attempt_id=attempt_id,
        config_digest="config",
        arm="arm",
        base_model="base",
        base_model_revision="base-rev",
        reward="random",
        prompt_strategy="r1_zero",
        dataset=_problems().source,
        train_size=10,
        validation_size=2,
        seed=seed,
        grpo=GRPOSettings(),
        commit="commit",
        python_version="3.13",
        package_versions={},
    )
    (run_dir / "provenance.json").write_text(record_json(provenance))
    (run_dir / "status.json").write_text(
        record_json(
            RunStatus(
                attempt_id=attempt_id,
                state=state,
            )
        )
    )
    return checkpoint


def test_local_completed_checkpoint_captures_verified_lineage(tmp_path, monkeypatch) -> None:
    from grpo_decomp.eval.generate import generate_completion_set
    from grpo_decomp.provenance import content_digest, file_digest

    checkpoint = _local_checkpoint(tmp_path)
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(1))
    result = generate_completion_set(
        str(checkpoint), _problems(), SamplingConfig(seed=7), backend="transformers"
    ).provenance
    assert result.training_run_id == "a" * 32
    assert result.training_seed == 42
    assert result.sampling.seed == 7
    assert result.training_base_model == "base"
    assert result.training_base_model_revision == "base-rev"
    assert result.training_reward == "random"
    assert result.training_config_digest == "config"
    assert result.training_protocol_digest is not None
    assert result.checkpoint_digest == content_digest(
        {"model.safetensors": file_digest(checkpoint / "model.safetensors")}
    )


@pytest.mark.parametrize("state", ["running", "failed"])
def test_local_incomplete_checkpoint_fails_before_model_load(tmp_path, monkeypatch, state) -> None:
    from grpo_decomp.eval.generate import generate_completion_set

    checkpoint = _local_checkpoint(tmp_path, state=state)

    def unexpected_backend(*args, **kwargs):
        pytest.fail("model must not load from an incomplete attempt")

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", unexpected_backend)
    with pytest.raises(ValueError, match="not completed"):
        generate_completion_set(str(checkpoint), _problems(), SamplingConfig())


def test_local_lineage_conflict_fails_before_model_load(tmp_path, monkeypatch) -> None:
    from grpo_decomp.eval.generate import generate_completion_set

    checkpoint = _local_checkpoint(tmp_path)

    def unexpected_backend(*args, **kwargs):
        pytest.fail("model must not load with conflicting lineage")

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", unexpected_backend)
    with pytest.raises(ValueError, match="training_seed conflicts"):
        generate_completion_set(str(checkpoint), _problems(), SamplingConfig(), training_seed=99)


def test_protocol_digest_matches_across_training_seeds(tmp_path, monkeypatch) -> None:
    from grpo_decomp.eval.generate import generate_completion_set

    first = _local_checkpoint(tmp_path / "first", seed=42)
    second = _local_checkpoint(tmp_path / "second", attempt_id="b" * 32, seed=43)
    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", _fake_backend(1))
    provenances = [
        generate_completion_set(
            str(checkpoint), _problems(), SamplingConfig(), backend="transformers"
        ).provenance
        for checkpoint in (first, second)
    ]
    assert provenances[0].training_seed != provenances[1].training_seed
    assert provenances[0].training_protocol_digest == provenances[1].training_protocol_digest


def test_managed_checkpoint_cannot_hide_status_with_legacy_provenance(
    tmp_path, monkeypatch
) -> None:
    import json

    from grpo_decomp.eval.generate import generate_completion_set

    checkpoint = _local_checkpoint(tmp_path, state="running")
    provenance_path = checkpoint.parent.parent / "provenance.json"
    data = json.loads(provenance_path.read_text())
    del data["attempt_id"]
    provenance_path.write_text(json.dumps(data))

    def unexpected_backend(*args, **kwargs):
        pytest.fail("managed status must remain authoritative")

    monkeypatch.setattr(_GENERATE_MODULE, "_generate_transformers", unexpected_backend)
    with pytest.raises(ValueError, match="not completed"):
        generate_completion_set(str(checkpoint), _problems(), SamplingConfig())
