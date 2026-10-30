"""Sample completions from a model, backend-agnostically.

One interface, two backends: `transformers` (CPU/MPS local generation)
and `vllm` (CUDA high-n generation). The heavy dependency for each backend
is imported *inside its branch*, so this module imports on a host with no CUDA and no
vLLM, and the `transformers` path runs without vLLM ever being touched.

Both backends draw on the shared training prompt and stop on the model's native EOS,
so eval measures the model on the distribution it trained on.
"""

# The CPU environment has no GPU libraries (torch, transformers, trl, vllm).
# pyright: reportMissingImports=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from grpo_decomp import registries
from grpo_decomp.eval.completions import (
    CompletionSet,
    ProblemCompletions,
    SamplingConfig,
    capture_generation_provenance,
)
from grpo_decomp.provenance import content_digest, file_digest, problem_set_digest
from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, PromptStrategy, get_prompt_strategy
from grpo_decomp.schemas import ProblemSet
from grpo_decomp.train.provenance import completed_checkpoint, load_run_provenance

#: Prompts per forward pass on the transformers backend (CPU/MPS memory-bound).
_BATCH_SIZE = 16


@dataclass(frozen=True)
class _GeneratedSamples:
    groups: list[list[str]]
    backend: str
    device: str | None = None
    resolved_model_revision: str | None = None
    prompt_digest: str | None = None


def resolve_backend(backend: str) -> str:
    """Resolve `auto`/`transformers`/`vllm` to a concrete backend.

    `auto` picks `vllm` when a CUDA device is visible, else `transformers`.
    """
    if backend in ("transformers", "vllm"):
        return backend
    if backend != "auto":
        raise ValueError(f"backend must be 'auto', 'transformers', or 'vllm', got {backend!r}")
    return "vllm" if _cuda_available() else "transformers"


def generate(
    model: str,
    problems: ProblemSet,
    config: SamplingConfig,
    *,
    backend: str = "auto",
    model_revision: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
) -> dict[str, list[str]]:
    """Sample `config.n` completions per problem; return `problem id -> samples`.

    The result is uniform in n (every problem gets exactly `config.n` samples) — a
    backend that returns a different count is an explicit error. `prompt_strategy` MUST
    match the strategy the arm trained on, or eval measures an off-distribution prompt.
    """
    output = _sample_completions(
        model,
        problems,
        config,
        backend=backend,
        model_revision=model_revision,
        prompt_strategy=prompt_strategy,
    )
    return {
        problem.id: list(samples) for problem, samples in zip(problems, output.groups, strict=True)
    }


def _sample_completions(
    model: str,
    problems: ProblemSet,
    config: SamplingConfig,
    *,
    backend: str,
    model_revision: str | None,
    prompt_strategy: str,
) -> _GeneratedSamples:
    if not problems:
        raise ValueError("cannot generate an empty problem set")
    if config.temperature == 0.0 and config.n > 1:
        raise ValueError(
            "greedy decoding (temperature=0) yields identical samples; "
            "raise temperature for n>1 sampling"
        )
    resolved = resolve_backend(backend)
    strategy = get_prompt_strategy(prompt_strategy)
    prompts = [strategy.build_prompt(problem.question) for problem in problems]
    if resolved == "transformers":
        output = _generate_transformers(
            model, prompts, config, revision=model_revision, strategy=strategy
        )
    else:
        output = _generate_vllm(model, prompts, config, revision=model_revision)
    if len(output.groups) != len(prompts):
        raise ValueError(
            f"backend returned {len(output.groups)} completion groups for {len(prompts)} prompts"
        )
    for problem, samples in zip(problems, output.groups, strict=True):
        if len(samples) != config.n:
            raise ValueError(
                f"problem {problem.id!r} got {len(samples)} samples, expected n={config.n}"
            )
    return replace(output, prompt_digest=content_digest(prompts))


def generate_completion_set(
    model: str,
    problems: ProblemSet,
    config: SamplingConfig,
    *,
    backend: str = "auto",
    model_revision: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    commit: str | None = None,
    dirty: bool | None = None,
    training_run_id: str | None = None,
    training_base_model: str | None = None,
    training_base_model_revision: str | None = None,
    training_reward: str | None = None,
    training_protocol_digest: str | None = None,
    training_seed: int | None = None,
    training_config_digest: str | None = None,
    checkpoint_digest: str | None = None,
) -> CompletionSet:
    """Sample completions and package them as a provenance-carrying `CompletionSet`.

    The one assembly path for the generation artifact (items in problem order, the
    resolved backend and prompt strategy recorded), shared by the CLI and every Modal
    eval function. `commit`/`dirty` override git-derived provenance (Modal images strip
    `.git`).
    """
    model_path = Path(model)
    run_dir = model_path.parent.parent
    if (
        model_path.is_dir()
        and model_path.parent.name == "checkpoints"
        and (
            (run_dir / "provenance.json").is_file()
            or (run_dir / "status.json").exists()
            or run_dir.parent.name == "attempts"
        )
    ):
        run = load_run_provenance(run_dir)
        if run.attempt_id is not None or (run_dir / "status.json").exists():
            if completed_checkpoint(run_dir).resolve() != model_path.resolve():
                raise ValueError(
                    f"{model_path}: generation requires the completed final checkpoint"
                )
            protocol = run.model_dump(
                include={
                    "base_model",
                    "base_model_revision",
                    "prompt_strategy",
                    "dataset",
                    "train_size",
                    "validation_size",
                    "grpo",
                }
            )
            actual_checkpoint_digest = content_digest(
                {
                    path.relative_to(model_path).as_posix(): file_digest(path)
                    for path in sorted(model_path.rglob("*"))
                    if path.is_file()
                }
            )
            captured = {
                "training_run_id": run.attempt_id,
                "training_base_model": run.base_model,
                "training_base_model_revision": run.base_model_revision,
                "training_reward": run.reward,
                "training_protocol_digest": content_digest(protocol),
                "training_seed": run.seed,
                "training_config_digest": run.config_digest,
                "checkpoint_digest": actual_checkpoint_digest,
            }
            supplied = {
                "training_run_id": training_run_id,
                "training_base_model": training_base_model,
                "training_base_model_revision": training_base_model_revision,
                "training_reward": training_reward,
                "training_protocol_digest": training_protocol_digest,
                "training_seed": training_seed,
                "training_config_digest": training_config_digest,
                "checkpoint_digest": checkpoint_digest,
            }
            for field, value in supplied.items():
                if value is not None and value != captured[field]:
                    raise ValueError(f"{model_path}: {field} conflicts with completed run lineage")
            training_run_id = run.attempt_id
            training_base_model = run.base_model
            training_base_model_revision = run.base_model_revision
            training_reward = run.reward
            training_protocol_digest = content_digest(protocol)
            training_seed = run.seed
            training_config_digest = run.config_digest
            checkpoint_digest = actual_checkpoint_digest
    grader_identity = registries.verifier_identity(problems.source)
    output = _sample_completions(
        model,
        problems,
        config,
        backend=backend,
        model_revision=model_revision,
        prompt_strategy=prompt_strategy,
    )
    items = tuple(
        ProblemCompletions(problem=problem, samples=tuple(samples))
        for problem, samples in zip(problems, output.groups, strict=True)
    )
    provenance = capture_generation_provenance(
        model=model,
        dataset=problems.source,
        sampling=config,
        backend=output.backend,
        device=output.device,
        resolved_model_revision=output.resolved_model_revision,
        training_run_id=training_run_id,
        training_base_model=training_base_model,
        training_base_model_revision=training_base_model_revision,
        training_reward=training_reward,
        training_protocol_digest=training_protocol_digest,
        training_seed=training_seed,
        training_config_digest=training_config_digest,
        checkpoint_digest=checkpoint_digest,
        prompt_digest=output.prompt_digest,
        dataset_digest=problem_set_digest(problems),
        grader_identity=grader_identity,
        n_problems=len(problems),
        model_revision=model_revision,
        prompt_strategy=prompt_strategy,
        commit=commit,
        dirty=dirty,
    )
    return CompletionSet(provenance=provenance, items=items)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _transformers_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _generate_transformers(
    model: str,
    prompts: Sequence[str],
    config: SamplingConfig,
    *,
    revision: str | None,
    strategy: PromptStrategy,
) -> _GeneratedSamples:
    """CPU/MPS backend: batched `model.generate`, native EOS, left-padded."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    strategy.prepare_tokenizer(tokenizer)

    language_model = AutoModelForCausalLM.from_pretrained(model, revision=revision)
    language_model.eval()
    device = _transformers_device()
    language_model.to(device)
    torch.manual_seed(config.seed)

    do_sample = config.temperature > 0.0
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": config.max_new_tokens,
        "num_return_sequences": config.n,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = config.temperature
        gen_kwargs["top_p"] = config.top_p

    groups: list[list[str]] = []
    for start in range(0, len(prompts), _BATCH_SIZE):
        batch = list(prompts[start : start + _BATCH_SIZE])
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = language_model.generate(**encoded, **gen_kwargs)
        generated = out[:, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i in range(len(batch)):
            groups.append(decoded[i * config.n : (i + 1) * config.n])
    revision = getattr(language_model.config, "_commit_hash", None)
    return _GeneratedSamples(
        groups=groups,
        backend="transformers",
        device=str(language_model.device),
        resolved_model_revision=revision if isinstance(revision, str) else None,
    )


def _generate_vllm(
    model: str, prompts: Sequence[str], config: SamplingConfig, *, revision: str | None
) -> _GeneratedSamples:
    """CUDA backend: vLLM with `n` samples per prompt; native EOS via generation_config."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, revision=revision)
    params = SamplingParams(
        n=config.n,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens,
        seed=config.seed,
    )
    outputs = llm.generate(list(prompts), params)
    model_config = llm.llm_engine.model_config
    resolved_revision = getattr(model_config.hf_config, "_commit_hash", None)
    return _GeneratedSamples(
        groups=[[choice.text for choice in output.outputs] for output in outputs],
        backend="vllm",
        device=str(llm.llm_engine.vllm_config.device_config.device),
        resolved_model_revision=resolved_revision if isinstance(resolved_revision, str) else None,
    )
