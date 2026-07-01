"""Sample completions from a model, backend-agnostically.

One interface, two backends: `transformers` (CPU/MPS local generation)
and `vllm` (CUDA high-n generation). The heavy dependency for each backend
is imported *inside its branch*, so this module imports on a host with no CUDA and no
vLLM, and the `transformers` path runs without vLLM ever being touched.

Both backends draw on the shared training prompt and stop on the model's native EOS,
so eval measures the model on the distribution it trained on.
"""

from __future__ import annotations

from collections.abc import Sequence

from grpo_decomp.eval.completions import (
    CompletionSet,
    ProblemCompletions,
    SamplingConfig,
    capture_generation_provenance,
)
from grpo_decomp.registries import DEFAULT_PROMPT_STRATEGY, PromptStrategy, get_prompt_strategy
from grpo_decomp.schemas import ProblemSet

#: Prompts per forward pass on the transformers backend (CPU/MPS memory-bound).
_BATCH_SIZE = 16


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
    if config.temperature == 0.0 and config.n > 1:
        raise ValueError(
            "greedy decoding (temperature=0) yields identical samples; "
            "raise temperature for n>1 sampling"
        )
    resolved = resolve_backend(backend)
    strategy = get_prompt_strategy(prompt_strategy)
    prompts = [strategy.build_prompt(problem.question) for problem in problems]

    if resolved == "transformers":
        raw = _generate_transformers(
            model, prompts, config, revision=model_revision, strategy=strategy
        )
    else:
        raw = _generate_vllm(model, prompts, config, revision=model_revision)

    if len(raw) != len(prompts):
        raise ValueError(
            f"backend returned {len(raw)} completion groups for {len(prompts)} prompts"
        )
    result: dict[str, list[str]] = {}
    for problem, samples in zip(problems, raw, strict=True):
        if len(samples) != config.n:
            raise ValueError(
                f"problem {problem.id!r} got {len(samples)} samples, expected n={config.n}"
            )
        result[problem.id] = list(samples)
    return result


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
) -> CompletionSet:
    """Sample completions and package them as a provenance-carrying `CompletionSet`.

    The one assembly path for the generation artifact (items in problem order, the
    resolved backend and prompt strategy recorded), shared by the CLI and every Modal
    eval function. `commit`/`dirty` override git-derived provenance (Modal images strip
    `.git`).
    """
    resolved = resolve_backend(backend)
    samples = generate(
        model,
        problems,
        config,
        backend=resolved,
        model_revision=model_revision,
        prompt_strategy=prompt_strategy,
    )
    items = tuple(
        ProblemCompletions(problem=problem, samples=tuple(samples[problem.id]))
        for problem in problems
    )
    provenance = capture_generation_provenance(
        model=model,
        dataset=problems.source,
        sampling=config,
        backend=resolved,
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
) -> list[list[str]]:
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
    return groups


def _generate_vllm(
    model: str, prompts: Sequence[str], config: SamplingConfig, *, revision: str | None
) -> list[list[str]]:
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
    return [[choice.text for choice in output.outputs] for output in outputs]
