"""The evaluation battery: answer extraction, grading, pass@k, and completion
analyzers (CoT-chain verification and code-reasoning detection)."""

from llm_grpo_gains.eval.answers import extract_lenient, extract_strict, is_correct
from llm_grpo_gains.eval.battery import BatteryResult, PassK, grade, run_battery
from llm_grpo_gains.eval.code_reasoning import code_reasoning_frequency, is_code_reasoning
from llm_grpo_gains.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    capture_generation_provenance,
    load_completion_set,
    write_completion_set,
)
from llm_grpo_gains.eval.cot import chain_is_valid, has_verifiable_chain, verify_steps
from llm_grpo_gains.eval.generate import generate, generate_completion_set, resolve_backend
from llm_grpo_gains.eval.passk import estimate_pass_at_k, pass_at_k

__all__ = [
    "BatteryResult",
    "CompletionSet",
    "GenerationProvenance",
    "PassK",
    "ProblemCompletions",
    "SamplingConfig",
    "capture_generation_provenance",
    "chain_is_valid",
    "code_reasoning_frequency",
    "estimate_pass_at_k",
    "extract_lenient",
    "extract_strict",
    "generate",
    "generate_completion_set",
    "grade",
    "has_verifiable_chain",
    "is_code_reasoning",
    "is_correct",
    "load_completion_set",
    "pass_at_k",
    "resolve_backend",
    "run_battery",
    "verify_steps",
    "write_completion_set",
]
