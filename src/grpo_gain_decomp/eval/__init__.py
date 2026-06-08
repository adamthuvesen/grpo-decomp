"""The evaluation battery: answer extraction, grading, pass@k, and completion
analyzers (CoT-chain verification and code-reasoning detection)."""

from grpo_gain_decomp.eval.answers import extract_lenient, extract_strict, is_correct
from grpo_gain_decomp.eval.battery import BatteryResult, PassK, grade, run_battery
from grpo_gain_decomp.eval.code_reasoning import code_reasoning_frequency, is_code_reasoning
from grpo_gain_decomp.eval.completions import (
    CompletionSet,
    GenerationProvenance,
    ProblemCompletions,
    SamplingConfig,
    capture_generation_provenance,
    load_completion_set,
    write_completion_set,
)
from grpo_gain_decomp.eval.cot import chain_is_valid, has_verifiable_chain, verify_steps
from grpo_gain_decomp.eval.generate import generate, resolve_backend
from grpo_gain_decomp.eval.passk import estimate_pass_at_k, pass_at_k

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
