"""Reproducible single-GPU GRPO training: per-arm config and run provenance.

The TRL launcher and the CPU dry-run run on the rented GPU instance (vLLM is
Linux/CUDA-only); this package holds the GPU-independent pieces — the arm config
and the provenance record — so they can be pinned and reviewed before any spend.
"""

from llm_grpo_gains.train.config import ArmConfig, GRPOSettings, load_arm_config
from llm_grpo_gains.train.launcher import launch, prepare_run, smoke_overrides, to_dataset
from llm_grpo_gains.train.provenance import PROVENANCE_PACKAGES, RunProvenance, capture_provenance

__all__ = [
    "PROVENANCE_PACKAGES",
    "ArmConfig",
    "GRPOSettings",
    "RunProvenance",
    "capture_provenance",
    "launch",
    "load_arm_config",
    "prepare_run",
    "smoke_overrides",
    "to_dataset",
]
