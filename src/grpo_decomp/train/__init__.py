"""Reproducible single-GPU GRPO training: per-arm config and run provenance.

The TRL launcher and the CPU dry-run run on the rented GPU instance (vLLM is
Linux/CUDA-only); this package holds the GPU-independent pieces — the arm config
and the provenance record — so they can be pinned and reviewed before any spend.
Import from the submodules (``train.config``, ``train.launcher``,
``train.provenance``) directly.
"""
