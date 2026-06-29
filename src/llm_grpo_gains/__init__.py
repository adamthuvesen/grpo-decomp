"""llm_grpo_gains — the reference study built on the grpo-decomp harness.

A controlled GRPO decomposition on GSM8K (the primary panel) plus a generated Countdown
positive control. This package supplies the study's datasets, verifiable rewards, and
task profiles; the reusable measurement machinery (training, generation, battery, stats,
report) lives in :mod:`grpo_decomp`. Call :func:`llm_grpo_gains.registration.register`
(done automatically via the ``grpo_decomp.plugins`` entry point) to wire the study into the
harness registries.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
