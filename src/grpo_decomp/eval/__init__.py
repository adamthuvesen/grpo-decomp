"""The evaluation battery: answer extraction, grading, pass@k, and CoT-chain checks.

Import from the submodules directly; ``generate``/``heldout`` pull in the model
backend lazily, so CPU-only callers should import only the submodule they need.
"""
