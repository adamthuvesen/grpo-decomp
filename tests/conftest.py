"""Register the reference study into the harness registries for the whole test session.

The harness (``grpo_decomp``) ships task-agnostic; the study's eval sets, rewards,
verifiers, and task profiles must be registered before tests that exercise them. We call
the study's ``register()`` directly (rather than going through entry-point discovery) so
the suite does not depend on installed package metadata. Registration is idempotent.
"""

from __future__ import annotations

from llm_grpo_gains.registration import register

register()
