"""Wire the GSM8K + Countdown reference study into the grpo-decomp harness registries.

:func:`register` fills the harness mappings in :mod:`grpo_decomp.registries` with the
study's eval sets, training datasets, rewards, verifiers, held-out reconstructors, and task
profiles. Every study entry point (:mod:`llm_grpo_gains.cli`, ``modal_app.py``) calls it.

This module is the study-specific coupling that the harness does not carry.
A new RL task is a sibling of this file (in its own package).
"""

from __future__ import annotations

from grpo_decomp.registries import (
    DEFAULTS,
    EVAL_SETS,
    PROBES,
    REWARDS,
    TASKS,
    TRAIN_DATASETS,
    VALIDATION_RECONSTRUCTORS,
    VERIFIERS,
    EvalTaskProfile,
    TrainDataset,
)
from grpo_decomp.schemas import ProblemSet
from grpo_decomp.splits import dev_slice, validation_split
from grpo_decomp.train.provenance import RunProvenance
from llm_grpo_gains.data import (
    countdown_is_correct,
    load_countdown,
    load_gsm8k,
    load_gsm8k_platinum,
    load_gsm_plus,
    load_gsm_symbolic,
)
from llm_grpo_gains.esme_countdown import (
    ESME_COUNTDOWN_SOURCE,
    esme_countdown_is_correct,
    load_esme_countdown,
)
from llm_grpo_gains.rewards import correct, countdown

#: GSM8K's pinned HuggingFace repo id — the `DatasetRef.name` its loader stamps.
_GSM8K_SOURCE = "openai/gsm8k"
#: Countdown's `DatasetRef.name` (procedurally generated, not a HF repo).
_COUNTDOWN_SOURCE = "countdown"


def _gsm8k_validation(provenance: RunProvenance) -> ProblemSet:
    """Rebuild the exact GSM8K held-out split a run carved from its train split."""
    train = load_gsm8k(provenance.dataset.split, revision=provenance.dataset.revision)
    _, validation = validation_split(train, n=provenance.validation_size, seed=provenance.seed)
    return validation


def register() -> None:
    """Fill the harness registries with the study (safe to repeat)."""
    # Named eval sets (`generate --set`, report commands' `--task-set`).
    EVAL_SETS.update(
        {
            "gsm8k-test": lambda: load_gsm8k("test"),
            "dev": lambda: dev_slice(load_gsm8k("test")),
            "gsm-symbolic": lambda: load_gsm_symbolic("main"),
            "gsm-plus": lambda: load_gsm_plus("test"),
            "gsm8k-platinum": load_gsm8k_platinum,
            "countdown-test": lambda: load_countdown("test"),
            # Esme-214M-RL decomposition: the held-out Countdown-Lite set the esme-posttrain
            # emitter samples over. The completions arrive as artifacts; the harness only grades.
            "esme-countdown": lambda: load_esme_countdown("heldout_large"),
        }
    )

    # Training datasets (ArmConfig.dataset). GSM8K carves a per-seed validation split from
    # train; Countdown ships a fixed, seed-independent one.
    TRAIN_DATASETS.update(
        {
            "gsm8k": TrainDataset(
                name="gsm8k",
                load=lambda seed: validation_split(load_gsm8k("train"), seed=seed),
            ),
            "countdown": TrainDataset(
                name="countdown",
                load=lambda seed: (load_countdown("train"), load_countdown("validation")),
            ),
        }
    )

    # Verifiable rewards (ArmConfig.reward); the placebo `random` is harness-provided.
    REWARDS.update({"correct": lambda seed: correct, "countdown": lambda seed: countdown})

    # Grading: GSM8K family uses the harness default (math-verify); Countdown overrides.
    # Esme Countdown-Lite is stricter than the general Countdown control (each supplied
    # number used exactly once, + - * only), so it has its own verifier.
    VERIFIERS.update(
        {
            _COUNTDOWN_SOURCE: countdown_is_correct,
            ESME_COUNTDOWN_SOURCE: esme_countdown_is_correct,
        }
    )

    # Held-out reconstruction for `grpo-decomp heldout`.
    VALIDATION_RECONSTRUCTORS.update(
        {
            _GSM8K_SOURCE: _gsm8k_validation,
            _COUNTDOWN_SOURCE: lambda provenance: load_countdown("validation"),
        }
    )

    # Control eval sets + their probe labels (the §3 controlled rows; GSM8K only).
    PROBES.update(
        {
            "gsm-symbolic": "memorization (templated renumbering)",
            "gsm8k-platinum": "label noise (cleaned labels)",
            "gsm-plus": "robustness (adversarial perturbation)",
        }
    )

    # Eval/Modal task profiles.
    TASKS.update(
        {
            "gsm8k": EvalTaskProfile(
                base_config="configs/correct.yaml",
                task_set="gsm8k-test",
                control_sets=("gsm-symbolic", "gsm-plus", "gsm8k-platinum"),
                run_prefix="",
                battery_root="battery",
                passk_multiseed_root="passk-multiseed",
            ),
            "countdown": EvalTaskProfile(
                base_config="configs/countdown-correct.yaml",
                task_set="countdown-test",
                control_sets=(),
                run_prefix="countdown-",
                battery_root="battery-countdown",
                passk_multiseed_root="passk-multiseed-countdown",
            ),
            # Same-base positive control: Qwen2.5-Math-1.5B (the GSM8K base) trained on
            # Countdown, so the GSM8K/Countdown contrast no longer changes the base model too.
            "countdown-math": EvalTaskProfile(
                base_config="configs/countdown-math-correct.yaml",
                task_set="countdown-test",
                control_sets=(),
                run_prefix="countdown-math-",
                battery_root="battery-countdown-math",
                passk_multiseed_root="passk-multiseed-countdown-math",
            ),
            # Off-policy placebo: the GSM8K random arm with steps_per_generation > 1, so the PPO
            # clip can bind (the Spurious Rewards mechanism). Evaluated against the GSM8K base.
            "gsm8k-offpolicy": EvalTaskProfile(
                base_config="configs/correct.yaml",
                task_set="gsm8k-test",
                control_sets=(),
                run_prefix="offpolicy-",
                battery_root="battery-offpolicy",
                passk_multiseed_root="passk-multiseed-offpolicy",
            ),
        }
    )

    DEFAULTS.update({"task": "gsm8k", "training dataset": "gsm8k"})
