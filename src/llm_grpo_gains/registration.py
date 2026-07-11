"""Wire the GSM8K + Countdown reference study into the grpo-decomp harness registries.

:func:`register` is the ``grpo_decomp.plugins`` entry point (declared in pyproject): the
harness CLI and Modal entrypoints call it at startup so the study's eval sets, training
datasets, rewards, verifiers, held-out reconstructors, and task profiles are available.
Registration is idempotent — safe to call more than once.

This module is the study-specific coupling that the harness does not carry.
A new RL task is a sibling of this file (in its own package) with its own ``register``.
"""

from __future__ import annotations

from grpo_decomp.registries import (
    EvalTaskProfile,
    TrainDataset,
    register_control_set,
    register_eval_set,
    register_reward,
    register_task,
    register_train_dataset,
    register_validation_reconstructor,
    register_verifier,
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
    """Register every study dataset, reward, verifier, and task profile (idempotent)."""
    # Named eval sets (`generate --set`, report commands' `--task-set`).
    register_eval_set("gsm8k-test", lambda: load_gsm8k("test"))
    register_eval_set("dev", lambda: dev_slice(load_gsm8k("test")))
    register_eval_set("gsm-symbolic", lambda: load_gsm_symbolic("main"))
    register_eval_set("gsm-plus", lambda: load_gsm_plus("test"))
    register_eval_set("gsm8k-platinum", load_gsm8k_platinum)
    register_eval_set("countdown-test", lambda: load_countdown("test"))
    # Esme-214M-RL decomposition: the held-out Countdown-Lite set the esme-posttrain
    # emitter samples over. The completions arrive as artifacts; the harness only grades.
    register_eval_set("esme-countdown", lambda: load_esme_countdown("heldout_fresh"))

    # Training datasets (ArmConfig.dataset). GSM8K carves a per-seed validation split from
    # train; Countdown ships a fixed, seed-independent one.
    register_train_dataset(
        TrainDataset(
            name="gsm8k",
            load=lambda seed: validation_split(load_gsm8k("train"), seed=seed),
        )
    )
    register_train_dataset(
        TrainDataset(
            name="countdown",
            load=lambda seed: (load_countdown("train"), load_countdown("validation")),
        )
    )

    # Verifiable rewards (ArmConfig.reward); the placebo `random` is harness-provided.
    register_reward("correct", lambda seed: correct)
    register_reward("countdown", lambda seed: countdown)

    # Grading: GSM8K family uses the harness default (math-verify); Countdown overrides.
    register_verifier(_COUNTDOWN_SOURCE, countdown_is_correct)
    # Esme Countdown-Lite is stricter than the general Countdown control (each supplied
    # number used exactly once, + - * only), so it registers its own verifier.
    register_verifier(ESME_COUNTDOWN_SOURCE, esme_countdown_is_correct)

    # Held-out reconstruction for `grpo-decomp heldout`.
    register_validation_reconstructor(_GSM8K_SOURCE, _gsm8k_validation)
    register_validation_reconstructor(
        _COUNTDOWN_SOURCE, lambda provenance: load_countdown("validation")
    )

    # Control eval sets + their probe labels (the §3 controlled rows; GSM8K only).
    register_control_set("gsm-symbolic", "memorization (templated renumbering)")
    register_control_set("gsm8k-platinum", "label noise (cleaned labels)")
    register_control_set("gsm-plus", "robustness (adversarial perturbation)")

    # Eval/Modal task profiles.
    register_task(
        "gsm8k",
        EvalTaskProfile(
            base_config="configs/correct.yaml",
            task_set="gsm8k-test",
            control_sets=("gsm-symbolic", "gsm-plus", "gsm8k-platinum"),
            run_prefix="",
            battery_root="battery",
            passk_multiseed_root="passk-multiseed",
        ),
    )
    register_task(
        "countdown",
        EvalTaskProfile(
            base_config="configs/countdown-correct.yaml",
            task_set="countdown-test",
            control_sets=(),
            run_prefix="countdown-",
            battery_root="battery-countdown",
            passk_multiseed_root="passk-multiseed-countdown",
        ),
    )
