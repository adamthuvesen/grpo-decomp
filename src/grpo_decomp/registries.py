"""The plug-in surface between the harness and a study.

``grpo_decomp`` is task-agnostic: it knows how to train a GRPO arm, sample a
``CompletionSet``, grade it, and decompose the gain — but it does not know what GSM8K
or Countdown are. A *study* (e.g. ``llm_grpo_gains``) or a new RL task fills these
registries with its concrete datasets, rewards, verifiers, prompt strategies, and task
profiles. Registration happens either by importing this module and calling the
``register_*`` functions directly, or — for the CLI and Modal entrypoints — via a
``grpo_decomp.plugins`` entry point loaded at startup (see :mod:`grpo_decomp.plugins`).

The dependency is one-way: the harness never imports a study. That is what lets the
owner point this stack at their own model + task without forking it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from grpo_decomp.schemas import DatasetRef, ProblemSet

if TYPE_CHECKING:
    from grpo_decomp.train.provenance import RunProvenance

#: The arms a decomposition compares. `correct` is the treatment arm (the real reward),
#: `random` the placebo control; both are compared against `base`. These are *arm labels*
#: (the ``<arm>__<set>`` run-dir prefix), independent of which reward function backs the
#: treatment — a new task keeps ``name: correct`` while setting ``reward: <its reward>``.
ARMS: tuple[str, ...] = ("base", "correct", "random")

#: The placebo reward, provided by the harness (it is the control the method leans on).
PLACEBO_REWARD = "random"

#: The harness's default prompt strategy (registered in :mod:`grpo_decomp.prompts`).
DEFAULT_PROMPT_STRATEGY = "r1_zero"


# --- types ---------------------------------------------------------------------------

#: A reward: generated ``completions`` + forwarded dataset columns -> one score each.
RewardFn = Callable[..., list[float]]
#: Builds a reward for one run (``seed`` parameterizes stateful rewards like the placebo).
RewardFactory = Callable[[int], RewardFn]
#: Grades an extracted answer against the gold key -> correct?
Verifier = Callable[[str | None, str], bool]
#: Loads a named eval set as a canonical ``ProblemSet``.
EvalSetLoader = Callable[[], ProblemSet]
#: Reconstructs the exact held-out split a finished run used, from its provenance.
ValidationReconstructor = Callable[["RunProvenance"], ProblemSet]


@dataclass(frozen=True)
class PromptStrategy:
    """How a task turns a question into a prompt and prepares the tokenizer.

    ``build_prompt`` MUST produce the *identical* wording at train and eval time — the
    decomposition is invalid if the model is evaluated on a distribution it never trained
    on. ``prepare_tokenizer`` runs the model-side setup (pad/eos token, padding side)
    needed before batched generation or training.
    """

    name: str
    build_prompt: Callable[[str], str]
    prepare_tokenizer: Callable[[Any], None]


@dataclass(frozen=True)
class TrainDataset:
    """The training data for one task, selected by ``ArmConfig.dataset``.

    ``load(seed)`` returns ``(train, validation)``: the GRPO training split and the
    held-out split for checkpoint selection. Held out together because some tasks (GSM8K)
    carve the validation set out of train per seed, while others (Countdown) ship a fixed,
    seed-independent one.
    """

    name: str
    load: Callable[[int], tuple[ProblemSet, ProblemSet]]


@dataclass(frozen=True)
class EvalTaskProfile:
    """Modal/eval wiring for one study task (its base config, eval/control sets, run dirs)."""

    base_config: str
    task_set: str
    control_sets: tuple[str, ...]
    run_prefix: str
    battery_root: str
    elicitation_root: str
    passk_multiseed_root: str


# --- the registries ------------------------------------------------------------------

EVAL_SETS: dict[str, EvalSetLoader] = {}
TRAIN_DATASETS: dict[str, TrainDataset] = {}
REWARDS: dict[str, RewardFactory] = {}
VERIFIERS: dict[str, Verifier] = {}
VALIDATION_RECONSTRUCTORS: dict[str, ValidationReconstructor] = {}
PROMPT_STRATEGIES: dict[str, PromptStrategy] = {}
TASKS: dict[str, EvalTaskProfile] = {}
#: Control eval-set names (perturbation / clean-label probes), filled by the study.
CONTROL_SETS: list[str] = []
#: ``eval-set slug -> human label`` for the controlled report rows.
PROBES: dict[str, str] = {}


# --- registration --------------------------------------------------------------------


def register_eval_set(name: str, loader: EvalSetLoader) -> None:
    """Register a named eval set (a ``generate --set`` target)."""
    EVAL_SETS[name] = loader


def register_train_dataset(dataset: TrainDataset) -> None:
    """Register a training dataset (an ``ArmConfig.dataset`` value)."""
    TRAIN_DATASETS[dataset.name] = dataset


def register_reward(name: str, factory: RewardFactory) -> None:
    """Register a reward factory (an ``ArmConfig.reward`` value)."""
    REWARDS[name] = factory


def register_verifier(source_name: str, verifier: Verifier) -> None:
    """Register a grading verifier for a dataset, keyed on ``DatasetRef.name``.

    Datasets without an override are graded by the harness default (math-verify equality
    on the boxed answer), so most math tasks register nothing here.
    """
    VERIFIERS[source_name] = verifier


def register_validation_reconstructor(source_name: str, fn: ValidationReconstructor) -> None:
    """Register how to rebuild a run's held-out split, keyed on ``DatasetRef.name``."""
    VALIDATION_RECONSTRUCTORS[source_name] = fn


def register_prompt_strategy(strategy: PromptStrategy) -> None:
    """Register a prompt strategy (an ``ArmConfig.prompt_strategy`` / ``--prompt-strategy``)."""
    PROMPT_STRATEGIES[strategy.name] = strategy


def register_task(name: str, profile: EvalTaskProfile) -> None:
    """Register an eval/Modal task profile."""
    TASKS[name] = profile


def register_control_set(slug: str, probe: str) -> None:
    """Register a control eval set and its human-readable probe label."""
    if slug not in CONTROL_SETS:
        CONTROL_SETS.append(slug)
    PROBES[slug] = probe


# --- lookup --------------------------------------------------------------------------


def get_prompt_strategy(name: str) -> PromptStrategy:
    """Return a registered prompt strategy by name."""
    try:
        return PROMPT_STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt strategy {name!r}; registered: {tuple(sorted(PROMPT_STRATEGIES))}"
        ) from exc


def get_task_profile(name: str) -> EvalTaskProfile:
    """Return a registered task profile by name."""
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"eval task must be one of {tuple(sorted(TASKS))}, got {name!r}") from exc


def verifier_for(source: DatasetRef) -> Verifier:
    """Select the grading verifier for a dataset: a registered override, else the default.

    The default is math-verify exact equality on the boxed answer (the harness's answer
    contract); a task with a different notion of correctness registers its own verifier.
    """
    if source.name in VERIFIERS:
        return VERIFIERS[source.name]
    # Lazy import keeps this module free of the eval layer at import time.
    from grpo_decomp.grading import is_correct

    return is_correct
