"""The plug-in surface between the harness and a study.

``grpo_decomp`` is task-agnostic: it knows how to train a GRPO arm, sample a
``CompletionSet``, grade it, and decompose the gain, but it does not know what GSM8K
or Countdown are. A study (e.g. ``llm_grpo_gains``) fills these mappings with its
datasets, rewards, verifiers, prompt strategies, and task profiles when its registration
module is imported (see :mod:`llm_grpo_gains.registration`).

The dependency is one-way: the harness never imports a study.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from grpo_decomp.schemas import DatasetRef, ProblemSet

if TYPE_CHECKING:
    from grpo_decomp.train.provenance import RunProvenance

#: The placebo reward, provided by the harness (it is the control the method leans on).
PLACEBO_REWARD = "random"

#: The harness's default prompt strategy (registered in :mod:`grpo_decomp.prompts`).
DEFAULT_PROMPT_STRATEGY = "r1_zero"


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
    held-out split for training diagnostics. Held out together because some tasks (GSM8K)
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
    passk_multiseed_root: str


EVAL_SETS: dict[str, EvalSetLoader] = {}
TRAIN_DATASETS: dict[str, TrainDataset] = {}
REWARDS: dict[str, RewardFactory] = {}
VERIFIERS: dict[str, Verifier] = {}
VALIDATION_RECONSTRUCTORS: dict[str, ValidationReconstructor] = {}
PROMPT_STRATEGIES: dict[str, PromptStrategy] = {}
TASKS: dict[str, EvalTaskProfile] = {}
#: ``eval-set slug -> human label`` for the controlled report rows.
PROBES: dict[str, str] = {}
#: The study's default ``"task"`` and ``"training dataset"`` names.
DEFAULTS: dict[str, str] = {}


def default_task() -> str | None:
    """Return the study's explicitly selected task, or None without a study default."""
    return DEFAULTS.get("task")


def default_task_set() -> str | None:
    """Return the study's default eval set, or None without a registered default."""
    name = DEFAULTS.get("task")
    return get_task_profile(name).task_set if name is not None else None


def default_train_dataset() -> str | None:
    """Return the study's default training dataset, or None without a registered default."""
    return DEFAULTS.get("training dataset")


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


def verifier_for(source: DatasetRef, *, name: str | None = None) -> Verifier:
    """Resolve an explicit grader, or retain legacy source-name and boxed-math grading."""
    if name is not None and name != "boxed-math":
        try:
            return VERIFIERS[name]
        except KeyError as exc:
            raise ValueError(f"unknown verifier {name!r} for dataset {source.name!r}") from exc
    if name is None and source.name in VERIFIERS:
        return VERIFIERS[source.name]
    from grpo_decomp.grading import is_correct

    return is_correct


def verifier_identity(source: DatasetRef, *, name: str | None = None) -> str:
    """Return the resolved grader key and callable identity for analysis provenance."""
    verifier = verifier_for(source, name=name)
    key = name if name is not None else source.name if source.name in VERIFIERS else "boxed-math"
    module = getattr(verifier, "__module__", type(verifier).__module__)
    qualified_name = getattr(verifier, "__qualname__", type(verifier).__qualname__)
    return f"{key}:{module}.{qualified_name}"
