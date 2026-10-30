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

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from grpo_decomp.schemas import DatasetRef, ProblemSet

if TYPE_CHECKING:
    from grpo_decomp.train.provenance import RunProvenance

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


# --- the registries ------------------------------------------------------------------

EVAL_SETS: dict[str, EvalSetLoader] = {}
TRAIN_DATASETS: dict[str, TrainDataset] = {}
REWARDS: dict[str, RewardFactory] = {}
VERIFIERS: dict[str, Verifier] = {}
VALIDATION_RECONSTRUCTORS: dict[str, ValidationReconstructor] = {}
PROMPT_STRATEGIES: dict[str, PromptStrategy] = {}
TASKS: dict[str, EvalTaskProfile] = {}
#: ``eval-set slug -> human label`` for the controlled report rows.
PROBES: dict[str, str] = {}
_DEFAULTS: dict[str, str] = {}


# --- registration --------------------------------------------------------------------

_OWNERS: dict[tuple[str, str], str] = {}
_OWNER: ContextVar[tuple[str, bool]] = ContextVar("registration_owner", default=("direct", False))


@contextmanager
def registration_owner(name: str, *, idempotent: bool = False) -> Iterator[None]:
    """Attribute registrations to one plugin and explicitly permit repeated registration."""
    token = _OWNER.set((name, idempotent))
    try:
        yield
    finally:
        _OWNER.reset(token)


def _registry_restore[T](registry: dict[str, T]) -> Callable[[], None]:
    snapshot = registry.copy()

    def restore() -> None:
        registry.clear()
        registry.update(snapshot)

    return restore


@contextmanager
def registration_transaction() -> Iterator[None]:
    """Restore all registries and ownership records if plugin startup fails."""
    restore_registries = (
        _registry_restore(EVAL_SETS),
        _registry_restore(TRAIN_DATASETS),
        _registry_restore(REWARDS),
        _registry_restore(VERIFIERS),
        _registry_restore(VALIDATION_RECONSTRUCTORS),
        _registry_restore(PROMPT_STRATEGIES),
        _registry_restore(TASKS),
        _registry_restore(PROBES),
        _registry_restore(_DEFAULTS),
    )
    owners = _OWNERS.copy()
    try:
        yield
    except BaseException:
        for restore in restore_registries:
            restore()
        _OWNERS.clear()
        _OWNERS.update(owners)
        raise


def _register[T](registry: dict[str, T], kind: str, name: str, value: T) -> None:
    owner, idempotent = _OWNER.get()
    key = (kind, name)
    if kind == "reward" and name == PLACEBO_REWARD and owner != "grpo_decomp.rewards":
        raise ValueError(f"reward {name!r} is reserved for grpo_decomp.rewards, got {owner!r}")
    if name in registry:
        previous = _OWNERS.get(key, "unowned")
        if owner == previous and idempotent:
            return
        raise ValueError(f"{kind} {name!r} already belongs to {previous!r}, rejected {owner!r}")
    registry[name] = value
    _OWNERS[key] = owner


def register_eval_set(name: str, loader: EvalSetLoader) -> None:
    """Register a named eval set (a ``generate --set`` target)."""
    _register(EVAL_SETS, "eval set", name, loader)


def register_train_dataset(dataset: TrainDataset) -> None:
    """Register a training dataset (an ``ArmConfig.dataset`` value)."""
    _register(TRAIN_DATASETS, "training dataset", dataset.name, dataset)


def register_reward(name: str, factory: RewardFactory) -> None:
    """Register a reward factory (an ``ArmConfig.reward`` value)."""
    _register(REWARDS, "reward", name, factory)


def register_verifier(source_name: str, verifier: Verifier) -> None:
    """Register a grading verifier, keyed on ``DatasetRef.name`` or an explicit grader name."""
    _register(VERIFIERS, "verifier", source_name, verifier)


def register_validation_reconstructor(source_name: str, fn: ValidationReconstructor) -> None:
    """Register how to rebuild a run's held-out split, keyed on ``DatasetRef.name``."""
    _register(VALIDATION_RECONSTRUCTORS, "validation reconstructor", source_name, fn)


def register_prompt_strategy(strategy: PromptStrategy) -> None:
    """Register a prompt strategy (an ``ArmConfig.prompt_strategy`` / ``--prompt-strategy``)."""
    _register(PROMPT_STRATEGIES, "prompt strategy", strategy.name, strategy)


def register_task(name: str, profile: EvalTaskProfile) -> None:
    """Register an eval/Modal task profile."""
    _register(TASKS, "task", name, profile)


def register_control_set(slug: str, probe: str) -> None:
    """Register a control eval set and its human-readable probe label."""
    _register(PROBES, "control set", slug, probe)


def register_default_task(name: str) -> None:
    """Choose the study's default eval task explicitly."""
    _register(_DEFAULTS, "default", "task", name)


def register_default_train_dataset(name: str) -> None:
    """Choose the study's default training dataset explicitly."""
    _register(_DEFAULTS, "default", "training dataset", name)


def default_task() -> str | None:
    """Return the study's explicitly selected task, or None without a study default."""
    return _DEFAULTS.get("task")


def default_task_set() -> str | None:
    """Return the study's default eval set, or None without a registered default."""
    name = _DEFAULTS.get("task")
    return get_task_profile(name).task_set if name is not None else None


def default_train_dataset() -> str | None:
    """Return the study's default training dataset, or None without a registered default."""
    return _DEFAULTS.get("training dataset")


def validate_registrations() -> None:
    """Check task and control references after every plugin has registered."""
    for kind, registry in (("task", TASKS), ("training dataset", TRAIN_DATASETS)):
        if kind in _DEFAULTS and _DEFAULTS[kind] not in registry:
            raise ValueError(f"default {kind} {_DEFAULTS[kind]!r} is not registered")
    for name, profile in TASKS.items():
        for slug in (profile.task_set, *profile.control_sets):
            if slug not in EVAL_SETS:
                owner = _OWNERS.get(("task", name), "unowned")
                raise ValueError(
                    f"task {name!r} from {owner!r} references unknown eval set {slug!r}"
                )
    for slug in PROBES:
        if slug not in EVAL_SETS:
            owner = _OWNERS.get(("control set", slug), "unowned")
            raise ValueError(f"control set {slug!r} from {owner!r} has no registered eval set")


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
