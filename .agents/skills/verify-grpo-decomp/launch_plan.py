"""Resolve the paid Modal launch plan and every unapproved launch through the real modal package.

Run by `verify launch-plan` from the repo root. Prints one JSON object. Nothing dispatches.
"""

import json
from pathlib import Path

import modal

import modal_app as m
from grpo_decomp.train.config import load_arm_config
from grpo_decomp.train.launcher import preflight_arm

FUNCTIONS = ["train_arm", "heldout_arm", "eval_matrix", "passk_multiseed"]
CALLS = {
    "train": {"command": "train", "smoke_problems": 8, "max_steps": 5, "spawn": False},
    "heldout": {"command": "heldout"},
    "battery": {"command": "battery", "seed": 1, "scope": "placebo"},
    "passk-multiseed": {"command": "passk-multiseed", "task": "countdown"},
}

arm = preflight_arm(load_arm_config(Path("configs/correct.yaml")), smoke_problems=8, max_steps=5)
refusals: dict[str, str] = {}
for name, kwargs in CALLS.items():
    try:
        m.main(**kwargs)
        refusals[name] = "DISPATCHED"
    except SystemExit as exc:
        refusals[name] = str(exc)
    except Exception as exc:
        refusals[name] = f"CRASH {type(exc).__name__}: {exc}"
for name in ["train_arm", "heldout_arm"]:
    try:
        getattr(m, name).local("configs/correct.yaml")
        refusals[name] = "RAN"
    except SystemExit as exc:
        refusals[name] = str(exc)

plan = {
    "entrypoint_is_modal": isinstance(m.main, modal.app.LocalEntrypoint),  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime proof modal wrapped main
    "image": m.CUDA_IMAGE,
    "gpus": {name: getattr(m, name).spec.gpus for name in FUNCTIONS},
    "arm": {
        "name": arm.name,
        "base_model": arm.base_model,
        "max_steps": arm.grpo.max_steps,
        "reward": arm.reward,
    },
    "refusals": refusals,
}
print(json.dumps(plan))
