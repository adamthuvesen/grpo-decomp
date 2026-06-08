"""Docs <-> JSON consistency guard (plan #5; built first to guard #1-#4 doc edits).

Every load-bearing number quoted in README / FINDINGS is derived *here* from its
committed JSON artifact and asserted to appear, at quoted precision, in the prose.
This automates the manual grep: regenerate an artifact and a stale doc number now
fails CI; edit a doc number away from its source and it fails too.

Adding a headline number to the docs? Add a claim below pointing at its JSON field
so the doc and the artifact can never silently disagree. The match is a normalized
substring (markdown emphasis stripped, U+2212 minus unified to ASCII), so prose
styling is free to change as long as the number is faithful to its source.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# Glyphs the prose uses in number phrases; named so expected strings read clearly.
_DELTA = "Δ"
_ARROW = "→"


def _load(rel: str) -> dict:
    return json.loads((_ROOT / rel).read_text())


def _norm(text: str) -> str:
    """Canonicalize prose for substring matching.

    Unifies the U+2212 minus to ASCII, drops markdown emphasis/code ticks, and
    collapses whitespace. Keeps %, brackets, +, and the Δ / → glyphs so a claim
    can still pin a number to its arm.
    """
    text = text.replace(chr(0x2212), "-")  # U+2212 minus -> ASCII hyphen
    text = re.sub(r"[*`]", "", text)  # markdown bold / italic / inline code
    text = re.sub(r"\s+", " ", text)
    return text


def _pct(x: float, *, sign: bool = False, dp: int = 1) -> str:
    """Percent string at the docs' precision (Python's f-string emits ASCII '-')."""
    return f"{x * 100:+.{dp}f}" if sign else f"{x * 100:.{dp}f}"


# --- committed artifacts: the single source of truth ------------------------
_GP = _load("results/pass8-multiseed.json")  # GSM8K pass@8 panel (6 seeds)
_CP8 = _load("results/countdown/pass8-multiseed.json")  # Countdown pass@8 (3 seeds)
_GPL = _load("results/seed-placebo-comparison.json")  # GSM8K placebo (6 seeds)
_CPL = _load("results/countdown/seed-placebo-comparison.json")  # Countdown placebo
_SUM = _load("results/summary.json")  # GSM8K seed-0 controls (plan #3 replaces these)
_MECH = _load("results/mechanism.json")  # GSM8K per-problem migration + length shift
_MECH_CD = _load("results/countdown/mechanism.json")
_SYM = _load("results/decontam/pass8-symbolic.json")  # renumbered (GSM-Symbolic)
_PLAT = _load("results/decontam/pass8-platinum.json")  # cleaned labels (GSM8K-Platinum)


def _ctrl(needle: str) -> dict:
    """The decomposition `comparison` whose control name contains `needle`."""
    for row in _SUM["rows"]:
        if needle in row["control"]:
            return row["comparison"]
    raise KeyError(needle)


_FIND_G = "results/FINDINGS.md"
_FIND_C = "results/countdown/FINDINGS.md"
_FIND_D = "results/decontam/FINDINGS.md"
_README = "README.md"

_crf_mean = statistics.fmean(_GP["per_seed_code_reasoning_freq"])

# (claim_id, doc, expected substring derived from the JSON above).
_CLAIMS: list[tuple[str, str, str]] = [
    # -- GSM8K placebo (seed-placebo-comparison.json) --
    (
        "gsm.placebo",
        _FIND_G,
        f"{_pct(_GPL['mean_delta'], sign=True)} pp, 95% CI "
        f"[{_pct(_GPL['ci_low'])}, {_pct(_GPL['ci_high'])}]",
    ),
    (
        "readme.gsm-placebo",
        _README,
        f"{_pct(_GPL['mean_delta'], sign=True)} pp, 95% CI "
        f"[{_pct(_GPL['ci_low'])}, {_pct(_GPL['ci_high'])}]",
    ),
    (
        "xtab.gsm-placebo",
        _FIND_C,
        f"{_pct(_GPL['mean_delta'], sign=True)} pp "
        f"[{_pct(_GPL['ci_low'])}, {_pct(_GPL['ci_high'])}]",
    ),
    # -- GSM8K pass@8 (pass8-multiseed.json) --
    (
        "gsm.base-pass8",
        _FIND_G,
        f"{_pct(_GP['base_passk'])}% "
        f"[{_pct(_GP['base_passk_ci_low'])}, {_pct(_GP['base_passk_ci_high'])}]",
    ),
    (
        "gsm.correct-pass8",
        _FIND_G,
        f"{_pct(_GP['mean_correct_passk'])}% "
        f"[{_pct(_GP['correct_passk_ci_low'])}, {_pct(_GP['correct_passk_ci_high'])}]",
    ),
    ("gsm.delta-point", _FIND_G, f"{_DELTA} {_pct(_GP['delta'], sign=True)} pp"),
    (
        "gsm.delta-propagated",
        _FIND_G,
        f"[{_pct(_GP['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_GP['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "gsm.delta-seedlevel",
        _FIND_G,
        f"[{_pct(_GP['delta_ci_low'], sign=True)}, {_pct(_GP['delta_ci_high'], sign=True)}]",
    ),
    ("gsm.base-crf", _FIND_G, f"base {_pct(_GP['base_code_reasoning_freq'])}%"),
    ("gsm.correct-crf", _FIND_G, f"correct code-reasoning is {_pct(_crf_mean)}%"),
    ("readme.gsm-base-pass8", _README, f"base pass@8 ({_pct(_GP['base_passk'])}%)"),
    ("readme.gsm-correct-pass1", _README, f"correct pass@1 ({_pct(_GP['mean_correct_pass1'])}%)"),
    ("readme.gsm-delta", _README, f"{_DELTA} {_pct(_GP['delta'], sign=True)} pp"),
    (
        "readme.gsm-delta-ci",
        _README,
        f"[{_pct(_GP['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_GP['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "xtab.gsm-delta",
        _FIND_C,
        f"{_pct(_GP['delta'], sign=True)} pp "
        f"[{_pct(_GP['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_GP['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "xtab.gsm-base-correct",
        _FIND_C,
        f"{_pct(_GP['base_passk'])} {_ARROW} {_pct(_GP['mean_correct_passk'])}",
    ),
    # -- CoT-gated pass@k: chain coverage 0 -> uninformative (pass8-multiseed.json) --
    (
        "cot.gsm-chain-coverage",
        _FIND_G,
        f"chain coverage is {_pct(_GP['base_chain_coverage'])}%",
    ),
    ("cot.gsm-cot-passk", _FIND_G, f"is {_pct(_GP['base_cot_passk'])}% for base and correct"),
    (
        "cot.readme-coverage",
        _README,
        f"{_pct(_GP['base_chain_coverage'])}% verifiable-chain coverage",
    ),
    (
        "cot.cd-chain-coverage",
        _FIND_C,
        f"chain coverage is {_pct(_CP8['base_chain_coverage'])}%",
    ),
    ("cot.cd-cot-passk", _FIND_C, f"CoT-gated pass@8 is {_pct(_CP8['base_cot_passk'])}% for both"),
    # -- Mechanism: per-problem migration + length shift (mechanism.json) --
    (
        "mech.gsm-base-reliable",
        _FIND_G,
        f"already solves {_pct(_MECH['frac_base_already_reliable'])}%",
    ),
    ("mech.gsm-migrated", _FIND_G, f"makes {_pct(_MECH['frac_migrated_to_reliable'])}% reliable"),
    ("mech.gsm-new", _FIND_G, f"{_pct(_MECH['frac_new_capability'])}% genuinely new"),
    (
        "mech.gsm-share",
        _FIND_G,
        f"{_pct(_MECH['migration_share_of_gain'], dp=0)}% of the added reliability is migration",
    ),
    (
        "mech.gsm-length",
        _FIND_G,
        f"{_MECH['base_mean_words']:.0f} {_ARROW} {_MECH['correct_mean_words']:.0f} words",
    ),
    (
        "mech.cd-base-reliable",
        _FIND_C,
        f"first-try-reliably ({_pct(_MECH_CD['frac_base_already_reliable'])}%)",
    ),
    ("mech.cd-migrated", _FIND_C, f"makes {_pct(_MECH_CD['frac_migrated_to_reliable'])}% migrated"),
    ("mech.cd-new", _FIND_C, f"{_pct(_MECH_CD['frac_new_capability'])}% genuinely new"),
    (
        "mech.cd-length",
        _FIND_C,
        f"{_MECH_CD['base_mean_words']:.0f} {_ARROW} {_MECH_CD['correct_mean_words']:.0f} words",
    ),
    (
        "mech.readme-gsm-new",
        _README,
        f"{_pct(_MECH['frac_new_capability'])}% of the GSM8K gain is new capability",
    ),
    ("mech.readme-cd-new", _README, f"{_pct(_MECH_CD['frac_new_capability'])}% on Countdown"),
    # -- Decontamination (decontam/pass8-{symbolic,platinum}.json) --
    (
        "decontam.test-base-pass8",  # the gsm8k-test reference row traces to the published panel
        _FIND_D,
        f"{_pct(_GP['base_passk'])}% "
        f"[{_pct(_GP['base_passk_ci_low'])}, {_pct(_GP['base_passk_ci_high'])}]",
    ),
    (
        "decontam.sym-base-pass8",
        _FIND_D,
        f"{_pct(_SYM['base_passk'])}% "
        f"[{_pct(_SYM['base_passk_ci_low'])}, {_pct(_SYM['base_passk_ci_high'])}]",
    ),
    ("decontam.sym-base-pass1", _FIND_D, f"| {_pct(_SYM['base_pass1'])}% |"),
    ("decontam.sym-correct-pass1", _FIND_D, f"| {_pct(_SYM['mean_correct_pass1'])}% |"),
    (
        "decontam.sym-delta",
        _FIND_D,
        f"{_pct(_SYM['delta'], sign=True)} "
        f"[{_pct(_SYM['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_SYM['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "decontam.plat-base-pass8",
        _FIND_D,
        f"{_pct(_PLAT['base_passk'])}% "
        f"[{_pct(_PLAT['base_passk_ci_low'])}, {_pct(_PLAT['base_passk_ci_high'])}]",
    ),
    (
        "decontam.plat-delta",
        _FIND_D,
        f"{_pct(_PLAT['delta'], sign=True)} "
        f"[{_pct(_PLAT['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_PLAT['delta_propagated_ci_high'], sign=True)}]",
    ),
    # the main FINDINGS section-2 decontam bullet quotes both pass@8 envelopes
    ("decontam.gsm-sym-envelope", _FIND_G, f"{_pct(_SYM['base_passk'])}%"),
    ("decontam.gsm-plat-envelope", _FIND_G, f"{_pct(_PLAT['base_passk'])}%"),
    ("decontam.readme-envelope", _README, f"base pass@8 holds at {_pct(_SYM['base_passk'])}%"),
    # -- Countdown placebo (countdown/seed-placebo-comparison.json) --
    (
        "countdown.placebo",
        _FIND_C,
        f"{_pct(_CPL['mean_delta'], sign=True)} pp, 95% CI "
        f"[{_pct(_CPL['ci_low'])}, {_pct(_CPL['ci_high'])}]",
    ),
    (
        "xtab.cd-placebo",
        _FIND_C,
        f"{_pct(_CPL['mean_delta'], sign=True)} pp "
        f"[{_pct(_CPL['ci_low'])}, {_pct(_CPL['ci_high'])}]",
    ),
    ("readme.cd-placebo-point", _README, f"{_pct(_CPL['mean_delta'], sign=True)} pp"),
    ("readme.cd-placebo-ci", _README, f"[{_pct(_CPL['ci_low'])}, {_pct(_CPL['ci_high'])}]"),
    # -- Countdown pass@8 (countdown/pass8-multiseed.json) --
    (
        "countdown.base-pass8",
        _FIND_C,
        f"{_pct(_CP8['base_passk'])}% "
        f"[{_pct(_CP8['base_passk_ci_low'])}, {_pct(_CP8['base_passk_ci_high'])}]",
    ),
    (
        "countdown.correct-pass8",
        _FIND_C,
        f"{_pct(_CP8['mean_correct_passk'])}% "
        f"[{_pct(_CP8['correct_passk_ci_low'])}, {_pct(_CP8['correct_passk_ci_high'])}]",
    ),
    ("countdown.delta-point", _FIND_C, f"{_DELTA} {_pct(_CP8['delta'], sign=True)} pp"),
    (
        "countdown.delta-propagated",
        _FIND_C,
        f"[{_pct(_CP8['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_CP8['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "countdown.delta-seedlevel",
        _FIND_C,
        f"seed-level [{_pct(_CP8['delta_ci_low'])}, {_pct(_CP8['delta_ci_high'])}]",
    ),
    (
        "xtab.cd-delta",
        _FIND_C,
        f"{_pct(_CP8['delta'], sign=True)} pp "
        f"[{_pct(_CP8['delta_propagated_ci_low'], sign=True)}, "
        f"{_pct(_CP8['delta_propagated_ci_high'], sign=True)}]",
    ),
    (
        "xtab.cd-base-correct",
        _FIND_C,
        f"{_pct(_CP8['base_passk'])} {_ARROW} {_pct(_CP8['mean_correct_passk'])}",
    ),
    (
        "countdown.coverage",
        _FIND_C,
        f"coverage {_pct(_CP8['base_passk'])} {_ARROW} {_pct(_CP8['mean_correct_passk'])}",
    ),
    (
        "readme.cd-delta",
        _README,
        f"{_DELTA} {_pct(_CP8['delta'], sign=True)} pp, 95% CI "
        f"[{_pct(_CP8['delta_ci_low'])}, {_pct(_CP8['delta_ci_high'])}]",
    ),
    (
        "readme.cd-base-correct",
        _README,
        f"base {_pct(_CP8['base_passk'])}% {_ARROW} correct {_pct(_CP8['mean_correct_passk'])}%",
    ),
    # -- GSM8K seed-0 controls (summary.json); plan #3 replaces these with multi-seed --
    ("control.gsm-plus", _FIND_G, f"gsm-plus {_pct(_ctrl('gsm-plus')['delta'], sign=True)}"),
    ("control.platinum", _FIND_G, f"platinum {_pct(_ctrl('platinum')['delta'], sign=True)}"),
    (
        "control.gsm-symbolic",
        _FIND_G,
        f"gsm-symbolic {_pct(_ctrl('gsm-symbolic')['delta'], sign=True)}",
    ),
    ("control.format", _FIND_G, f"format contributes {_pct(_ctrl('format')['delta'], sign=True)}"),
    (
        "control.contamination-drop",
        _FIND_G,
        f"Base drops {_pct(_ctrl('raw gain')['accuracy_a'], dp=0)}% "
        f"{_ARROW} {_pct(_ctrl('gsm-symbolic')['accuracy_a'], dp=0)}%",
    ),
]

_DOC_TEXT = {doc: _norm((_ROOT / doc).read_text()) for doc in {c[1] for c in _CLAIMS}}


@pytest.mark.parametrize(("claim_id", "doc", "expected"), _CLAIMS, ids=[c[0] for c in _CLAIMS])
def test_doc_number_traces_to_json(claim_id: str, doc: str, expected: str) -> None:
    assert _norm(expected) in _DOC_TEXT[doc], (
        f"[{claim_id}] {doc} is missing the JSON-derived string {_norm(expected)!r}. "
        "Either the prose drifted from its artifact, or the artifact was regenerated "
        "without updating the doc."
    )
