"""Render the two headline figures from the committed result JSON.

Reads `seed-placebo-comparison.json` (the seed-aggregated placebo comparison) and
`pass8-multiseed.json` (the multi-seed pass@k coverage panel, with CIs) and writes the
headline PNGs next to them. matplotlib is not a project dependency, so run
this with an ephemeral install:

    uv run --with matplotlib python scripts/make_figures.py

The figures are committed artifacts; this script documents their inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
from pydantic import BaseModel

from grpo_decomp.report.mechanism import MechanismReport
from grpo_decomp.report.passk_seeds import PassKMultiSeed
from grpo_decomp.report.seeds import SeedPlaceboComparison

HERE = Path(__file__).parents[1] / "results"
PP = 100  # fractions -> percentage points
T = TypeVar("T", bound=BaseModel)

# Shared palette so all three headline figures read as one set.
BASE_C, RL_C, MEAN_C = "#c7d2e8", "#1d4ed8", "#dc2626"
MUTED, GRID = "#64748b", "#edf1f7"


def _style_axis_grid_and_spines(ax, grid_axis: str = "y") -> None:
    """Despine top/right and drop a faint grid behind the data — the shared look."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.1)
    ax.spines[["top", "right"]].set_visible(False)


def _load_record(path: Path, model: type[T]) -> dict:
    return model.model_validate_json(path.read_text(encoding="utf-8")).model_dump()


def _placebo_comparison_figure(placebo: dict) -> None:
    """A forest-style view of the confirmatory test: correct - random per seed, with the
    seed-level mean and 95% CI. Every seed positive, the interval clearing zero.
    """
    deltas = [d * PP for d in placebo["per_seed_delta"]]
    mean, lo, hi = placebo["mean_delta"] * PP, placebo["ci_low"] * PP, placebo["ci_high"] * PP
    seeds = placebo["seeds"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axis_grid_and_spines(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    rows = list(range(len(seeds), 0, -1))  # seed 0 at top
    ax.axvline(0, color="#cbd5e1", linewidth=1.5)  # the no-effect line
    ax.scatter(deltas, rows, color=RL_C, s=74, zorder=4)
    ax.errorbar(
        mean,
        0,
        xerr=[[mean - lo], [hi - mean]],
        fmt="D",
        color=MEAN_C,
        capsize=6,
        markersize=11,
        linewidth=2.2,
        zorder=4,
    )
    ax.set_yticks([*rows, 0])
    ax.set_yticklabels([f"seed {s}" for s in seeds] + ["mean"])
    ax.set_ylim(-1.0, len(seeds) + 0.6)
    ax.set_xlim(-0.8, max(deltas) + 1.2)
    ax.set_xlabel("correct - random accuracy (percentage points)", fontsize=11)
    ax.set_title(
        f"Placebo comparison:  +{mean:.1f} pp   [{lo:.1f}, {hi:.1f}]",
        fontsize=14.5,
        fontweight="bold",
        pad=30,
    )
    ax.text(
        0.5,
        1.045,
        f"GSM8K-test · {placebo['n_seeds']} seeds · pre-registered confirmatory test "
        "· every seed positive, CI clears 0",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.savefig(HERE / "fig-placebo-comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _passk_curve_figure(panel: dict) -> None:
    """Grouped bars showing RL lifts pass@1 but barely moves pass@8 coverage — the
    visual signature of elicitation rather than capability expansion. pass@8 carries its
    uncertainty: the base anchor's problem-bootstrap CI and correct's seed-level t CI.
    """
    groups = ["pass@1", "pass@8"]
    base_v = [panel["base_pass1"] * PP, panel["base_passk"] * PP]
    rl_v = [panel["mean_correct_pass1"] * PP, panel["mean_correct_passk"] * PP]
    # Asymmetric error bars on pass@8 only — the coverage metric the verdict rests on.
    base_err = [
        [(panel["base_passk"] - panel["base_passk_ci_low"]) * PP],
        [(panel["base_passk_ci_high"] - panel["base_passk"]) * PP],
    ]
    rl_err = [
        [(panel["mean_correct_passk"] - panel["correct_passk_ci_low"]) * PP],
        [(panel["correct_passk_ci_high"] - panel["mean_correct_passk"]) * PP],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style_axis_grid_and_spines(ax, grid_axis="y")
    ax.spines["bottom"].set_visible(False)
    centers, width = [0.0, 1.0], 0.32
    for i, cx in enumerate(centers):
        bx, rx = cx - width / 2 - 0.03, cx + width / 2 + 0.03
        ax.bar(bx, base_v[i], width, color=BASE_C, zorder=3)
        ax.bar(rx, rl_v[i], width, color=RL_C, zorder=3)
        label_b, label_r = base_v[i], rl_v[i]
        if i == 1:  # pass@8: draw the CIs and lift the value labels above the whiskers
            ax.errorbar(bx, base_v[i], yerr=base_err, fmt="none", ecolor=MUTED, capsize=4, zorder=5)
            ax.errorbar(rx, rl_v[i], yerr=rl_err, fmt="none", ecolor="#0b2a8a", capsize=4, zorder=5)
            label_b += base_err[1][0]
            label_r += rl_err[1][0]
        ax.text(
            bx,
            label_b + 1.5,
            f"{base_v[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#475569",
        )
        ax.text(
            rx,
            label_r + 1.5,
            f"{rl_v[i]:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=RL_C,
            fontweight="bold",
        )
        ax.text(cx, -7, groups[i], ha="center", va="top", fontsize=12.5, fontweight="bold")
        ax.text(
            cx,
            -13,
            f"Δ +{rl_v[i] - base_v[i]:.1f}",
            ha="center",
            va="top",
            fontsize=10,
            color=MUTED,
        )
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("accuracy (%)", fontsize=11.5)
    ax.set_title(
        "RL improves pass@1 reliability, but pass@8 coverage was already high",
        fontsize=13.5,
        fontweight="bold",
        pad=34,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASE_C),
        plt.Rectangle((0, 0), 1, 1, color=RL_C),
    ]
    ax.legend(
        handles,
        ["base model", "after RL"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.1,
        columnspacing=1.8,
    )
    fig.savefig(HERE / "fig-passk-curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _contrast_figure(gsm8k: dict, countdown: dict) -> None:
    """The two-sided proof in one chart. The dashed line is what the base could already do at
    pass@8; the arrow is the capability RL *added* on top — a sliver on GSM8K (elicitation),
    a leap on Countdown (expansion). Same decomposition, opposite verdicts.
    """
    base_color, rl_color, expand, flat = "#c7d2e8", "#1d4ed8", "#15803d", "#94a3b8"
    tasks = ["GSM8K", "Countdown"]
    subtitles = ["base already near-saturated", "base lacks the skill"]
    panels = [gsm8k, countdown]
    base_p8 = [gsm8k["base_passk"] * PP, countdown["base_passk"] * PP]
    rl_p8 = [gsm8k["mean_correct_passk"] * PP, countdown["mean_correct_passk"] * PP]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#edf1f7", linewidth=1.1)
    centers, width = [0.0, 1.05], 0.30
    for i, cx in enumerate(centers):
        bx, rx = cx - width / 2 - 0.04, cx + width / 2 + 0.04
        delta = rl_p8[i] - base_p8[i]
        lift = expand if delta >= 10 else flat
        ax.bar(bx, base_p8[i], width, color=base_color, zorder=3)
        ax.bar(rx, rl_p8[i], width, color=rl_color, zorder=3)
        # pass@8 uncertainty: base anchor's problem-bootstrap CI, correct's seed-level t CI.
        panel = panels[i]
        base_err = [
            [(panel["base_passk"] - panel["base_passk_ci_low"]) * PP],
            [(panel["base_passk_ci_high"] - panel["base_passk"]) * PP],
        ]
        rl_err = [
            [(panel["mean_correct_passk"] - panel["correct_passk_ci_low"]) * PP],
            [(panel["correct_passk_ci_high"] - panel["mean_correct_passk"]) * PP],
        ]
        ax.errorbar(bx, base_p8[i], yerr=base_err, fmt="none", ecolor=MUTED, capsize=4, zorder=6)
        ax.errorbar(rx, rl_p8[i], yerr=rl_err, fmt="none", ecolor="#0b2a8a", capsize=4, zorder=6)
        ax.text(
            bx,
            base_p8[i] - 4,
            f"{base_p8[i]:.0f}",
            ha="center",
            va="top",
            fontsize=11,
            color="#475569",
        )
        ax.text(
            rx,
            rl_p8[i] + 2,
            f"{rl_p8[i]:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=rl_color,
            fontweight="bold",
        )
        # dashed = the base's existing ceiling; arrow = what RL added beyond it.
        ax.plot(
            [bx - width / 2, rx + width / 2],
            [base_p8[i]] * 2,
            ls=(0, (4, 3)),
            lw=1.3,
            color="#94a3b8",
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(rx, rl_p8[i] - 0.6),
            xytext=(rx, base_p8[i] + 0.6),
            arrowprops={"arrowstyle": "-|>", "color": lift, "lw": 2.4},
            zorder=5,
        )
        # The magnitude lives up top in the verdict badge (clear space), not crammed by the bars.
        verdict = "expansion" if delta >= 10 else "elicitation"
        ax.text(cx, 121, verdict, ha="center", fontsize=13, fontweight="bold", color=lift)
        ax.text(
            cx, 111, f"+{delta:.1f} pp", ha="center", fontsize=15, fontweight="bold", color=lift
        )
        ax.text(cx, -7, tasks[i], ha="center", va="top", fontsize=12.5, fontweight="bold")
        ax.text(cx, -13.5, subtitles[i], ha="center", va="top", fontsize=10, color="#64748b")
    ax.set_ylim(0, 134)
    ax.set_xlim(-0.58, 1.63)
    ax.set_xticks([])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("pass@8 accuracy (%)", fontsize=11.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title(
        "Did RL expand pass@k coverage — or just elicit it?",
        fontsize=14.5,
        fontweight="bold",
        pad=36,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=base_color),
        plt.Rectangle((0, 0), 1, 1, color=rl_color),
    ]
    ax.legend(
        handles,
        ["base model", "after RL (correct reward)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=1.1,
        columnspacing=1.8,
    )
    fig.savefig(HERE / "fig-passk-contrast.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _mechanism_figure(gsm8k: dict, countdown: dict) -> None:
    """Where the trained model's first-try-reliable solves come from. A stacked bar per task:
    base-already-reliable | migrated (within base pass@8) | new capability (beyond it) | still
    hard. GSM8K has no `new` segment (elicitation); Countdown shows a visible one (expansion).
    """
    cats = [
        ("frac_base_already_reliable", "base already reliable", BASE_C),
        ("frac_migrated_to_reliable", "migrated to reliable", RL_C),
        ("frac_new_capability", "new capability", "#15803d"),
        ("frac_still_hard", "still hard", "#e2e8f0"),
    ]
    tasks = [("GSM8K", gsm8k, 1), ("Countdown", countdown, 0)]

    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=1.1)
    for _name, panel, row in tasks:
        left = 0.0
        for key, _label, color in cats:
            width = panel[key] * PP
            ax.barh(row, width, left=left, height=0.52, color=color, edgecolor="white", zorder=3)
            left += width
        new_pp = panel["frac_new_capability"] * PP
        new_left = (panel["frac_base_already_reliable"] + panel["frac_migrated_to_reliable"]) * PP
        ax.annotate(
            f"new capability {new_pp:.1f}%",
            xy=(new_left, row),
            xytext=(0, 26 if row == 1 else -30),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#15803d",
            arrowprops={"arrowstyle": "-|>", "color": "#15803d", "lw": 1.6},
        )
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["GSM8K", "Countdown"], fontsize=12, fontweight="bold")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of problems (%)", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(
        "Where first-try-reliable solves come from: migration vs new capability",
        fontsize=13.5,
        fontweight="bold",
        pad=30,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
    ax.legend(
        handles,
        [label for _, label, _ in cats],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.0,
        columnspacing=1.3,
    )
    fig.savefig(HERE / "fig-mechanism.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _decontam_figure(test: dict, symbolic: dict, platinum: dict) -> None:
    """base pass@1 vs pass@8 across the published, renumbered, and cleaned distributions.

    Renumbering dents pass@1 (memorization helps the first try) but barely touches the pass@8
    envelope the elicitation verdict rests on — contamination is a pass@1 effect, not a pass@8
    one, so "base already solves it at pass@8" is genuine capability, not leakage.
    """
    dists = [
        ("GSM8K-test\n(published)", test),
        ("GSM-Symbolic\n(renumbered)", symbolic),
        ("GSM8K-Platinum\n(cleaned)", platinum),
    ]
    p1 = [d["base_pass1"] * PP for _, d in dists]
    p8 = [d["base_passk"] * PP for _, d in dists]
    p8_err = [
        [(d["base_passk"] - d["base_passk_ci_low"]) * PP for _, d in dists],
        [(d["base_passk_ci_high"] - d["base_passk"]) * PP for _, d in dists],
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    _style_axis_grid_and_spines(ax)
    xs = [0.0, 1.0, 2.0]
    width = 0.36
    ax.bar([x - width / 2 for x in xs], p1, width, color="#cbd5e1", zorder=3)
    ax.bar([x + width / 2 for x in xs], p8, width, color=RL_C, zorder=3)
    ax.errorbar(
        [x + width / 2 for x in xs],
        p8,
        yerr=p8_err,
        fmt="none",
        ecolor="#0b2a8a",
        capsize=4,
        zorder=5,
    )
    for x, v1, v8 in zip(xs, p1, p8, strict=True):
        ax.text(
            x - width / 2,
            v1 + 1.5,
            f"{v1:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#475569",
        )
        ax.text(
            x + width / 2,
            v8 + 4,
            f"{v8:.0f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=RL_C,
            fontweight="bold",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _ in dists], fontsize=10.5)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("base accuracy (%)", fontsize=11.5)
    ax.set_title(
        "Renumbering dents base pass@1, not the pass@8 envelope",
        fontsize=13.5,
        fontweight="bold",
        pad=30,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#cbd5e1"),
        plt.Rectangle((0, 0), 1, 1, color=RL_C),
    ]
    ax.legend(
        handles,
        ["base pass@1 (first try)", "base pass@8 (the envelope)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=1.1,
        columnspacing=1.8,
    )
    fig.savefig(HERE / "decontam" / "fig-decontam.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _esme_sampled_figure(summary: dict) -> None:
    """Esme's held-out signal is form first, exact solving second."""
    valid = {
        "base": summary["arms"]["base"]["valid_rate"] * PP,
        "random": summary["arms"]["random"]["valid_rate"] * PP,
        "correct": summary["arms"]["correct"]["valid_rate"] * PP,
    }
    pass16 = {
        "base": summary["arms"]["base"]["pass_at_k"]["16"] * PP,
        "random": summary["arms"]["random"]["pass_at_k"]["16"] * PP,
        "correct": summary["arms"]["correct"]["pass_at_k"]["16"] * PP,
    }
    valid_test = summary["valid_rate_tests"]["random"]

    def y(value: float) -> float:
        return 300 - (value / 30) * 260

    def h(value: float) -> float:
        return (value / 30) * 260

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="620"
  viewBox="0 0 920 620" role="img" aria-labelledby="title desc">
  <title id="title">Esme-214M-RL sampled decomposition</title>
  <desc id="desc">Two-panel chart showing that Esme-214M-RL separates from placebo on
    valid-expression form, while exact solving moves in the same direction but is
    underpowered.</desc>
  <style>
    .bg {{ fill: #ffffff; }}
    .panel {{ fill: #ffffff; stroke: #d9dde7; stroke-width: 1; }}
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    .title {{ fill: #1f2937; font-size: 22px; font-weight: 700; }}
    .subtitle {{ fill: #6b7280; font-size: 14px; }}
    .axis {{ stroke: #cfd5df; stroke-width: 1; }}
    .grid {{ stroke: #eef1f6; stroke-width: 1; }}
    .tick {{ fill: #6b7280; font-size: 12px; }}
    .label {{ fill: #374151; font-size: 13px; }}
    .value {{ fill: #1f2937; font-size: 13px; font-weight: 700; }}
    .small {{ fill: #6b7280; font-size: 12px; }}
    .legend {{ fill: #374151; font-size: 13px; }}
  </style>
  <rect class="bg" width="920" height="620" rx="10"/>
  <rect class="panel" x="28" y="24" width="864" height="560" rx="8"/>

  <text class="title" x="56" y="64">Esme-214M-RL: reward sharpens form first</text>
  <text class="subtitle" x="56" y="88">Held-out Countdown-Lite: real reward separates from
    placebo on valid expressions; exact solving remains underpowered.</text>

  <g transform="translate(90 142)">
    <text class="label" x="0" y="0">Form validity</text>
    <text class="small" x="0" y="24">valid-expression rate, 16 samples/problem</text>
    <line class="grid" x1="0" y1="300" x2="330" y2="300"/>
    <line class="grid" x1="0" y1="213.3" x2="330" y2="213.3"/>
    <line class="grid" x1="0" y1="126.7" x2="330" y2="126.7"/>
    <line class="grid" x1="0" y1="40" x2="330" y2="40"/>
    <line class="axis" x1="0" y1="40" x2="0" y2="300"/>
    <line class="axis" x1="0" y1="300" x2="330" y2="300"/>
    <text class="tick" x="-24" y="304" text-anchor="end">0%</text>
    <text class="tick" x="-24" y="217.3" text-anchor="end">10%</text>
    <text class="tick" x="-24" y="130.7" text-anchor="end">20%</text>
    <text class="tick" x="-24" y="44" text-anchor="end">30%</text>
    <rect x="50" y="{y(valid["base"]):.1f}" width="54"
      height="{h(valid["base"]):.1f}" fill="#c7d2e8" rx="4"/>
    <rect x="138" y="{y(valid["random"]):.1f}" width="54"
      height="{h(valid["random"]):.1f}" fill="#aeb6ff" rx="4"/>
    <rect x="226" y="{y(valid["correct"]):.1f}" width="54"
      height="{h(valid["correct"]):.1f}" fill="#ef553b" rx="4"/>
    <text class="value" x="77" y="{y(valid["base"]) - 10:.1f}"
      text-anchor="middle">{valid["base"]:.1f}%</text>
    <text class="value" x="165" y="{y(valid["random"]) - 10:.1f}"
      text-anchor="middle">{valid["random"]:.1f}%</text>
    <text class="value" x="253" y="{y(valid["correct"]) - 10:.1f}"
      text-anchor="middle">{valid["correct"]:.1f}%</text>
    <text class="label" x="77" y="326" text-anchor="middle">base</text>
    <text class="label" x="165" y="326" text-anchor="middle">placebo</text>
    <text class="label" x="253" y="326" text-anchor="middle">real reward</text>
    <text class="value" x="165" y="74" text-anchor="middle">
      +{valid_test["mean_delta"] * PP:.1f} pp</text>
    <text class="small" x="165" y="94" text-anchor="middle">
      95% CI [{valid_test["ci_low"] * PP:.1f}, {valid_test["ci_high"] * PP:.1f}]
    </text>
  </g>

  <g transform="translate(512 142)">
    <text class="label" x="0" y="0">Exact solving</text>
    <text class="small" x="0" y="24">pass@16 exact solve rate</text>
    <line class="grid" x1="0" y1="300" x2="330" y2="300"/>
    <line class="grid" x1="0" y1="213.3" x2="330" y2="213.3"/>
    <line class="grid" x1="0" y1="126.7" x2="330" y2="126.7"/>
    <line class="grid" x1="0" y1="40" x2="330" y2="40"/>
    <line class="axis" x1="0" y1="40" x2="0" y2="300"/>
    <line class="axis" x1="0" y1="300" x2="330" y2="300"/>
    <text class="tick" x="-24" y="304" text-anchor="end">0%</text>
    <text class="tick" x="-24" y="217.3" text-anchor="end">10%</text>
    <text class="tick" x="-24" y="130.7" text-anchor="end">20%</text>
    <text class="tick" x="-24" y="44" text-anchor="end">30%</text>
    <rect x="50" y="{y(pass16["base"]):.1f}" width="54"
      height="{h(pass16["base"]):.1f}" fill="#c7d2e8" rx="4"/>
    <rect x="138" y="{y(pass16["random"]):.1f}" width="54"
      height="{h(pass16["random"]):.1f}" fill="#aeb6ff" rx="4"/>
    <rect x="226" y="{y(pass16["correct"]):.1f}" width="54"
      height="{h(pass16["correct"]):.1f}" fill="#ef553b" rx="4"/>
    <text class="value" x="77" y="{y(pass16["base"]) - 10:.1f}"
      text-anchor="middle">{pass16["base"]:.1f}%</text>
    <text class="value" x="165" y="{y(pass16["random"]) - 10:.1f}"
      text-anchor="middle">{pass16["random"]:.1f}%</text>
    <text class="value" x="253" y="{y(pass16["correct"]) - 10:.1f}"
      text-anchor="middle">{pass16["correct"]:.1f}%</text>
    <text class="label" x="77" y="326" text-anchor="middle">base</text>
    <text class="label" x="165" y="326" text-anchor="middle">placebo</text>
    <text class="label" x="253" y="326" text-anchor="middle">real reward</text>
  </g>

  <g transform="translate(506 520)">
    <rect x="0" y="-10" width="12" height="12" fill="#c7d2e8" rx="2"/>
    <text class="legend" x="18" y="1">base</text>
    <rect x="78" y="-10" width="12" height="12" fill="#aeb6ff" rx="2"/>
    <text class="legend" x="96" y="1">placebo</text>
    <rect x="176" y="-10" width="12" height="12" fill="#ef553b" rx="2"/>
    <text class="legend" x="194" y="1">real reward</text>
  </g>
  <text class="small" x="56" y="560">Preliminary: one training seed. The significant
    signal is form validity; exact solve moves the same way but is underpowered at n=30.
  </text>
</svg>
"""
    (HERE / "esme-countdown" / "fig-sampled-form-vs-exact.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    placebo = _load_record(HERE / "seed-placebo-comparison.json", SeedPlaceboComparison)
    panel = _load_record(HERE / "pass8-multiseed.json", PassKMultiSeed)
    countdown = _load_record(HERE / "countdown" / "pass8-multiseed.json", PassKMultiSeed)
    mech = _load_record(HERE / "mechanism.json", MechanismReport)
    mech_cd = _load_record(HERE / "countdown" / "mechanism.json", MechanismReport)
    symbolic = _load_record(HERE / "decontam" / "pass8-symbolic.json", PassKMultiSeed)
    platinum = _load_record(HERE / "decontam" / "pass8-platinum.json", PassKMultiSeed)
    esme_sampled = json.loads((HERE / "esme-countdown" / "sampled_summary.json").read_text())
    _placebo_comparison_figure(placebo)
    _passk_curve_figure(panel)
    _contrast_figure(panel, countdown)
    _mechanism_figure(mech, mech_cd)
    _decontam_figure(panel, symbolic, platinum)
    _esme_sampled_figure(esme_sampled)
    print(
        "wrote fig-placebo-comparison.png, fig-passk-curve.png, fig-passk-contrast.png, "
        "fig-mechanism.png, decontam/fig-decontam.png, "
        "esme-countdown/fig-sampled-form-vs-exact.svg"
    )


if __name__ == "__main__":
    main()
