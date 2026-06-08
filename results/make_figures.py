"""Render the two headline figures from the committed result JSON.

Reads `seed-placebo-comparison.json` (the seed-aggregated placebo comparison) and
`pass8-multiseed.json` (the multi-seed pass@k coverage panel, with CIs) and writes the
headline PNGs next to them. matplotlib is intentionally NOT a project dependency, so run
this with an ephemeral install:

    uv run --with matplotlib python results/make_figures.py

The figures are committed artifacts; this script documents exactly how they were made.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PP = 100  # fractions -> percentage points

# Shared palette so all three headline figures read as one set.
BASE_C, RL_C, MEAN_C = "#c7d2e8", "#1d4ed8", "#dc2626"
MUTED, GRID = "#64748b", "#edf1f7"


def _clean(ax, grid_axis: str = "y") -> None:
    """Despine top/right and drop a faint grid behind the data — the shared look."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.1)
    ax.spines[["top", "right"]].set_visible(False)


def _placebo_comparison_figure(placebo: dict) -> None:
    """A forest-style view of the confirmatory test: correct - random per seed, with the
    seed-level mean and 95% CI. Every seed positive, the interval clearing zero.
    """
    deltas = [d * PP for d in placebo["per_seed_delta"]]
    mean, lo, hi = placebo["mean_delta"] * PP, placebo["ci_low"] * PP, placebo["ci_high"] * PP
    seeds = placebo["seeds"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _clean(ax, grid_axis="x")
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
    _clean(ax, grid_axis="y")
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


def main() -> None:
    placebo = json.loads((HERE / "seed-placebo-comparison.json").read_text(encoding="utf-8"))
    panel = json.loads((HERE / "pass8-multiseed.json").read_text(encoding="utf-8"))
    countdown = json.loads(
        (HERE / "countdown" / "pass8-multiseed.json").read_text(encoding="utf-8")
    )
    mech = json.loads((HERE / "mechanism.json").read_text(encoding="utf-8"))
    mech_cd = json.loads((HERE / "countdown" / "mechanism.json").read_text(encoding="utf-8"))
    _placebo_comparison_figure(placebo)
    _passk_curve_figure(panel)
    _contrast_figure(panel, countdown)
    _mechanism_figure(mech, mech_cd)
    print(
        "wrote fig-placebo-comparison.png, fig-passk-curve.png, fig-passk-contrast.png, "
        "fig-mechanism.png"
    )


if __name__ == "__main__":
    main()
