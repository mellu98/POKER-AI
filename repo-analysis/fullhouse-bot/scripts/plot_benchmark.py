"""Regenerate docs/assets/benchmark.png.

The per-line numbers are the best standard-pool checkpoints with 95%
normal-approximation CIs (1.96 * stderr over per-match deltas, matching
tooling/harness/benchmarks.py) from the local harness (the same figures quoted in README.md and
docs/research/experiments.md). The raw sims are untracked, so the aggregated
results are inlined here to keep the figure reproducible from one command:

    python scripts/plot_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "benchmark.png"

SHIPPED = "#2E7D32"
PARKED = "#E8920A"
REFUTED = "#9E9E9E"

# Top-to-bottom order, strongest standard-pool line first.
ROWS = [
    (
        "v23  side-pot representation",
        23.73,
        20.88,
        26.57,
        "56f · 5a",
        False,
        PARKED,
        None,
    ),
    (
        "v21  multiway-equity feature",
        22.66,
        19.88,
        25.45,
        "52f · 5a",
        False,
        PARKED,
        None,
    ),
    ("v17  baseline policy", 22.41, 19.61, 25.21, "51f · 5a", True, SHIPPED, 19.89),
    ("v24  side-pot + 8-action", 12.49, 9.84, 15.14, "56f · 8a", False, REFUTED, None),
    ("v18  strength-percentile", 12.39, 9.73, 15.04, "51f · 5a", True, REFUTED, None),
    ("v22  8-action menu", 10.48, 7.78, 13.17, "51f · 8a", False, REFUTED, None),
]

CAPTION = (
    "Each line changed one variable. Promotion needed a CI-clear gain, a non-negative held-out\n"
    "pool, and a match to the 51-feature / 5-action runtime contract. That last constraint\n"
    "kept the two strongest local lines (v23, v21) off the submission."
)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=150)

    n = len(ROWS)
    ys = list(range(n))[::-1]  # row 0 at the top

    shipped_value = next(r[1] for r in ROWS if r[6] == SHIPPED)
    ax.axvline(shipped_value, color=SHIPPED, ls="--", lw=1.2, alpha=0.5, zorder=0)

    for y, (label, val, lo, hi, shape, serves, color, heldout) in zip(ys, ROWS):
        if serves:
            ax.axhspan(y - 0.42, y + 0.42, color=SHIPPED, alpha=0.07, zorder=0)
        ax.errorbar(
            val,
            y,
            xerr=[[val - lo], [hi - val]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.4,
            capsize=4,
            capthick=2.4,
            markersize=10,
            zorder=3,
        )
        weight = "bold" if color == SHIPPED else "normal"
        ax.text(
            hi + 0.7,
            y,
            f"+{val:.2f}",
            va="center",
            ha="left",
            color=color,
            fontsize=12,
            fontweight=weight,
        )
        ax.text(38.5, y, shape, va="center", ha="right", color="#555555", fontsize=11)
        mark, mcolor = ("✓", SHIPPED) if serves else ("✗", "#B22222")
        ax.text(41.3, y, mark, va="center", ha="center", color=mcolor, fontsize=14)
        if heldout is not None:
            ax.plot(
                heldout,
                y,
                "o",
                mfc="white",
                mec=SHIPPED,
                mew=2.0,
                markersize=10,
                zorder=4,
            )
            ax.text(
                heldout - 0.6,
                y + 0.34,
                f"held-out +{heldout:.2f}",
                va="bottom",
                ha="right",
                color=SHIPPED,
                fontsize=10,
                style="italic",
            )

    ax.text(
        38.5,
        n - 0.35,
        "runtime\nshape",
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=11,
        fontweight="bold",
        linespacing=0.95,
    )
    ax.text(
        41.3,
        n - 0.35,
        "serves\nas-is",
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=11,
        fontweight="bold",
        linespacing=0.95,
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=12)
    ax.set_ylim(-0.7, n - 0.1)
    ax.set_xlim(0, 43)
    ax.set_xticks(range(0, 40, 5))
    ax.set_xlabel(
        "best checkpoint vs the standard opponent pool   (bb/100, 95% normal-approx CI)",
        fontsize=12,
    )
    ax.grid(axis="x", color="#DDDDDD", lw=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_title(
        CAPTION, fontsize=12, loc="left", color="#222222", pad=16, linespacing=1.3
    )

    legend = [
        Patch(color=SHIPPED, label="shipped (promotion-safe)"),
        Patch(color=PARKED, label="parked (stronger, but a retrain to serve)"),
        Patch(color=REFUTED, label="refuted (no gain over baseline)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            mfc="white",
            mec=SHIPPED,
            mew=2.0,
            markersize=10,
            label="held-out result (generalization gap)",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper left",
        frameon=True,
        fontsize=11,
        borderpad=0.8,
        labelspacing=0.6,
    )

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
