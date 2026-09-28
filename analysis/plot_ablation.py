"""Create the publication figure for the seed-42 CBAF-Net ablation diagnostic."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


VARIANTS = ["CBAF-Net", "Numerical only", "Style only", "BSE only"]
RESULTS = {
    "Twibot22 (3 classes)": [0.7246, 0.6983, 0.7255, 0.7250],
    "Quadbot (4 classes)": [0.7919, 0.7693, 0.7862, 0.7900],
}

# The common scale is deliberately shared across panels. This keeps the visual
# comparison honest while still making the small fixed-seed differences legible.
X_MIN = 0.690
X_MAX = 0.805
X_TICKS = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]

FULL_COLOR = "#1F5A94"
FULL_EDGE = "#163F69"
ABLATION_EDGE = "#4B5563"
ABLATION_FILL = "#FFFFFF"
CONNECTOR = "#AEB7C2"
GRID = "#DCE1E7"
TEXT = "#1F2937"
MUTED = "#5F6B78"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.3,
            "axes.titlesize": 10.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_panel(ax: plt.Axes, title: str, values: list[float], show_labels: bool) -> None:
    y_positions = list(range(len(VARIANTS)))[::-1]
    full_value = values[0]

    # Subtle alternating bands improve row tracking without turning the chart
    # into a table or introducing decorative framing.
    for row, y in enumerate(y_positions):
        if row % 2 == 0:
            ax.axhspan(y - 0.43, y + 0.43, color="#F7F8FA", zorder=0)

    ax.axvline(
        full_value,
        color=FULL_COLOR,
        linewidth=1.05,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.62,
        zorder=1,
    )

    # Each ablated result is linked to the full-model reference. This directly
    # shows the direction and magnitude of the diagnostic change.
    for y, value in zip(y_positions[1:], values[1:]):
        ax.plot(
            [min(value, full_value), max(value, full_value)],
            [y, y],
            color=CONNECTOR,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=2,
        )

    ax.scatter(
        [full_value],
        [y_positions[0]],
        s=58,
        marker="o",
        facecolor=FULL_COLOR,
        edgecolor=FULL_EDGE,
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        values[1:],
        y_positions[1:],
        s=47,
        marker="o",
        facecolor=ABLATION_FILL,
        edgecolor=ABLATION_EDGE,
        linewidth=1.25,
        zorder=4,
    )

    for idx, (y, value) in enumerate(zip(y_positions, values)):
        color = FULL_COLOR if idx == 0 else TEXT
        weight = "bold" if idx == 0 else "normal"
        ax.annotate(
            f"{value:.4f}",
            xy=(value, y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.8,
            color=color,
            fontweight=weight,
            zorder=5,
        )

    ax.set_title(title, color=TEXT, fontweight="bold", pad=7)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(-0.55, len(VARIANTS) - 0.45)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(VARIANTS)
    ax.tick_params(axis="y", length=0, pad=7, labelleft=show_labels)
    ax.tick_params(axis="x", colors=MUTED, length=3, width=0.7)
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9AA3AD")


def build_figure() -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.36), sharex=True, sharey=False)

    for index, (ax, (title, values)) in enumerate(zip(axes, RESULTS.items())):
        draw_panel(ax, title, values, show_labels=index == 0)

    fig.suptitle(
        "Behavior-statistics view ablation",
        x=0.5,
        y=0.985,
        fontsize=12.0,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.925,
        "Seed 42 diagnostic | test-set Macro-F1 | descriptive point estimates",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=FULL_COLOR,
            markeredgecolor=FULL_EDGE,
            markeredgewidth=0.8,
            markersize=6.5,
            label="Full model",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ABLATION_FILL,
            markeredgecolor=ABLATION_EDGE,
            markeredgewidth=1.2,
            markersize=6.2,
            label="Ablated variant",
        ),
        Line2D(
            [0],
            [0],
            color=FULL_COLOR,
            linewidth=1.05,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.62,
            label="Full-model reference",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.882),
        ncol=3,
        frameon=False,
        fontsize=7.7,
        handlelength=2.0,
        columnspacing=1.5,
        handletextpad=0.55,
    )

    fig.supxlabel("Macro-F1 (higher is better; common scale across panels)", y=0.035, fontsize=8.4, color=TEXT)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.19, top=0.77, wspace=0.17)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/cbaf_net_ablation_fcs"),
        help="Output path without a file extension.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(args.output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
