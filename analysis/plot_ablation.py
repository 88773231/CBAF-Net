"""Create the publication ablation figure from an audited strict-run summary."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


VARIANTS = [
    "CBAF-Net",
    "Numerical-only BSE fusion",
    "Style-only BSE fusion",
    "BSE only",
]
SUMMARY_KEYS = [
    "CBAF-Net",
    "BSE-numerical-only fusion",
    "BSE-style-only fusion",
    "BSE-only",
]

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


def results_from_aggregates(aggregates: dict) -> dict[str, list[float]]:
    if not isinstance(aggregates, dict):
        raise ValueError("strict summary does not contain an aggregates object")

    results = {}
    for dataset in ("Twibot22", "Quadbot"):
        dataset_summary = aggregates.get(dataset)
        if not isinstance(dataset_summary, dict):
            raise ValueError(f"strict summary is missing {dataset} aggregates")
        values = []
        for key in SUMMARY_KEYS:
            record = dataset_summary.get(key)
            if not isinstance(record, dict) or record.get("status") != "available":
                raise ValueError(f"strict summary is missing available {dataset}/{key}")
            values.append(float(record["statistics"]["macro_f1"]["mean"]))
        label = f"{dataset} ({3 if dataset == 'Twibot22' else 4} classes)"
        results[label] = values
    return results


def load_results(summary_path: Path) -> dict[str, list[float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return results_from_aggregates(payload.get("aggregates"))


def draw_panel(
    ax: plt.Axes,
    title: str,
    values: list[float],
    show_labels: bool,
    x_limits: tuple[float, float],
) -> None:
    y_positions = list(range(len(VARIANTS)))[::-1]
    full_value = values[0]

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

    # Place labels toward the open side of the shared scale. The two datasets
    # occupy opposite halves, so this also keeps Quadbot labels clear of the
    # full-model reference line near the right edge.
    label_switch = (x_limits[0] + x_limits[1]) / 2.0
    for index, (y, value) in enumerate(zip(y_positions, values)):
        label_on_left = value >= label_switch
        ax.annotate(
            f"{value:.4f}",
            xy=(value, y),
            xytext=(-6 if label_on_left else 6, 0),
            textcoords="offset points",
            ha="right" if label_on_left else "left",
            va="center",
            fontsize=7.8,
            color=FULL_COLOR if index == 0 else TEXT,
            fontweight="bold" if index == 0 else "normal",
            zorder=5,
        )

    ax.set_title(title, color=TEXT, fontweight="bold", pad=7)
    ax.set_xlim(*x_limits)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
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


def build_figure(results: dict[str, list[float]]) -> plt.Figure:
    configure_matplotlib()
    all_values = [value for values in results.values() for value in values]
    lower = min(all_values)
    upper = max(all_values)
    padding = max(0.006, (upper - lower) * 0.22)
    x_limits = (max(0.0, lower - padding), min(1.0, upper + padding))

    fig, axes = plt.subplots(1, 2, figsize=(7.12, 3.36), sharex=True, sharey=False)
    for index, (ax, (title, values)) in enumerate(zip(axes, results.items())):
        draw_panel(ax, title, values, show_labels=index == 0, x_limits=x_limits)

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
        "Mean test Macro-F1 over seeds 42, 43, and 44",
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
    fig.supxlabel(
        "Macro-F1 (higher is better; common scale across panels)",
        y=0.035,
        fontsize=8.4,
        color=TEXT,
    )
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.19, top=0.77, wspace=0.17)
    return fig


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/strict_postprocessed/strict_rerun_summary.json"),
        help="Audited strict-rerun summary produced by postprocess_strict_rerun.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strict_postprocessed/figures/cbaf_net_ablation"),
        help="Output path without a file extension.",
    )
    args = parser.parse_args()
    save_figure(build_figure(load_results(args.summary)), args.output)
    print(f"Wrote {args.output}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
