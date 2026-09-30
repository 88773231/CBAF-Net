#!/usr/bin/env python3
"""Create the four-panel CBAF-Net class-level confusion figure.

The displayed matrix for every task/model pair is computed as follows:

1. Build a test-set confusion matrix independently for seeds 42, 43, and 44.
2. Row-normalize each seed matrix by its true-class support.
3. Average the three normalized matrices.
4. Display the mean cells as percentages.

Example
-------
python analysis/plot_confusion.py \
  --predictions-root results/predictions \
  --output results/figures/cbaf_net_confusion_fcs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEEDS: Tuple[int, ...] = (42, 43, 44)

TASKS: Mapping[str, Mapping[str, Sequence[str]]] = {
    "Twibot22": {
        "labels": ("Human", "Traditional Bot", "LLM Bot"),
        "x_labels": ("Human", "Traditional\nBot", "LLM Bot"),
        "y_labels": ("Human", "Traditional\nBot", "LLM Bot"),
    },
    "Quadbot": {
        "labels": ("Human", "Traditional Bot", "LLM Bot", "Full-stack Agent"),
        # Compact x labels remain unambiguous in a two-column-width figure.
        # Full class names are retained on the y axis.
        "x_labels": ("Human", "Trad.\nBot", "LLM\nBot", "Full-stack\nAgent"),
        "y_labels": ("Human", "Traditional\nBot", "LLM Bot", "Full-stack\nAgent"),
    },
}

MODELS: Mapping[str, str] = {
    "BotDMM": "baseline_pred",
    "CBAF-Net": "pred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three-seed mean row-normalized confusion matrices for "
            "Twibot22 and Quadbot."
        )
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("results/predictions"),
        help=(
            "Directory containing Twibot22_seed{42,43,44} and "
            "Quadbot_seed{42,43,44} subdirectories. "
            "Default: results/predictions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/cbaf_net_confusion_fcs"),
        help=(
            "Output path stem. The script writes .png, .pdf, and .svg files. "
            "Default: results/figures/cbaf_net_confusion_fcs"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch. Default: 600",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    if y_true.shape != y_pred.shape:
        raise ValueError("Ground-truth and prediction arrays must have identical shapes")
    if np.any((y_true < 0) | (y_true >= class_count)):
        raise ValueError("Ground-truth labels fall outside the declared class range")
    if np.any((y_pred < 0) | (y_pred >= class_count)):
        raise ValueError("Predicted labels fall outside the declared class range")

    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    support = matrix.sum(axis=1, keepdims=True)
    if np.any(support == 0):
        missing = np.flatnonzero(support.ravel() == 0).tolist()
        raise ValueError(
            f"Cannot row-normalize because true classes {missing} have zero support"
        )
    return matrix.astype(np.float64) / support


def load_seed_matrix(
    predictions_root: Path,
    task: str,
    seed: int,
    prediction_field: str,
) -> Tuple[np.ndarray, np.ndarray]:
    class_count = len(TASKS[task]["labels"])
    path = predictions_root / f"{task}_seed{seed}" / "test_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    records = read_jsonl(path)
    missing_fields = [
        index
        for index, record in enumerate(records)
        if "label" not in record or prediction_field not in record
    ]
    if missing_fields:
        raise KeyError(
            f"{path} lacks label/{prediction_field} in rows {missing_fields[:5]}"
        )

    y_true = np.asarray([record["label"] for record in records], dtype=np.int64)
    y_pred = np.asarray([record[prediction_field] for record in records], dtype=np.int64)
    counts = confusion_counts(y_true, y_pred, class_count)
    return counts, row_normalize(counts)


def aggregate_matrices(
    predictions_root: Path,
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    aggregated: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for task in TASKS:
        for model, prediction_field in MODELS.items():
            counts_by_seed: List[np.ndarray] = []
            normalized_by_seed: List[np.ndarray] = []
            for seed in SEEDS:
                counts, normalized = load_seed_matrix(
                    predictions_root,
                    task,
                    seed,
                    prediction_field,
                )
                counts_by_seed.append(counts)
                normalized_by_seed.append(normalized)

            counts_stack = np.stack(counts_by_seed, axis=0)
            normalized_stack = np.stack(normalized_by_seed, axis=0)
            mean_normalized = normalized_stack.mean(axis=0)
            if not np.allclose(mean_normalized.sum(axis=1), 1.0):
                raise AssertionError(f"Rows do not sum to one for {task}/{model}")

            aggregated[(task, model)] = {
                "counts_by_seed": counts_stack,
                "normalized_by_seed": normalized_stack,
                "mean_normalized": mean_normalized,
            }
    return aggregated


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.8,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray) -> None:
    class_count = matrix.shape[0]
    font_size = 9.2 if class_count == 3 else 8.2
    for row in range(class_count):
        for column in range(class_count):
            value = float(matrix[row, column])
            ax.text(
                column,
                row,
                f"{value * 100:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "#172033",
                fontsize=font_size,
                fontweight="bold" if row == column else "normal",
            )


def draw_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    task: str,
    model: str,
    panel_letter: str,
) -> plt.Axes:
    x_labels = TASKS[task]["x_labels"]
    y_labels = TASKS[task]["y_labels"]
    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    annotate_matrix(ax, matrix)

    positions = np.arange(len(x_labels))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(x_labels, linespacing=0.95)
    ax.set_yticklabels(y_labels, linespacing=0.95)
    ax.tick_params(axis="x", which="major", length=0, pad=3)
    ax.tick_params(axis="y", which="major", length=0, pad=4)
    ax.set_title(f"({panel_letter}) {task} - {model}", pad=7)
    ax.set_xlabel("Predicted class", labelpad=4)
    ax.set_ylabel("True class", labelpad=5)

    # White internal boundaries keep cells legible without making the panel look tabular.
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#8190A2")
        spine.set_linewidth(0.7)
    return image


def save_figure(
    aggregated: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
    output_stem: Path,
    dpi: int,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    configure_style()

    # 7.12 inches matches a typical two-column text block. Keeping the source
    # at its final print width prevents labels that only work on an oversized
    # canvas from colliding after manuscript scaling.
    fig = plt.figure(figsize=(7.12, 5.72), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.115,
        right=0.895,
        bottom=0.105,
        top=0.855,
        wspace=0.42,
        hspace=0.50,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    colorbar_axis = fig.add_subplot(grid[:, 2])

    panels = (
        ("Twibot22", "BotDMM", "a"),
        ("Twibot22", "CBAF-Net", "b"),
        ("Quadbot", "BotDMM", "c"),
        ("Quadbot", "CBAF-Net", "d"),
    )
    image = None
    for ax, (task, model, letter) in zip(axes, panels):
        image = draw_panel(
            ax,
            aggregated[(task, model)]["mean_normalized"],
            task,
            model,
            letter,
        )

    if image is None:
        raise RuntimeError("No confusion panels were drawn")

    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=np.linspace(0, 1, 6))
    colorbar.set_label(
        "Mean row-normalized proportion",
        rotation=90,
        labelpad=7,
        fontsize=8.2,
    )
    colorbar.ax.set_yticklabels(
        [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)]
    )
    colorbar.outline.set_edgecolor("#8190A2")
    colorbar.outline.set_linewidth(0.7)

    fig.suptitle(
        "Class-level confusion comparison",
        fontsize=11.5,
        fontweight="semibold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "Mean of row-normalized test confusion matrices across seeds 42, 43, and 44",
        ha="center",
        va="center",
        fontsize=7.9,
        color="#4F5D6D",
    )

    for suffix, options in (
        (".png", {"dpi": dpi}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(
            output_stem.with_suffix(suffix),
            facecolor="white",
            **options,
        )
    plt.close(fig)


def print_audit_summary(
    aggregated: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
) -> None:
    for task in TASKS:
        for model in MODELS:
            result = aggregated[(task, model)]
            supports = result["counts_by_seed"].sum(axis=2)
            matrix = result["mean_normalized"] * 100
            print(f"{task} / {model}")
            print(f"  class support by seed: {supports.tolist()}")
            print("  three-seed mean row-normalized confusion (%):")
            print(np.array2string(matrix, precision=2, suppress_small=True))


def normalize_output_stem(path: Path) -> Path:
    if path.suffix.lower() in {".png", ".pdf", ".svg"}:
        return path.with_suffix("")
    return path


def main() -> None:
    args = parse_args()
    predictions_root = args.predictions_root.expanduser().resolve()
    output_stem = normalize_output_stem(args.output.expanduser())
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer")

    aggregated = aggregate_matrices(predictions_root)
    print_audit_summary(aggregated)
    save_figure(aggregated, output_stem, args.dpi)
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {output_stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
