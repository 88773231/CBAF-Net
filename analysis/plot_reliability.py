"""Build publication-ready reliability/confidence plots from frozen predictions.

The script reads the test ``predictions.jsonl`` artifacts produced for each seed,
computes per-seed accuracy and expected calibration error (ECE), and writes a
2x2 diagnostic figure comparing BotDMM, BSE-only, and fused CBAF-Net. The top
row contains reliability curves and the bottom row contains distributions of
maximum predicted confidence. Predictions are never modified.

Usage (from repository root)::

    python analysis/plot_reliability.py \
        --predictions-root results/predictions

Outputs are written to ``results/figures`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHODS = {
    "baseline_prob": "BotDMM",
    "bse_prob": "BSE",
    "fused_prob": "CBAF-Net (fused)",
}
COLORS = {
    "baseline_prob": "#2878B5",
    "bse_prob": "#D95F02",
    "fused_prob": "#1B9E77",
}
DATASETS = ("Twibot22", "Quadbot")
SEEDS = (42, 43, 44)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def _validate_rows(rows: Sequence[dict], path: Path) -> int:
    labels = []
    for row_number, row in enumerate(rows, start=1):
        if "label" not in row:
            raise ValueError(f"Missing label in {path}:{row_number}")
        labels.append(int(row["label"]))
        for key in METHODS:
            probs = np.asarray(row.get(key), dtype=float)
            if probs.ndim != 1 or probs.size == 0 or not np.all(np.isfinite(probs)):
                raise ValueError(f"Invalid {key} in {path}:{row_number}")
            if np.any(probs < -1e-7) or abs(float(probs.sum()) - 1.0) > 2e-3:
                raise ValueError(f"Non-normalized probabilities in {path}:{row_number}")
    n_classes = int(max(labels)) + 1
    if sorted(set(labels)) != list(range(n_classes)):
        raise ValueError(f"Labels are not contiguous in {path}: {sorted(set(labels))}")
    return n_classes


def _reliability(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> dict:
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include confidence==1 in the last bin; np.digitize otherwise returns n_bins.
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1], right=False), n_bins - 1)
    accuracy = np.full(n_bins, np.nan, dtype=float)
    mean_conf = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        mask = bin_ids == idx
        counts[idx] = int(mask.sum())
        if counts[idx]:
            accuracy[idx] = float(correct[mask].mean())
            mean_conf[idx] = float(confidence[mask].mean())
    ece = float(
        np.sum(
            (counts / max(1, len(labels)))
            * np.nan_to_num(np.abs(accuracy - mean_conf), nan=0.0)
        )
    )
    # Maximum calibration error is useful for the compact summary table.
    mce = float(np.nanmax(np.abs(accuracy - mean_conf))) if np.any(counts) else math.nan
    return {
        "accuracy": float(correct.mean()),
        "ece": ece,
        "mce": mce,
        "mean_confidence": float(confidence.mean()),
        "bin_accuracy": accuracy,
        "bin_confidence": mean_conf,
        "bin_counts": counts,
        "confidence": confidence,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def collect(root: Path, n_bins: int) -> Tuple[dict, dict]:
    """Return per-dataset/per-method metrics and per-seed curves."""
    curves: dict = defaultdict(lambda: defaultdict(list))
    summary: dict = defaultdict(lambda: defaultdict(dict))
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / f"{dataset}_seed{seed}" / "test_predictions.jsonl"
            rows = _read_jsonl(path)
            n_classes = _validate_rows(rows, path)
            labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
            for key in METHODS:
                probs = np.asarray([row[key] for row in rows], dtype=float)
                if probs.shape[1] != n_classes:
                    raise ValueError(
                        f"{path} has {probs.shape[1]} probabilities but {n_classes} labels"
                    )
                metric = _reliability(labels, probs, n_bins)
                curves[dataset][key].append(metric)
        for key in METHODS:
            metrics = curves[dataset][key]
            summary[dataset][key] = {
                "accuracy_mean": _mean_std([m["accuracy"] for m in metrics])[0],
                "accuracy_std": _mean_std([m["accuracy"] for m in metrics])[1],
                "ece_mean": _mean_std([m["ece"] for m in metrics])[0],
                "ece_std": _mean_std([m["ece"] for m in metrics])[1],
                "mce_mean": _mean_std([m["mce"] for m in metrics])[0],
                "mce_std": _mean_std([m["mce"] for m in metrics])[1],
                "mean_confidence_mean": _mean_std(
                    [m["mean_confidence"] for m in metrics]
                )[0],
                "n_test_per_seed": int(sum(m["bin_counts"].sum() for m in metrics) / len(metrics)),
            }
    return curves, summary


def _pooled_bins(metrics: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool bin sufficient statistics without treating seeds as new training data."""
    counts = np.sum([m["bin_counts"] for m in metrics], axis=0).astype(int)
    weighted_accuracy = np.zeros_like(counts, dtype=float)
    weighted_confidence = np.zeros_like(counts, dtype=float)
    for metric in metrics:
        valid = metric["bin_counts"] > 0
        weighted_accuracy[valid] += (
            metric["bin_accuracy"][valid] * metric["bin_counts"][valid]
        )
        weighted_confidence[valid] += (
            metric["bin_confidence"][valid] * metric["bin_counts"][valid]
        )
    pooled_accuracy = np.divide(
        weighted_accuracy,
        counts,
        out=np.full_like(weighted_accuracy, np.nan),
        where=counts > 0,
    )
    pooled_confidence = np.divide(
        weighted_confidence,
        counts,
        out=np.full_like(weighted_confidence, np.nan),
        where=counts > 0,
    )
    return pooled_confidence, pooled_accuracy, counts


def _plot_reliability(ax: mpl.axes.Axes, metrics: Sequence[dict], key: str) -> None:
    confidence, accuracy, counts = _pooled_bins(metrics)
    valid = counts > 0
    confidence = confidence[valid]
    accuracy = accuracy[valid]
    counts = counts[valid]
    total = int(counts.sum())
    support = counts / max(1, total)
    # Small, pale points disclose sparsely populated bins without allowing them
    # to visually dominate the well-supported part of the calibration curve.
    sizes = 18.0 + 82.0 * np.sqrt(support / max(float(support.max()), 1e-12))
    rgba = np.tile(np.asarray(mpl.colors.to_rgba(COLORS[key])), (len(counts), 1))
    rgba[:, 3] = np.where(support < 0.005, 0.28, np.where(support < 0.02, 0.58, 0.95))
    supported = support >= 0.005
    ax.plot(
        confidence[supported],
        accuracy[supported],
        lw=1.65,
        color=COLORS[key],
        alpha=0.78,
        zorder=2,
    )
    ax.scatter(
        confidence,
        accuracy,
        s=sizes,
        facecolors=rgba,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
        label=METHODS[key],
    )


def _plot_confidence_distribution(
    ax: mpl.axes.Axes, metrics: Sequence[dict], key: str
) -> None:
    confidence = np.concatenate([m["confidence"] for m in metrics])
    bins = np.linspace(0.25, 1.0, 21)
    ax.hist(
        confidence,
        bins=bins,
        weights=np.full(confidence.shape, 1.0 / len(confidence)),
        histtype="step",
        linewidth=1.85,
        color=COLORS[key],
        label=METHODS[key],
    )


def build_figure(curves: dict, summary: dict, output: Path, n_bins: int) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.3,
            "figure.dpi": 160,
            "savefig.dpi": 500,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.9),
        sharex="col",
        gridspec_kw={"height_ratios": [1.55, 0.78]},
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.89, wspace=0.20, hspace=0.32)
    for idx, dataset in enumerate(DATASETS):
        ax = axes[0, idx]
        ax.plot(
            [0.25, 1],
            [0.25, 1],
            ls="--",
            lw=1.0,
            color="#666666",
            alpha=0.85,
            label="Perfect calibration",
        )
        for key in METHODS:
            _plot_reliability(ax, curves[dataset][key], key)
        ax.set_title(f"({chr(97 + idx)}) {dataset}: reliability", fontweight="bold", pad=6)
        # Keep markers at probability/accuracy boundaries fully inside the
        # axes. The labeled ticks still show the meaningful [0, 1] range.
        ax.set_xlim(0.245, 1.015)
        ax.set_ylim(-0.015, 1.015)
        ax.set_xticks([0.25, 0.40, 0.55, 0.70, 0.85, 1.0])
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.grid(axis="both", color="#D9D9D9", lw=0.55, alpha=0.65)
        if idx == 0:
            ax.set_ylabel("Empirical accuracy")
        ece_parts = []
        for key in METHODS:
            s = summary[dataset][key]
            short_name = METHODS[key].replace(" (fused)", "")
            ece_parts.append(f"{short_name} {s['ece_mean']:.3f}")
        ax.text(
            0.025,
            0.955,
            "ECE: " + " | ".join(ece_parts),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.8,
            bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
        )

        hist_ax = axes[1, idx]
        for key in METHODS:
            _plot_confidence_distribution(hist_ax, curves[dataset][key], key)
        hist_ax.set_title(f"({chr(99 + idx)}) {dataset}: confidence distribution", fontweight="bold", pad=5)
        # The final histogram edge is exactly 1.0; a small right pad prevents
        # the step line from being clipped by the right spine.
        hist_ax.set_xlim(0.245, 1.015)
        hist_ax.set_xticks([0.25, 0.40, 0.55, 0.70, 0.85, 1.0])
        hist_ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.65)
        hist_ax.set_xlabel("Maximum predicted probability")
        if idx == 0:
            hist_ax.set_ylabel("Fraction of predictions")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[key],
            marker="o",
            markersize=5.5,
            linewidth=1.7,
            label=METHODS[key],
        )
        for key in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="Perfect calibration",
        )
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.5,
    )
    fig.text(
        0.5,
        0.018,
        "Reliability curves pool three test runs; marker size and opacity encode bin support. "
        "ECE values are means of seed-wise 10-bin estimates.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(summary: dict, output: Path, n_bins: int) -> None:
    payload = {
        "description": "Test-set reliability summary from frozen predictions; values are mean across seeds 42/43/44.",
        "n_bins": n_bins,
        "datasets": summary,
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "CBAF-Net confidence calibration summary",
        "Values are mean +/- sample SD over seeds 42, 43, and 44; ECE uses 10 equal-width bins.",
        "",
    ]
    for dataset in DATASETS:
        lines.append(dataset)
        for key in METHODS:
            s = summary[dataset][key]
            lines.append(
                f"  {METHODS[key]}: accuracy={s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}; "
                f"ECE={s['ece_mean']:.4f} +/- {s['ece_std']:.4f}; "
                f"MCE={s['mce_mean']:.4f} +/- {s['mce_std']:.4f}; "
                f"mean_confidence={s['mean_confidence_mean']:.4f}"
            )
        lines.append("")
    output.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help="Directory containing Twibot22_seed*/ and Quadbot_seed*/ prediction folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/cbaf_net_reliability_confidence"),
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if args.bins < 3:
        raise ValueError("--bins must be at least 3")
    curves, summary = collect(args.predictions_root, args.bins)
    build_figure(curves, summary, args.output, args.bins)
    write_summary(summary, args.output, args.bins)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.svg')}")
    print(f"Wrote {args.output.with_suffix('.json')}")
    print(f"Wrote {args.output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
