"""Assemble development results without selecting a winner on test scores."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_summary(summary, methods):
    if summary.get("test_state") != "NOT_OPENED":
        raise ValueError("candidate results do not declare a locked test")
    runs = summary["runs"]
    expected = {(method, seed) for method in methods for seed in (42, 43, 44)}
    actual = {(run["method"], run["seed"]) for run in runs}
    if actual != expected or len(runs) != len(expected):
        raise ValueError("missing or duplicated development run")
    if any(run.get("test_state") != "NOT_OPENED" for run in runs):
        raise ValueError("a run declares test use")
    result = []
    for method in methods:
        records = [run for run in runs if run["method"] == method]
        expected_scores = {
            key: {
                "mean": float(np.mean([row["validation"][key] for row in records])),
                "seed_sd": float(np.std([row["validation"][key] for row in records], ddof=1)),
            }
            for key in ("macro_f1", "accuracy", "mcc")
        }
        for key, expected_score in expected_scores.items():
            for statistic, value in expected_score.items():
                if not np.isclose(value, summary["aggregates"][method][key][statistic], atol=1e-12):
                    raise ValueError("aggregate differs from its individual run records")
        result.append({"method": method, **expected_scores})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    methods = load(args.protocol)["methods"]
    result = {"scope": "exploratory_validation_only", "test_used": False, "datasets": {}}
    markdown = [
        "# Development Evidence Repair Results", "",
        "These are validation-selection results, not confirmatory test results.",
        "Timing is derived from the same source records. HGB uses seven global",
        "statistics; temporal/MLP models also use seven per-step statistics;",
        "pooled comparators summarize those steps. Old `v1.1.0` results are unchanged.", "",
    ]
    for dataset in ("Twibot22", "Quadbot"):
        summary = load(args.root / f"{dataset}_timing_validation.json")
        table = audit_summary(summary, methods)
        adapter = load(args.root / f"{dataset}_timing_botsscl.json")
        if adapter.get("test_paths_opened") is not False or len(adapter["runs"]) != 3:
            raise ValueError("incomplete or unlocked contrastive adapter")
        table.append({
            "method": "BotSSCL-inspired adapter (not faithful reproduction)",
            **adapter["aggregates"],
        })
        collision = load(args.root / f"{dataset}_input_collision_audit.json")
        diagnostic = load(args.root / f"{dataset}_shortcut_audit.json")
        result["datasets"][dataset] = {
            "validation": table, "input_collision_audit": collision,
            "original_input_shortcut_audit": diagnostic,
        }
        markdown.extend([
            f"## {dataset}", "",
            "| Method | Validation Macro-F1 mean +/- seed SD | Accuracy | MCC |",
            "|---|---:|---:|---:|",
        ])
        for row in table:
            markdown.append(
                f"| {row['method']} | {row['macro_f1']['mean']:.4f} +/- "
                f"{row['macro_f1']['seed_sd']:.4f} | {row['accuracy']['mean']:.4f} "
                f"| {row['mcc']['mean']:.4f} |"
            )
        markdown.append("")
        for split in ("train", "val"):
            counts = collision["splits"][split]
            markdown.append(
                f"- {split}: {counts['identical_non_graph_inputs_and_selected_neighbors']} "
                "human/traditional pairs had identical old direct inputs and selected neighbors; "
                f"{counts['remaining_non_graph_collisions_with_relative_timing']} "
                "direct-input collisions remain after relative timing."
            )
        markdown.append("")
    markdown.extend([
        "## Interpretation Gates", "",
        "- Report every comparator and removal result, even when a simpler model wins.",
        "- Validation selection makes these scores optimistic; do not relabel them as test scores.",
        "- A gain after adding timing is not an architecture gain unless the comparator receives timing too.",
        "- No cross-generator test is possible in this one-generator archive.",
        "- No broad SOTA, real-platform robustness, or module-necessity claim follows automatically.",
        "- Any final test protocol must disclose prior repeated access to the existing held-out split.", "",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.with_suffix(".md").exists():
        raise FileExistsError("refusing to overwrite the evidence summary")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
