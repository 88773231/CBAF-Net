#!/usr/bin/env python3
"""Paired profile-cluster bootstrap for multi-seed method comparisons.

The manifest points to row-level prediction JSONL files. Predictions are
aligned by ``(base_profile_id, label)`` and every bootstrap replicate samples
whole profiles, preserving the paired comparison across methods and seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_f1", "mcc")


def metric_block(labels: np.ndarray, predictions: np.ndarray, classes: list[int]) -> dict:
    class_to_index = {value: index for index, value in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        if int(prediction) not in class_to_index:
            raise ValueError(f"prediction contains undeclared class {int(prediction)}")
        confusion[class_to_index[int(label)], class_to_index[int(prediction)]] += 1
    true_positive = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    denominators = 2.0 * true_positive + (predicted - true_positive) + (
        actual - true_positive
    )
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros_like(true_positive),
        where=denominators != 0,
    )
    sample_count = float(confusion.sum())
    correct = float(np.trace(confusion))
    mcc_numerator = correct * sample_count - float(np.dot(actual, predicted))
    mcc_denominator = np.sqrt(
        (sample_count**2 - float(np.dot(predicted, predicted)))
        * (sample_count**2 - float(np.dot(actual, actual)))
    )
    return {
        "accuracy": correct / sample_count,
        "macro_f1": float(per_class_f1.mean()),
        "mcc": float(mcc_numerator / mcc_denominator)
        if mcc_denominator > 0
        else 0.0,
    }


def load_prediction_map(path: Path, pred_key: str) -> dict[tuple[str, int], int]:
    rows: dict[tuple[str, int], int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("base_profile_id", "label", pred_key):
            if field not in row:
                raise ValueError(f"{path}:{line_number} is missing {field}")
        key = (str(row["base_profile_id"]), int(row["label"]))
        if not key[0]:
            raise ValueError(f"{path}:{line_number} has an empty base_profile_id")
        if key in rows:
            raise ValueError(f"{path} contains duplicate profile/label pair {key}")
        rows[key] = int(row[pred_key])
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def resolve_spec(manifest_dir: Path, spec: dict) -> tuple[Path, str]:
    if "path" not in spec:
        raise ValueError("each method specification requires path")
    path = Path(spec["path"])
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve(), str(spec.get("pred_key", "pred"))


def load_manifest(path: Path) -> tuple[dict, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    references = payload.get("reference_methods", [])
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("manifest.datasets must be a non-empty object")
    if not isinstance(references, list) or not references:
        raise ValueError("manifest.reference_methods must be a non-empty list")
    return datasets, [str(value) for value in references]


def aligned_dataset(manifest_path: Path, dataset_name: str, block: dict) -> dict:
    seeds = block.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        raise ValueError(f"{dataset_name}.seeds must be a non-empty object")

    seed_data = {}
    expected_methods: list[str] | None = None
    canonical_keys: list[tuple[str, int]] | None = None
    for seed, methods in sorted(seeds.items(), key=lambda item: str(item[0])):
        if not isinstance(methods, dict) or len(methods) < 2:
            raise ValueError(f"{dataset_name} seed {seed} requires at least two methods")
        method_names = list(methods)
        if expected_methods is None:
            expected_methods = method_names
        elif set(method_names) != set(expected_methods):
            raise ValueError(f"method set differs for {dataset_name} seed {seed}")

        loaded = {}
        for method, spec in methods.items():
            source, pred_key = resolve_spec(manifest_path.parent, spec)
            loaded[method] = load_prediction_map(source, pred_key)
        reference_keys = set(next(iter(loaded.values())))
        for method, values in loaded.items():
            if set(values) != reference_keys:
                missing = len(reference_keys - set(values))
                extra = len(set(values) - reference_keys)
                raise ValueError(
                    f"row alignment differs for {dataset_name} seed {seed} method {method}: "
                    f"missing={missing}, extra={extra}"
                )
        ordered_keys = sorted(reference_keys)
        if canonical_keys is None:
            canonical_keys = ordered_keys
        elif ordered_keys != canonical_keys:
            raise ValueError(
                f"profile/label rows differ across seeds for dataset {dataset_name}"
            )
        seed_data[str(seed)] = {
            method: np.asarray([values[key] for key in ordered_keys], dtype=np.int64)
            for method, values in loaded.items()
        }

    assert canonical_keys is not None and expected_methods is not None
    labels = np.asarray([label for _, label in canonical_keys], dtype=np.int64)
    classes = sorted(set(labels.tolist()))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (profile_id, _) in enumerate(canonical_keys):
        groups[profile_id].append(index)
    return {
        "labels": labels,
        "classes": classes,
        "groups": {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()},
        "methods": expected_methods,
        "seeds": seed_data,
    }


def interval(estimate: float, values: np.ndarray) -> dict:
    return {
        "estimate": float(estimate),
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def difference_interval(estimate: float, values: np.ndarray) -> dict:
    tail_probability = min(
        1.0,
        2.0
        * min(
            float(np.mean(values <= 0.0)),
            float(np.mean(values >= 0.0)),
        ),
    )
    return {
        **interval(estimate, values),
        "bootstrap_probability_positive": float(np.mean(values > 0.0)),
        "two_sided_bootstrap_tail_probability": tail_probability,
    }


def analyze_dataset(
    data: dict,
    references: list[str],
    replicates: int,
    seed: int,
) -> dict:
    labels = data["labels"]
    classes = data["classes"]
    methods = data["methods"]
    seed_names = list(data["seeds"])
    for reference in references:
        if reference not in methods:
            raise ValueError(f"reference method is absent from dataset: {reference}")

    observed_by_seed = {}
    for seed_name, predictions in data["seeds"].items():
        observed_by_seed[seed_name] = {
            method: metric_block(labels, predictions[method], classes)
            for method in methods
        }

    observed_aggregate = {
        method: {
            metric: float(
                np.mean(
                    [observed_by_seed[name][method][metric] for name in seed_names]
                )
            )
            for metric in METRICS
        }
        for method in methods
    }

    comparisons = [
        (method, reference)
        for reference in references
        for method in methods
        if method != reference
    ]
    group_names = sorted(data["groups"])
    rng = np.random.default_rng(seed)
    per_seed_values = {
        seed_name: {
            method: {metric: np.empty(replicates) for metric in METRICS}
            for method in methods
        }
        for seed_name in seed_names
    }
    aggregate_values = {
        method: {metric: np.empty(replicates) for metric in METRICS}
        for method in methods
    }
    comparison_values = {
        f"{method}_minus_{reference}": {
            metric: np.empty(replicates) for metric in METRICS
        }
        for method, reference in comparisons
    }

    for iteration in range(replicates):
        chosen = rng.integers(0, len(group_names), size=len(group_names))
        indices = np.concatenate(
            [data["groups"][group_names[position]] for position in chosen]
        )
        sampled_labels = labels[indices]
        for seed_name in seed_names:
            for method in methods:
                current = metric_block(
                    sampled_labels,
                    data["seeds"][seed_name][method][indices],
                    classes,
                )
                for metric in METRICS:
                    per_seed_values[seed_name][method][metric][iteration] = current[metric]
        for method in methods:
            for metric in METRICS:
                aggregate_values[method][metric][iteration] = np.mean(
                    [per_seed_values[name][method][metric][iteration] for name in seed_names]
                )
        for method, reference in comparisons:
            name = f"{method}_minus_{reference}"
            for metric in METRICS:
                comparison_values[name][metric][iteration] = (
                    aggregate_values[method][metric][iteration]
                    - aggregate_values[reference][metric][iteration]
                )

    per_seed = {}
    for seed_name in seed_names:
        per_seed[seed_name] = {
            method: {
                metric: interval(
                    observed_by_seed[seed_name][method][metric],
                    per_seed_values[seed_name][method][metric],
                )
                for metric in METRICS
            }
            for method in methods
        }
    aggregate = {
        method: {
            metric: interval(
                observed_aggregate[method][metric], aggregate_values[method][metric]
            )
            for metric in METRICS
        }
        for method in methods
    }
    aggregate_comparisons = {}
    for method, reference in comparisons:
        name = f"{method}_minus_{reference}"
        aggregate_comparisons[name] = {
            metric: difference_interval(
                observed_aggregate[method][metric]
                - observed_aggregate[reference][metric],
                comparison_values[name][metric],
            )
            for metric in METRICS
        }
    return {
        "row_count": int(len(labels)),
        "profile_count": int(len(group_names)),
        "classes": classes,
        "seeds": seed_names,
        "methods": methods,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "aggregate_comparisons": aggregate_comparisons,
    }


def run(manifest_path: Path, replicates: int, seed: int) -> dict:
    datasets, references = load_manifest(manifest_path)
    results = {}
    for offset, (dataset_name, block) in enumerate(sorted(datasets.items())):
        data = aligned_dataset(manifest_path, dataset_name, block)
        results[dataset_name] = analyze_dataset(
            data,
            references,
            replicates,
            seed + offset,
        )
    return {
        "schema_version": "paired-profile-cluster-bootstrap-v1",
        "manifest": manifest_path.as_posix(),
        "sampling_unit": "base_profile_id",
        "aggregate_estimand": "mean of fixed-seed metrics per synchronized profile replicate",
        "bootstrap_replicates": replicates,
        "base_seed": seed,
        "reference_methods": references,
        "datasets": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("--replicates must be at least 100")
    result = run(args.manifest, args.replicates, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
