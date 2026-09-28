#!/usr/bin/env python3
"""Compute profile-cluster bootstrap intervals from prediction JSONL files.

Each ``base_profile_id`` is sampled as a unit, so paired counterfactual
trajectories from one source profile never become independent bootstrap units.
The input is intentionally plain JSONL and can be produced by the formal
CBAF-Net evaluator without loading a model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("base_profile_id", "label", "pred"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number} is missing {field}")
            rows.append(row)
    if not rows:
        raise ValueError(f"prediction file is empty: {path}")
    return rows


def metrics(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        # Keep the class denominator fixed across bootstrap replicates.  A
        # resample can omit a class by chance, but that must not silently
        # change the meaning of Macro-F1.
        "macro_f1": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def bootstrap(rows: list[dict], reps: int, seed: int) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    seen_pairs: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        base_id = str(row["base_profile_id"])
        label = int(row["label"])
        pair = (base_id, label)
        if pair in seen_pairs:
            raise ValueError(f"duplicate base_profile_id/label pair: {pair}")
        seen_pairs.add(pair)
        groups[base_id].append(index)
    group_names = sorted(groups)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    pred = np.asarray([int(row["pred"]) for row in rows], dtype=np.int64)
    labels = sorted(set(int(value) for value in y.tolist()))
    if not labels:
        raise ValueError("prediction file contains no labels")
    observed = metrics(y, pred, labels)
    rng = np.random.default_rng(seed)
    sampled = {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}
    values = {name: np.empty(reps, dtype=np.float64) for name in observed}
    for iteration in range(reps):
        chosen = rng.integers(0, len(group_names), size=len(group_names))
        indices = np.concatenate([sampled[group_names[index]] for index in chosen])
        current = metrics(y[indices], pred[indices], labels)
        for name in values:
            values[name][iteration] = current[name]
    intervals = {
        name: {
            "estimate": observed[name],
            "lower_95": float(np.percentile(array, 2.5)),
            "upper_95": float(np.percentile(array, 97.5)),
        }
        for name, array in values.items()
    }
    return {
        "trajectory_count": len(rows),
        "profile_count": len(group_names),
        "records_per_profile": dict(sorted((str(size), count) for size, count in Counter(map(len, groups.values())).items())),
        "label_values": labels,
        "bootstrap_replicates": reps,
        "seed": seed,
        "metrics": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.replicates < 100:
        raise SystemExit("--replicates must be at least 100")
    result = {
        "schema_version": "quadbot-v3-profile-cluster-bootstrap-1",
        "input": str(args.predictions),
        **bootstrap(load_rows(args.predictions), args.replicates, args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
