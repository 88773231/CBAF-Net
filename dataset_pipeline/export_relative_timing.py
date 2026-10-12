"""Add label-free relative timing sidecars without changing any account records."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.evidence_repair import write_new_json
from src.provenance import sha256_file


FEATURE_NAMES = (
    "log_span", "log_mean_gap", "log_gap_sd", "gap_cv", "burstiness",
    "zero_gap_fraction", "adjacent_gap_variation",
)


def parse_time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("each post requires a valid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps require an explicit timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def gap_statistics(times):
    if not times:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    ordered = np.sort(np.asarray(times, dtype=np.float64))
    # Round gaps, not absolute epoch times. Sub-second serialization precision
    # is deliberately discarded so it cannot become a generator fingerprint.
    gaps = np.round(np.diff(ordered))
    span = float(np.round(ordered[-1] - ordered[0]))
    mean = float(gaps.mean()) if len(gaps) else 0.0
    std = float(gaps.std()) if len(gaps) else 0.0
    cv = std / max(mean, 1.0)
    burstiness = (std - mean) / max(std + mean, 1.0)
    zero_fraction = float((gaps == 0).mean()) if len(gaps) else 0.0
    adjacent = float(np.abs(np.diff(gaps)).mean()) / max(mean, 1.0) if len(gaps) > 1 else 0.0
    return np.asarray([
        np.log1p(span), np.log1p(mean), np.log1p(std), cv,
        burstiness, zero_fraction, adjacent,
    ], dtype=np.float32)


def relative_timing(row, steps=5):
    if steps < 1:
        raise ValueError("steps must be positive")
    groups = [[] for _ in range(steps)]
    all_times = []
    for post in row.get("posts") or []:
        value = parse_time(post.get("created_at"))
        step = int(post["time_step"])
        if not 1 <= step <= steps:
            raise ValueError("post step is outside the fixed timeline")
        groups[step - 1].append(value)
        all_times.append(value)
    if not all_times or any(not values for values in groups):
        raise ValueError("every timeline step requires at least one timestamp")
    return gap_statistics(all_times), np.stack([gap_statistics(values) for values in groups])


def export(source, features, output, dataset):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite relative-timing sidecars")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "dataset": dataset, "version": "relative-rounded-gap-v1",
        "feature_names": list(FEATURE_NAMES), "steps": 5,
        "absolute_epoch_or_calendar_features": False,
        "subsecond_precision_features": False, "test_paths_opened": False,
        "source_files": {}, "sidecar_files": {},
        "exporter_sha256": sha256_file(Path(__file__)),
    }
    for split in ("train", "val"):
        view = source / f"{split}.jsonl"
        metadata = features / f"{split}_metadata.jsonl"
        with view.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        with metadata.open(encoding="utf-8") as stream:
            identities = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != len(identities):
            raise ValueError("timing source and tensor metadata rows differ")
        global_values, step_values = [], []
        for row, identity in zip(rows, identities):
            if any(row[key] != identity[key] for key in ("sample_id", "base_profile_id", "label")):
                raise ValueError("timing sidecar identity/order differs from original tensors")
            global_value, step_value = relative_timing(row)
            global_values.append(global_value)
            step_values.append(step_value)
        for name, values in (("global", global_values), ("steps", step_values)):
            path = output / f"{split}_{name}.pt"
            torch.save(torch.from_numpy(np.stack(values)), path)
            receipt["sidecar_files"][path.name] = sha256_file(path)
        copied_metadata = output / f"{split}_metadata.jsonl"
        with copied_metadata.open("x", encoding="utf-8") as stream:
            for identity in identities:
                stream.write(json.dumps(identity) + "\n")
        receipt["sidecar_files"][copied_metadata.name] = sha256_file(copied_metadata)
        receipt["source_files"][view.name] = sha256_file(view)
    write_new_json(output / "export_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.features, args.output, args.dataset), indent=2))


if __name__ == "__main__":
    main()
