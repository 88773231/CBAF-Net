"""Independently recompute all development metrics from retained predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.run_observable_evolution import CONDITIONS, POLICIES, decision, validate_settings
from src.evidence_repair import metrics
from src.provenance import sha256_file


def audit(study):
    summary = json.loads((study / "summary.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    configuration_fields = {
        key: protocol[key] for key in json.loads(
            (Path(__file__).resolve().parents[1] / "configs/observable_evolution.json").read_text())
    }
    validate_settings(configuration_fields)
    if summary["protocol_sha256"] != sha256_file(study / "protocol.json"):
        raise ValueError("observation protocol hash differs")
    records, hash_checks, profile_sets = [], 0, {}
    for policy in POLICIES:
        expected_samples = None
        for condition in CONDITIONS:
            folder = study / policy / condition
            record = json.loads((folder / "diagnostic.json").read_text())
            expected = next(item for item in summary["records"]
                            if (item["policy"], item["condition"]) == (policy, condition))
            if record != expected or record["policy"] != policy or record["condition"] != condition:
                raise ValueError("observation diagnostic identity differs")
            for name, key in (("selected_model.pt", "checkpoint_sha256"),
                              ("inner_selection_predictions.jsonl", "predictions_sha256")):
                if sha256_file(folder / name) != record[key]:
                    raise ValueError("observation evidence member hash differs")
                hash_checks += 1
            predictions = [json.loads(line) for line in
                           (folder / "inner_selection_predictions.jsonl").read_text().splitlines()]
            samples = [(item["sample_id"], item["base_profile_id"], item["label"]) for item in predictions]
            if not samples or len({item[0] for item in samples}) != len(samples):
                raise ValueError("empty or duplicate observation predictions")
            grouped = {}
            for _, profile, label in samples:
                grouped.setdefault(profile, []).append(label)
            if any(sorted(labels) != list(range(4)) for labels in grouped.values()):
                raise ValueError("class-incomplete observation selection profiles")
            if expected_samples is None:
                expected_samples = samples
            elif expected_samples != samples:
                raise ValueError("same-input control prediction identities differ")
            labels = np.asarray([item["label"] for item in predictions])
            probabilities = np.asarray([item["probabilities"] for item in predictions])
            if (probabilities.shape != (len(labels), 4) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
                raise ValueError("invalid observation probabilities")
            recomputed = metrics(labels, probabilities, 4)
            recorded = record["inner_selection"]["all_profiles"]
            for key, value in recomputed.items():
                if not np.allclose(value, recorded[key], atol=1e-10, rtol=1e-10):
                    raise ValueError("prediction-derived observation metric differs")
            if (record["original_validation_test_or_excluded_policy_forwards"]
                    or record["submission_allowed"]
                    or record["maximum_replay_probability_difference"] > 1e-6):
                raise ValueError("observation replay or access boundary differs")
            records.append(record)
        profile_sets[policy] = {profile for _, profile, _ in expected_samples}
    result = decision(records)
    if any(summary[key] != value for key, value in result.items()):
        raise ValueError("observation summary decision differs")
    if (summary["new_fits_completed"] != 15 or summary["checkpoint_replays_completed"] != 15
            or summary["gpu_used"] or summary["new_account_generation"]
            or summary["original_validation_or_test_access"]):
        raise ValueError("observation study budget or access boundary differs")
    return {
        "audited_cells": len(records), "member_hash_checks": hash_checks,
        "unique_overlapping_selection_profiles": len(set().union(*profile_sets.values())),
        "summary_sha256": sha256_file(study / "summary.json"),
        "prediction_metrics_recomputed": True, "independent_model_evaluation": False,
        **result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.study), indent=2))


if __name__ == "__main__":
    main()
