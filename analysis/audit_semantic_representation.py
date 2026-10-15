"""Independently recompute saved selection metrics and evidence hashes."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.run_semantic_representation import CONDITIONS, POLICIES, decision, validate_protocol
from src.evidence_repair import metrics
from src.provenance import sha256_file


def audit(study):
    summary = json.loads((study / "summary.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/semantic_representation.json").read_text())
    validate_protocol({key: protocol[key] for key in settings})
    if summary["protocol_sha256"] != sha256_file(study / "protocol.json"):
        raise ValueError("representation protocol hash differs")
    records, checks = [], 0
    for policy in POLICIES:
        receipt = json.loads((study / policy / "feature_receipt.json").read_text())
        if (receipt["excluded_policy_encoded_transformed_or_forwarded"]
                or receipt["normalization_scope"] != "inner_fitting_only"
                or receipt["partition_sha256"] != protocol["prior_partition_hashes"][policy]):
            raise ValueError("representation feature access boundary differs")
        if sha256_file(study / policy / "known_observations.pt") != receipt["cache_sha256"]:
            raise ValueError("representation cache hash differs")
        checks += 1
        expected_samples, matched_fits = None, {}
        for condition in CONDITIONS:
            folder = study / policy / condition
            record = json.loads((folder / "diagnostic.json").read_text())
            expected = next(row for row in summary["records"]
                            if (row["policy"], row["condition"]) == (policy, condition))
            if record != expected:
                raise ValueError("representation cell record differs")
            for name, key in (("selected_model.pt", "checkpoint_sha256"),
                              ("inner_selection_predictions.jsonl", "predictions_sha256")):
                if sha256_file(folder / name) != record[key]:
                    raise ValueError("representation evidence member hash differs")
                checks += 1
            predictions = [json.loads(line) for line in
                           (folder / "inner_selection_predictions.jsonl").read_text().splitlines()]
            samples = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in predictions]
            grouped = {}
            for _, profile, label in samples:
                grouped.setdefault(profile, []).append(label)
            if (not samples or len({row[0] for row in samples}) != len(samples)
                    or any(sorted(labels) != list(range(4)) for labels in grouped.values())):
                raise ValueError("invalid representation selection identities")
            if expected_samples is None:
                expected_samples = samples
            elif expected_samples != samples:
                raise ValueError("matched representation selection identities differ")
            labels = np.asarray([row["label"] for row in predictions])
            probabilities = np.asarray([row["probabilities"] for row in predictions])
            if (probabilities.shape != (len(labels), 4) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
                raise ValueError("invalid representation probabilities")
            recomputed = metrics(labels, probabilities, 4)
            for key, value in recomputed.items():
                if not np.allclose(value, record["inner_selection"]["all_profiles"][key],
                                   atol=1e-10, rtol=1e-10):
                    raise ValueError("representation prediction metrics differ")
            if (record["maximum_replay_probability_difference"] > 1e-6
                    or record["original_validation_test_or_excluded_policy_forwards"]
                    or record["submission_allowed"]):
                raise ValueError("representation replay boundary differs")
            architecture = condition.split("_")[1]
            matched_fits.setdefault(architecture, []).append(record["fit"])
            records.append(record)
        for fits in matched_fits.values():
            if (len({fit["initialization_digest"] for fit in fits}) != 1
                    or len({fit["training_parameter_count"] for fit in fits}) != 1):
                raise ValueError("representation architecture pair is not matched")
    result = decision(records)
    if any(summary[key] != value for key, value in result.items()):
        raise ValueError("representation summary decision differs")
    if (summary["new_fits_completed"] != 12 or summary["checkpoint_replays_completed"] != 12
            or summary["gpu_used"] or summary["new_account_generation"]
            or summary["original_validation_or_test_access"]):
        raise ValueError("representation budget boundary differs")
    return {
        "audited_cells": len(records), "member_hash_checks": checks,
        "summary_sha256": sha256_file(study / "summary.json"),
        "prediction_metrics_recomputed": True, "independent_evaluation": False, **result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.study), indent=2))


if __name__ == "__main__":
    main()
