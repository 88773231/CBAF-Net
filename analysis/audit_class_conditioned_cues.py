"""Prediction/hash audit for the frozen class-conditioned development check."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.run_class_conditioned_cues import CONDITIONS, POLICIES, decision, settings, validate_settings
from src.evidence_repair import metrics
from src.provenance import sha256_file


def audit(study):
    summary = json.loads((study / "summary.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    validate_settings({key: protocol[key] for key in settings()})
    if summary["protocol_sha256"] != sha256_file(study / "protocol.json"):
        raise ValueError("class-conditioned protocol hash differs")
    records, profiles, checks = [], {}, 0
    for policy in POLICIES:
        expected_samples = None
        for condition in CONDITIONS:
            folder = study / policy / condition
            record = json.loads((folder / "diagnostic.json").read_text())
            expected = next(row for row in summary["records"]
                            if (row["policy"], row["condition"]) == (policy, condition))
            if record != expected:
                raise ValueError("class-conditioned cell record differs")
            for name, key in (("selected_model.pt", "checkpoint_sha256"),
                              ("inner_selection_predictions.jsonl", "predictions_sha256")):
                if sha256_file(folder / name) != record[key]:
                    raise ValueError("class-conditioned evidence member hash differs")
                checks += 1
            predictions = [json.loads(line) for line in
                           (folder / "inner_selection_predictions.jsonl").read_text().splitlines()]
            samples = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in predictions]
            grouped = {}
            for _, profile, label in samples:
                grouped.setdefault(profile, []).append(label)
            if (not samples or len(set(row[0] for row in samples)) != len(samples)
                    or any(sorted(labels) != list(range(4)) for labels in grouped.values())):
                raise ValueError("invalid class-conditioned selection identities")
            if expected_samples is None:
                expected_samples = samples
            elif expected_samples != samples:
                raise ValueError("class-conditioned matched selection identities differ")
            labels = np.asarray([row["label"] for row in predictions])
            probabilities = np.asarray([row["probabilities"] for row in predictions])
            if (probabilities.shape != (len(labels), 4) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
                raise ValueError("invalid class-conditioned probabilities")
            recomputed = metrics(labels, probabilities, 4)
            for key, value in recomputed.items():
                if not np.allclose(value, record["inner_selection"]["all_profiles"][key],
                                   atol=1e-10, rtol=1e-10):
                    raise ValueError("class-conditioned prediction metrics differ")
            if (record["maximum_replay_probability_difference"] > 1e-6
                    or record["original_validation_test_or_excluded_policy_forwards"]
                    or record["submission_allowed"]):
                raise ValueError("class-conditioned replay/access boundary differs")
            records.append(record)
        profiles[policy] = {row[1] for row in expected_samples}
    result = decision(records)
    if any(summary[key] != value for key, value in result.items()):
        raise ValueError("class-conditioned summary decision differs")
    if (summary["new_fits_completed"] != 12 or summary["checkpoint_replays_completed"] != 18
            or summary["cached_controls_replayed"] != 6
            or sum(bool(row["fit"]["new_fit"]) for row in records) != 12
            or summary["gpu_used"] or summary["new_account_generation"]
            or summary["original_validation_or_test_access"]):
        raise ValueError("class-conditioned budget/access boundary differs")
    return {
        "audited_cells": len(records), "member_hash_checks": checks,
        "unique_overlapping_selection_profiles": len(set().union(*profiles.values())),
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
