"""Recompute known-policy results from receipts without research tensors."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch

from analysis.run_compact_correspondence_stability import (
    CONDITIONS, SEEDS, decision, source_hashes, validate_settings,
)
from analysis.run_factor_training_screen import POLICIES
from src.compact_correspondence import CONDITIONS as COMPACT
from src.evidence_repair import metrics, write_new_json
from src.policy_robust import selection_key
from src.provenance import sha256_file


def audit(archive_path, summary_path, receipt_path, settings_path=None):
    receipt = json.loads(receipt_path.read_text())
    if (receipt["phase"] != "complete" or receipt["completed_new_fits"] != 30
            or receipt["reused_reference_checkpoints"] != 6 or receipt["checkpoint_replays"] != 36
            or receipt["gpu_used"] or receipt["challenge_forward_or_scoring"]
            or receipt["submission_allowed"] or not receipt["reference_files_unchanged"]
            or receipt["summary_sha256"] != sha256_file(summary_path)
            or receipt["evidence_archive_sha256"] != sha256_file(archive_path)):
        raise ValueError("stability completion receipt or evidence hashes differ")
    verified = []
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith(("/", "\\")) or "\\" in name
            or ".." in PurePosixPath(name).parts or "challenge_predictions" in name
            or PurePosixPath(name).name.startswith(("val_", "test_"))
            for name in names
        ):
            raise ValueError("unexpected or duplicate stability evidence member")
        if archive.read("summary.json") != summary_path.read_bytes():
            raise ValueError("archive stability summary differs")
        summary = json.loads(archive.read("summary.json"))
        if (summary["scope"] != "fixed_partition_three_training_seed_known_policy_stability"
                or summary["dataset"] != "Quadbot" or summary["new_fits"] != 30
                or summary["reused_reference_checkpoints"] != 6 or summary["checkpoint_replays"] != 36
                or not summary["reference_files_unchanged"] or summary["challenge_forward_or_scoring"]
                or summary["independent_confirmation"] or summary["architecture_novelty_established"]
                or summary["submission_allowed"]):
            raise ValueError("stability summary scope differs")
        settings_path = settings_path or (Path(__file__).parents[1] /
                                         "configs/compact_correspondence_stability.json")
        settings = json.loads(settings_path.read_text())
        validate_settings(settings)
        protocol_bytes = archive.read("protocol.json")
        protocol = json.loads(protocol_bytes)
        if ({key: protocol[key] for key in settings} != settings
                or protocol["source_hashes"] != source_hashes()
                or protocol["excluded_row_feature_transforms_or_model_forwards"]
                or protocol["exact_capacity_or_compute_matching_claimed"]
                or protocol["cached_checkpoint_challenge_fields_used_for_model_selection"]):
            raise ValueError("stability source/protocol boundary differs")
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        records = {(row["seed"], row["policy"], row["condition"]): row for row in summary["records"]}
        decision(summary["records"])
        predictions, normalizations, identities, profile_folds = {}, {}, {}, {}
        for policy in POLICIES:
            normalizations[policy] = json.loads(archive.read(policy + "/normalization.json"))
            for seed in SEEDS:
                for condition in CONDITIONS:
                    prefix = f"{policy}/seed{seed}/{condition}"
                    rows = [json.loads(line) for line in archive.read(
                        prefix + "/inner_selection_predictions.jsonl",
                    ).decode().splitlines()]
                    predictions[(seed, policy, condition)] = rows
                    ordered = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows]
                    if (not rows or len({row[0] for row in ordered}) != len(rows)
                            or len(rows) != len(normalizations[policy]["selection_indices"])):
                        raise ValueError("stability selection identities or counts differ")
                    reference_order = [(row["sample_id"], row["base_profile_id"], row["label"])
                                       for row in predictions[(42, policy, COMPACT[0])]]
                    if ordered != reference_order:
                        raise ValueError("fixed-partition paired selection order differs")
                    groups = {}
                    for sample, profile, label in ordered:
                        if sample in identities and identities[sample] != (profile, label):
                            raise ValueError("cross-fold stability identity differs")
                        identities[sample] = profile, label
                        groups.setdefault(profile, []).append(label)
                        profile_folds.setdefault(profile, set()).add(policy)
                    if any(sorted(labels) != list(range(4)) for labels in groups.values()):
                        raise ValueError("stability selection profiles are not class-complete")
        if any(len(folds) != 2 for folds in profile_folds.values()):
            raise ValueError("fixed selection profile fold membership differs")
        profile_policies = {
            profile: next(iter(set(POLICIES) - folds)) for profile, folds in profile_folds.items()
        }
        new_fits, reused = 0, 0
        for policy in POLICIES:
            for seed in SEEDS:
                digests = set()
                for condition in CONDITIONS:
                    key = seed, policy, condition
                    prefix = f"{policy}/seed{seed}/{condition}"
                    record = json.loads(archive.read(prefix + "/diagnostic.json"))
                    if record != records[key] or record["partition_seed"] != 42:
                        raise ValueError("stability cell differs from summary")
                    checkpoint_bytes = archive.read(prefix + "/selected_model.pt")
                    prediction_bytes = archive.read(prefix + "/inner_selection_predictions.jsonl")
                    if (hashlib.sha256(checkpoint_bytes).hexdigest() != record["checkpoint_sha256"]
                            or hashlib.sha256(prediction_bytes).hexdigest() != record["predictions_sha256"]
                            or not 0 <= record["maximum_replay_probability_difference"] <= 1e-6
                            or record["challenge_forward_or_scoring"] or record["submission_allowed"]):
                        raise ValueError("stability checkpoint/prediction provenance differs")
                    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
                    compact = condition in COMPACT
                    expected_keys = {
                        "state_dict", "condition", "policy", "seed", "partition_seed",
                        "configuration", "normalization", "protocol_sha256",
                    }
                    if not compact:
                        expected_keys.add("input_dimension")
                    if (set(saved) != expected_keys or saved["condition"] != condition
                            or saved["policy"] != policy or saved["seed"] != seed
                            or saved["partition_seed"] != 42
                            or saved["configuration"] != (settings["configuration"] if compact else
                                                          protocol["reference_configuration"])
                            or saved["normalization"] != normalizations[policy]
                            or saved["protocol_sha256"] != protocol_hash):
                        raise ValueError("stability checkpoint envelope differs")
                    state, fitted = saved["state_dict"], record["fit"]
                    parameter_count = sum(value.numel() for value in state.values() if value.is_floating_point())
                    count_key = "parameter_count" if compact else "training_parameter_count"
                    if (parameter_count != fitted[count_key]
                            or any(not torch.isfinite(value).all() for value in state.values())):
                        raise ValueError("stability parameters or finite values differ")
                    if compact:
                        width = settings["configuration"]["block_dim"]
                        if (state["readout.0.weight"].shape[1] != width * 7
                                or state["block_encoder.0.weight"].shape[1] != width * (
                                    4 if condition == COMPACT[0] else 2)
                                or any(name.startswith("transition.") for name in state)
                                or fitted["inactive_serialized_weight_count"] != 0
                                or (condition == COMPACT[1] and
                                    any(name.startswith("correspondence.") for name in state))):
                            raise ValueError("stability compact inactive-path pruning differs")
                        digests.add(fitted["shared_initialization_digest"])
                    expected_new = compact or seed != 42
                    if fitted["new_fit"] is not expected_new:
                        raise ValueError("stability fitting/reuse identity differs")
                    new_fits += int(expected_new)
                    reused += int(not expected_new)
                    rows = predictions[key]
                    labels = np.array([row["label"] for row in rows])
                    probabilities = np.asarray([row["probabilities"] for row in rows])
                    if (probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all()
                            or np.any(probabilities < 0) or np.any(probabilities > 1)
                            or not np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)):
                        raise ValueError("invalid stability probabilities")
                    scores = {"all_profiles": metrics(labels, probabilities, 4), "by_fitting_policy": {}}
                    for known in set(POLICIES) - {policy}:
                        indices = [index for index, row in enumerate(rows)
                                   if profile_policies[row["base_profile_id"]] == known]
                        scores["by_fitting_policy"][known] = metrics(
                            labels[indices], probabilities[indices], 4,
                        )
                    scores["worst_policy_macro_f1"] = min(
                        row["macro_f1"] for row in scores["by_fitting_policy"].values()
                    )
                    if scores != record["inner_selection"]:
                        raise ValueError("stability selection metrics differ from predictions")
                    history = fitted["history"]
                    if (not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
                            or len(history) > settings["configuration"]["max_epochs"]):
                        raise ValueError("stability training budget/history differs")
                    best = max(selection_key(row["inner_selection"]) for row in history)
                    selected = next(row for row in history if selection_key(row["inner_selection"]) == best)
                    if selected["epoch"] != fitted["best_epoch"] or selected["inner_selection"] != scores:
                        raise ValueError("stability checkpoint selection differs")
                    verified.append({
                        "seed": seed, "policy": policy, "condition": condition,
                        "inner_selection": scores, "parameter_count": parameter_count,
                        "new_fit": expected_new, "best_epoch": fitted["best_epoch"],
                    })
                if len(digests) != 1:
                    raise ValueError("paired compact initialization differs")
        if (new_fits, reused) != (30, 6) or decision(verified) != summary["decision"]:
            raise ValueError("stability aggregate or fitting counts differ")
    return {
        "scope": "read_only_compact_stability_receipt_audit", "verified_cells": len(verified),
        "new_fits_verified": new_fits, "reused_reference_checkpoints_verified": reused,
        "research_fits_added": 0, "summary_sha256": sha256_file(summary_path),
        "archive_sha256": sha256_file(archive_path),
        "selection_profiles_across_folds": len(profile_folds), "cells": verified,
        "decision": decision(verified), "research_tensors_opened": False,
        "challenge_predictions_opened": False, "independent_confirmation": False,
        "submission_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("archive", "summary", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite a completed stability audit")
    torch.set_num_threads(2)
    result = audit(args.archive, args.summary, args.receipt)
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
