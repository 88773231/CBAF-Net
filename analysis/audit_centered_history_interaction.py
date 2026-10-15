"""Recompute the complete retained interaction matrix without research tensors."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch

from analysis.run_centered_history_interaction import CONDITIONS, NEW, decision, source_hashes, validate_settings
from analysis.run_factor_training_screen import POLICIES
from src.evidence_repair import metrics, write_new_json
from src.policy_robust import selection_key
from src.provenance import sha256_file


def audit(archive_path, summary_path, receipt_path, settings_path=None):
    receipt = json.loads(receipt_path.read_text())
    if (receipt["phase"] != "complete" or receipt["new_fits"] != 6
            or receipt["reused_exact_partition_checkpoints"] != 9 or receipt["checkpoint_replays"] != 15
            or receipt["gpu_used"] or receipt["challenge_forward_or_scoring"]
            or receipt["submission_allowed"] or not receipt["reference_checkpoints_unchanged"]
            or receipt["summary_sha256"] != sha256_file(summary_path)
            or receipt["archive_sha256"] != sha256_file(archive_path)):
        raise ValueError("interaction completion receipt differs")
    settings_path = settings_path or (Path(__file__).parents[1] / "configs/centered_history_interaction.json")
    settings = json.loads(settings_path.read_text())
    validate_settings(settings)
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            "\\" in name or name.startswith("/") or ".." in PurePosixPath(name).parts
            or "challenge_predictions" in name or PurePosixPath(name).name.startswith(("test_", "val_"))
            for name in names):
            raise ValueError("unexpected interaction archive member")
        if archive.read("summary.json") != summary_path.read_bytes():
            raise ValueError("retained interaction summary differs")
        summary = json.loads(archive.read("summary.json"))
        if (summary["scope"] != "one_seed_known_policy_centered_interaction_screen"
                or summary["dataset"] != "Quadbot" or summary["new_fits"] != 6
                or summary["reused_exact_partition_checkpoints"] != 9 or summary["checkpoint_replays"] != 15
                or not summary["reference_checkpoints_unchanged"] or summary["challenge_forward_or_scoring"]
                or summary["independent_confirmation"] or summary["algorithmic_novelty_established"]
                or summary["submission_allowed"]):
            raise ValueError("interaction summary boundary differs")
        protocol_bytes = archive.read("protocol.json")
        protocol = json.loads(protocol_bytes)
        if (any(protocol[key] != value for key, value in settings.items())
                or protocol["source_hashes"] != source_hashes()
                or protocol["excluded_row_feature_transforms_or_model_forwards"]
                or protocol["exact_capacity_or_compute_matching_claimed"]):
            raise ValueError("interaction source or access boundary differs")
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        decision(summary["records"], settings["minimum_module_mean_macro_f1_gain"])
        records = {(row["policy"], row["condition"]): row for row in summary["records"]}
        predictions, profile_folds = {}, {}
        for policy in POLICIES:
            for condition in CONDITIONS:
                prefix = f"{policy}/{condition}"
                rows = [json.loads(line) for line in
                        archive.read(prefix + "/inner_selection_predictions.jsonl").decode().splitlines()]
                identities = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows]
                if not rows or len({row[0] for row in identities}) != len(rows):
                    raise ValueError("invalid selection identities")
                predictions[(policy, condition)] = rows
                if identities != [(row["sample_id"], row["base_profile_id"], row["label"])
                                  for row in predictions[(policy, CONDITIONS[0])]]:
                    raise ValueError("paired interaction rows differ")
                groups = {}
                for _, profile, label in identities:
                    groups.setdefault(profile, []).append(label)
                    profile_folds.setdefault(profile, set()).add(policy)
                if any(sorted(labels) != list(range(4)) for labels in groups.values()):
                    raise ValueError("selection profiles are not class complete")
        if any(len(folds) != 2 for folds in profile_folds.values()):
            raise ValueError("overlapping fold membership differs")
        profile_policy = {
            profile: next(iter(set(POLICIES) - folds)) for profile, folds in profile_folds.items()}
        verified = []
        for policy in POLICIES:
            normalization = json.loads(archive.read(policy + "/normalization.json"))
            for condition in CONDITIONS:
                prefix = f"{policy}/{condition}"
                record = json.loads(archive.read(prefix + "/diagnostic.json"))
                if record != records[(policy, condition)] or record["seed"] != 42:
                    raise ValueError("interaction cell differs from summary")
                checkpoint = archive.read(prefix + "/selected_model.pt")
                prediction = archive.read(prefix + "/inner_selection_predictions.jsonl")
                if (hashlib.sha256(checkpoint).hexdigest() != record["checkpoint_sha256"]
                        or hashlib.sha256(prediction).hexdigest() != record["predictions_sha256"]
                        or not 0 <= record["maximum_replay_probability_difference"] <= 1e-6
                        or record["challenge_forward_or_scoring"] or record["submission_allowed"]):
                    raise ValueError("interaction provenance differs")
                saved = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
                expected = {
                    "condition": condition, "policy": policy, "seed": 42,
                    "normalization": normalization, "protocol_sha256": protocol_hash}
                if set(saved) != {"state_dict", *expected} or any(
                    saved[key] != value for key, value in expected.items()):
                    raise ValueError("interaction checkpoint envelope differs")
                state, fitted = saved["state_dict"], record["fit"]
                count = sum(value.numel() for value in state.values() if value.is_floating_point())
                parameter_key = ("parameter_count" if condition in (*NEW, "without_correspondence")
                                 else "training_parameter_count")
                if (count != fitted[parameter_key] or any(not torch.isfinite(value).all()
                                                          for value in state.values())
                        or fitted["new_fit"] is not (condition in NEW)):
                    raise ValueError("interaction parameters or fitting identity differ")
                rows = predictions[(policy, condition)]
                labels = np.array([row["label"] for row in rows])
                probabilities = np.asarray([row["probabilities"] for row in rows])
                if (len(rows) != len(normalization["selection_indices"])
                        or probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all()
                        or np.any(probabilities < 0) or np.any(probabilities > 1)
                        or not np.allclose(probabilities.sum(1), 1, atol=1e-6, rtol=0)):
                    raise ValueError("invalid interaction probabilities")
                scores = {"all_profiles": metrics(labels, probabilities, 4), "by_fitting_policy": {}}
                for known in set(POLICIES) - {policy}:
                    indices = [index for index, row in enumerate(rows)
                               if profile_policy[row["base_profile_id"]] == known]
                    scores["by_fitting_policy"][known] = metrics(labels[indices], probabilities[indices], 4)
                scores["worst_policy_macro_f1"] = min(
                    row["macro_f1"] for row in scores["by_fitting_policy"].values())
                if scores != record["inner_selection"]:
                    raise ValueError("interaction scores differ from archived predictions")
                history = fitted["history"]
                if (not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
                        or len(history) > settings["configuration"]["max_epochs"]):
                    raise ValueError("interaction training budget differs")
                best = max(selection_key(row["inner_selection"]) for row in history)
                selected = next(row for row in history if selection_key(row["inner_selection"]) == best)
                if selected["epoch"] != fitted["best_epoch"] or selected["inner_selection"] != scores:
                    raise ValueError("interaction checkpoint choice differs")
                verified.append({"policy": policy, "condition": condition, "inner_selection": scores,
                                 "parameter_count": count, "new_fit": condition in NEW})
        evaluated = decision(verified, settings["minimum_module_mean_macro_f1_gain"])
        if evaluated != summary["decision"]:
            raise ValueError("interaction aggregate differs")
    return {
        "scope": "read_only_centered_history_interaction_receipt_audit",
        "verified_cells": len(verified), "new_fits_verified": 6, "reused_checkpoints_verified": 9,
        "selection_profiles_across_folds": len(profile_folds), "cells": verified, "decision": evaluated,
        "summary_sha256": sha256_file(summary_path), "archive_sha256": sha256_file(archive_path),
        "research_tensors_opened": False, "research_fits_added": 0,
        "challenge_predictions_opened": False, "submission_allowed": False}


def main():
    parser = argparse.ArgumentParser()
    for name in ("archive", "summary", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.summary, args.receipt)
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
