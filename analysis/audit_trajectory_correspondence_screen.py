"""Read-only receipt audit; never open research tensors or challenge predictions."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from analysis.run_factor_training_screen import POLICIES
from analysis.run_trajectory_correspondence_screen import source_hashes, validate_settings
from src.evidence_repair import metrics, write_new_json
from src.policy_robust import selection_key
from src.provenance import sha256_file
from src.trajectory_correspondence import CONDITIONS


def decision(records, reference):
    expected = {(policy, condition) for policy in POLICIES for condition in CONDITIONS}
    identities = [(row["policy"], row["condition"]) for row in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicated architecture comparison")
    if set(reference) != set(POLICIES):
        raise ValueError("incomplete reference selection comparison")
    scores = {
        identity: row["inner_selection"]["all_profiles"]["macro_f1"]
        for identity, row in zip(identities, records)
    }
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value in [*scores.values(), *reference.values()]):
        raise ValueError("invalid selection score")
    full = np.array([scores[(policy, CONDITIONS[0])] for policy in POLICIES])
    comparisons = {}
    for condition in CONDITIONS[1:]:
        removed = np.array([scores[(policy, condition)] for policy in POLICIES])
        difference = full - removed
        comparisons[condition] = {
            "mean_macro_f1": float(removed.mean()),
            "paired_macro_f1_differences": difference.tolist(),
            "mean_difference": float(difference.mean()),
            "strict_positive_folds": int((difference > 0).sum()),
            "removal_gate_passed": bool(difference.mean() > 0 and (difference > 0).sum() >= 2),
        }
    reference_mean = float(np.mean([reference[policy] for policy in POLICIES]))
    reference_gate = bool(full.mean() >= reference_mean)
    return {
        "full_mean_macro_f1": float(full.mean()),
        "reference_mean_macro_f1": reference_mean,
        "reference_mean_gate_passed": reference_gate,
        "removal_comparisons": comparisons,
        "feasibility_gate_passed": bool(
            reference_gate and all(row["removal_gate_passed"] for row in comparisons.values())
        ),
        "folds_are_overlapping_not_independent_replications": True,
        "independent_confirmation": False, "submission_allowed": False,
    }


def audit(archive_path, summary_path, receipt_path, reference_protocol_path, reference_selection_path):
    receipt = json.loads(receipt_path.read_text())
    if (receipt["phase"] != "complete" or receipt["completed_new_fits"] != 12
            or receipt["checkpoint_replays"] != 12 or receipt["gpu_used"]
            or receipt["challenge_forward_or_scoring"] or receipt["submission_allowed"]
            or not receipt["reference_files_unchanged"]
            or receipt["summary_sha256"] != sha256_file(summary_path)
            or receipt["evidence_archive_sha256"] != sha256_file(archive_path)):
        raise ValueError("completion receipt or evidence hashes differ")
    reference_protocol = json.loads(reference_protocol_path.read_text())
    if reference_protocol["original_val_paths_opened"] or reference_protocol["test_paths_opened"]:
        raise ValueError("reference was not training-only")
    reference_summary = json.loads(reference_selection_path.read_text())
    if (reference_summary["dataset"] != "Quadbot"
            or reference_summary["challenge_forward_or_scoring"]
            or reference_summary["independent_confirmation"]
            or reference_summary["submission_allowed"]):
        raise ValueError("reference selection scope differs")
    reference_records = {
        row["policy"]: row for row in reference_summary["records"]
        if row["condition"] == "supervised_reference" and row["seed"] == 42
    }
    if len(reference_records) != 3:
        raise ValueError("reference selections are incomplete")
    preserved_reference_hashes = {
        str(path): sha256_file(path)
        for path in (reference_protocol_path, reference_selection_path)
    }
    verified = []
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith(("/", "\\")) or ".." in Path(name).parts
            or "challenge_predictions" in name or Path(name).name.startswith(("val_", "test_"))
            for name in names
        ):
            raise ValueError("unexpected or duplicate architecture evidence member")
        if archive.read("summary.json") != summary_path.read_bytes():
            raise ValueError("archive summary differs from downloaded summary")
        summary = json.loads(archive.read("summary.json"))
        if (summary["scope"] != "one_seed_known_policy_architecture_feasibility"
                or summary["dataset"] != "Quadbot" or summary["new_fits"] != 12
                or summary["checkpoint_replays"] != 12 or not summary["reference_files_unchanged"]
                or summary["challenge_forward_or_scoring"] or summary["independent_confirmation"]
                or summary["architecture_novelty_established"] or summary["submission_allowed"]):
            raise ValueError("architecture summary scope differs")
        protocol_bytes = archive.read("protocol.json")
        protocol = json.loads(protocol_bytes)
        settings = json.loads((Path(__file__).parents[1] /
                               "configs/trajectory_correspondence_screen.json").read_text())
        validate_settings(settings)
        if ({key: protocol[key] for key in settings} != settings
                or protocol["source_hashes"] != source_hashes()
                or protocol["input_hashes"] != reference_protocol["input_hashes"]
                or protocol["reference_protocol_sha256"] != sha256_file(reference_protocol_path)
                or protocol["excluded_row_feature_transforms_or_model_forwards"]
                or protocol["exact_capacity_or_compute_matching_claimed"]):
            raise ValueError("architecture input/source/protocol boundary differs")
        protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
        predictions, normalizations = {}, {}
        for policy in POLICIES:
            normalizations[policy] = json.loads(archive.read(policy + "/normalization.json"))
            for condition in CONDITIONS:
                prefix = policy + "/" + condition
                predictions[(policy, condition)] = [
                    json.loads(line) for line in archive.read(
                        prefix + "/inner_selection_predictions.jsonl"
                    ).decode().splitlines()
                ]
        identities, profile_folds = {}, {}
        for policy in POLICIES:
            rows = predictions[(policy, CONDITIONS[0])]
            ordered = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows]
            if (not rows or len({item[0] for item in ordered}) != len(rows)
                    or len(rows) != len(normalizations[policy]["selection_indices"])):
                raise ValueError("selection identities or row counts differ")
            groups = {}
            for sample, profile, label in ordered:
                if sample in identities and identities[sample] != (profile, label):
                    raise ValueError("cross-fold sample identity differs")
                identities[sample] = profile, label
                groups.setdefault(profile, []).append(label)
                profile_folds.setdefault(profile, set()).add(policy)
            if any(sorted(labels) != list(range(4)) for labels in groups.values()):
                raise ValueError("selection profiles are not class-complete")
            for condition in CONDITIONS:
                if [(row["sample_id"], row["base_profile_id"], row["label"])
                    for row in predictions[(policy, condition)]] != ordered:
                    raise ValueError("paired selection order differs")
        if any(len(folds) != 2 for folds in profile_folds.values()):
            raise ValueError("seed-42 known-policy selection membership differs")
        profile_policies = {
            profile: next(iter(set(POLICIES) - folds)) for profile, folds in profile_folds.items()
        }
        records = {(row["policy"], row["condition"]): row for row in summary["records"]}
        if len(records) != 12 or len(summary["records"]) != 12:
            raise ValueError("architecture records are incomplete or duplicated")
        for policy in POLICIES:
            digests = set()
            for condition in CONDITIONS:
                prefix = policy + "/" + condition
                record = json.loads(archive.read(prefix + "/diagnostic.json"))
                if record != records[(policy, condition)] or record["seed"] != 42:
                    raise ValueError("cell record differs from summary")
                checkpoint_bytes = archive.read(prefix + "/selected_model.pt")
                prediction_bytes = archive.read(prefix + "/inner_selection_predictions.jsonl")
                if (hashlib.sha256(checkpoint_bytes).hexdigest() != record["checkpoint_sha256"]
                        or hashlib.sha256(prediction_bytes).hexdigest() != record["predictions_sha256"]
                        or record["maximum_replay_probability_difference"] > 1e-6
                        or record["challenge_forward_or_scoring"] or record["submission_allowed"]):
                    raise ValueError("cell provenance/replay boundary differs")
                saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
                if (set(saved) != {"state_dict", "condition", "policy", "seed", "configuration",
                                   "normalization", "protocol_sha256"}
                        or saved["condition"] != condition or saved["policy"] != policy
                        or saved["seed"] != 42 or saved["configuration"] != settings["configuration"]
                        or saved["normalization"] != normalizations[policy]
                        or saved["protocol_sha256"] != protocol_hash):
                    raise ValueError("checkpoint identity differs")
                fit = record["fit"]
                digests.add(fit["shared_initialization_digest"])
                parameter_count = sum(
                    value.numel() for value in saved["state_dict"].values()
                    if value.is_floating_point()
                )
                if parameter_count != fit["parameter_count"] or any(
                    not torch.isfinite(value).all() for value in saved["state_dict"].values()
                ):
                    raise ValueError("checkpoint parameter count or finite values differ")
                rows = predictions[(policy, condition)]
                labels = [row["label"] for row in rows]
                probabilities = np.asarray([row["probabilities"] for row in rows])
                scores = {"all_profiles": metrics(labels, probabilities, 4), "by_fitting_policy": {}}
                for known in set(POLICIES) - {policy}:
                    indices = [index for index, row in enumerate(rows)
                               if profile_policies[row["base_profile_id"]] == known]
                    scores["by_fitting_policy"][known] = metrics(
                        np.asarray(labels)[indices], probabilities[indices], 4,
                    )
                scores["worst_policy_macro_f1"] = min(
                    row["macro_f1"] for row in scores["by_fitting_policy"].values()
                )
                if scores != record["inner_selection"]:
                    raise ValueError("selection metrics differ from predictions")
                best = max(selection_key(row["inner_selection"]) for row in fit["history"])
                selected = next(row for row in fit["history"]
                                if selection_key(row["inner_selection"]) == best)
                if selected["epoch"] != fit["best_epoch"] or selected["inner_selection"] != scores:
                    raise ValueError("checkpoint selection rule differs")
                verified.append({
                    "policy": policy, "condition": condition, "seed": 42,
                    "inner_selection": scores, "parameter_count": parameter_count,
                    "inactive_serialized_weight_count": fit["inactive_serialized_weight_count"],
                    "best_epoch": fit["best_epoch"], "epochs_run": len(fit["history"]),
                    "optimizer_updates": fit["optimizer_updates"],
                    "elapsed_seconds": fit["elapsed_seconds"],
                })
            if len(digests) != 1:
                raise ValueError("shared initialization digest differs")
    if any(sha256_file(Path(path)) != value for path, value in preserved_reference_hashes.items()):
        raise ValueError("local reference evidence changed")
    return {
        "scope": "read_only_known_policy_receipt_audit",
        "verified_cells": len(verified), "research_fits_added": 0,
        "summary_sha256": sha256_file(summary_path),
        "archive_sha256": sha256_file(archive_path),
        "reference_selection_sha256": sha256_file(reference_selection_path),
        "selection_profiles_across_folds": len(profile_folds), "cells": verified,
        "gate": decision(verified, {
            policy: reference_records[policy]["inner_selection"]["all_profiles"]["macro_f1"]
            for policy in POLICIES
        }),
        "challenge_predictions_opened": False, "research_tensors_opened": False,
        "independent_confirmation": False, "submission_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("archive", "summary", "receipt", "reference-protocol", "reference-selections", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite a completed audit")
    torch.set_num_threads(2)
    result = audit(args.archive, args.summary, args.receipt,
                   args.reference_protocol, args.reference_selections)
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
