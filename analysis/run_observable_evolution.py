"""Frozen same-input screen on existing known-policy training partitions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_observable_schema import training_identities
from analysis.run_encoder_objective_diagnostic import write_predictions
from analysis.run_factor_training_screen import POLICIES, policies_from_partitions
from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS, observe
from src.data_loader import apply_column_standardizer, fit_column_standardizer
from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import write_new_json
from src.observable_evolution import CONDITIONS, INPUTS, make_model, predict, train
from src.policy_robust import selection_scores
from src.provenance import sha256_file


MASTER_SHA256 = "b5ce0d14eed32b053f5ca78409cc07f0d8ff595d9511e762a127e7a5c2e7b16b"
SOURCE_SHA256 = "b93bf2bb3b78823ab25dcd8b2b7ea4ab2e1585af8d6cfbf7a5d5300323f3b89f"
METADATA_SHA256 = "30007e8b65c4770dc72a258f13c0e8a98ef1d26f745cdbd972609b66014e4dc6"
CONFIGURATION = {
    "seed": 42, "width": 32, "readout": 128, "dropout": .2, "learning_rate": .001,
    "weight_decay": .001, "batch_size": 128, "max_epochs": 30, "patience": 6,
    "cue_supervision_weight": .1, "distillation_weight": .1, "temperature": 2., "warmup_epochs": 3,
}
SOURCES = (
    "dataset_pipeline/observable_cues.py", "dataset_pipeline/export_quadbot_v3_features.py",
    "dataset_pipeline/export_relative_timing.py", "src/observable_evolution.py",
    "analysis/run_observable_evolution.py", "analysis/audit_observable_evolution.py",
    "configs/observable_evolution.json", "src/data_loader.py", "src/policy_robust.py",
    "src/encoder_objective_diagnostic.py", "src/evidence_repair.py",
    "analysis/run_factor_training_screen.py",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256_file(root / name) for name in SOURCES}


def validate_settings(settings):
    expected = {
        "schema": "known-policy-observable-evolution-screen-v1", "dataset": "Quadbot",
        "conditions": list(CONDITIONS), "configuration": CONFIGURATION,
        "new_fits": 15, "checkpoint_replays": 15, "minimum_module_mean_macro_f1_gain": .0025,
        "no_hyperparameter_search": True, "input_representation_changed": True,
        "legacy_input_scores_are_not_matched_controls": True,
        "same_input_compute_or_capacity_matching_claimed": False,
        "original_validation_or_test_access": False,
        "excluded_policy_feature_extraction_or_forward": False,
        "independent_confirmation": False, "algorithmic_novelty_established": False,
        "gpu_required": False, "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("frozen observation screen configuration differs")


def prepare_fold(source, metadata, receipts, policy):
    if source.name != "train.jsonl" or sha256_file(source) != SOURCE_SHA256:
        raise ValueError("original training source changed")
    if sha256_file(metadata) != METADATA_SHA256:
        raise ValueError("original training identities changed")
    identities = training_identities(metadata)
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    policies = policies_from_partitions(rows, receipts)
    receipt = receipts[policy]
    indices = [receipt["fitting_indices"], receipt["selection_indices"]]
    selected = set(indices[0] + indices[1])
    excluded = set(range(len(rows))) - selected
    if (len(selected) != sum(map(len, indices))
            or excluded != set(receipt["challenge_indices"])
            or any(policies[rows[index]["base_profile_id"]] != policy for index in excluded)
            or any(policies[rows[index]["base_profile_id"]] == policy for index in selected)):
        raise ValueError("known-policy selection identity differs")
    allowed = {rows[index]["sample_id"] for index in selected}
    features = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["sample_id"]
            if identity not in allowed:
                continue
            if identity in features or (str(row["base_profile_id"]), row["label"]) != identities[identity]:
                raise ValueError("selected source identity differs")
            features[identity] = observe(row)
    if set(features) != allowed:
        raise ValueError("selected observation records missing")
    values = [{
        "metadata": [rows[index] for index in part],
        "labels": torch.tensor([rows[index]["label"] for index in part]),
        **{name: torch.from_numpy(np.stack([
            features[rows[index]["sample_id"]][name] for index in part
        ])) for name in INPUTS},
    } for part in indices]
    validate_populations(*values, policies, 4)
    normalization = {}
    for name in ("numeric", "behavior", "style", "timing"):
        shape = values[0][name].shape
        fitting_matrix = values[0][name].reshape(-1, shape[-1])
        transform = fit_column_standardizer(fitting_matrix)
        normalization[name] = {key: value.tolist() for key, value in transform.items()}
        for population in values:
            tensor = population[name]
            population[name] = apply_column_standardizer(
                tensor.reshape(-1, tensor.shape[-1]), transform).reshape_as(tensor)
    return values, policies, normalization


def decision(records):
    identities = [(record["policy"], record["condition"]) for record in records]
    expected = {(policy, condition) for policy in POLICIES for condition in CONDITIONS}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicate observation screen")
    scores = {identity: record["inner_selection"]["all_profiles"]["macro_f1"]
              for identity, record in zip(identities, records)}
    if any(not np.isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
        raise ValueError("invalid observation screen score")
    full = np.array([scores[(policy, CONDITIONS[0])] for policy in POLICIES])
    comparisons = {}
    for condition in CONDITIONS[1:]:
        differences = full - np.array([scores[(policy, condition)] for policy in POLICIES])
        minimum = .0025 if condition != "same_input_mlp" else 0.
        comparisons[condition] = {
            "paired_fold_differences": differences.tolist(),
            "mean_difference": float(differences.mean()), "positive_folds": int((differences > 0).sum()),
            "required_mean_gain": minimum,
            "gate_passed": bool(differences.mean() >= minimum and differences.mean() > 0
                                and (differences > 0).sum() >= 2),
        }
    return {
        "mean_macro_f1_by_condition": {
            condition: float(np.mean([scores[(policy, condition)] for policy in POLICIES]))
            for condition in CONDITIONS},
        "comparisons": comparisons,
        "development_continuation_gate_passed": all(item["gate_passed"] for item in comparisons.values()),
        "folds_overlap_not_independent_replications": True,
        "single_seed_not_stability_confirmation": True,
        "threshold_is_not_a_ccfb_acceptance_standard": True,
        "legacy_input_scores_are_not_matched_comparisons": True,
        "independent_confirmation": False, "submission_allowed": False,
    }


def run(source, metadata, partition_root, protocol_path, output):
    if output.exists():
        raise FileExistsError("preserve observation study evidence")
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings)
    receipt_paths = {policy: partition_root / policy / "normalization.json" for policy in POLICIES}
    receipt_hashes = {policy: sha256_file(path) for policy, path in receipt_paths.items()}
    receipts = {policy: json.loads(path.read_text()) for policy, path in receipt_paths.items()}
    write_new_json(output / "protocol.json", {
        **settings, "source_hashes": source_hashes(),
        "input_hashes": {"training_source": SOURCE_SHA256, "training_metadata": METADATA_SHA256},
        "prior_partition_hashes": receipt_hashes,
        "feature_dimensions": FEATURE_DIMENSIONS, "steps": STEPS,
        "hash_features_are_not_semantic_embeddings": True,
        "simulator_internal_actions_graph_step_ids_or_provenance_features": False,
        "training_only_self_distillation_not_cdrbot_reproduction": True,
    })
    records = []
    for policy in POLICIES:
        print(json.dumps({"event": "extracting_known_observations", "policy": policy}), flush=True)
        values, policies, normalization = prepare_fold(source, metadata, receipts, policy)
        write_new_json(output / policy / "normalization.json", {
            "transforms": normalization, "fit_scope": "inner_fitting_profiles_only",
            "partition_sha256": receipt_hashes[policy],
            "fitting_indices": receipts[policy]["fitting_indices"],
            "selection_indices": receipts[policy]["selection_indices"],
            "excluded_policy_transformed_or_forwarded": False,
        })
        for condition in CONDITIONS:
            folder = output / policy / condition
            folder.mkdir(parents=True)
            def progress(event):
                print(json.dumps({"policy": policy, "condition": condition, **event}), flush=True)
            model, fit = train(*values, policies, condition, CONFIGURATION, progress)
            probabilities = predict(model, values[1], CONFIGURATION["batch_size"])
            checkpoint = folder / "selected_model.pt"
            saved = {"state_dict": model.state_dict(), "condition": condition, "policy": policy,
                     "configuration": CONFIGURATION, "normalization": normalization,
                     "protocol_sha256": sha256_file(output / "protocol.json")}
            torch.save(saved, checkpoint)
            reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if set(saved) != set(reloaded) or any(
                    saved[key] != reloaded[key] for key in saved if key != "state_dict"):
                raise ValueError("observation checkpoint envelope differs")
            replay = make_model(CONFIGURATION, condition)
            replay.load_state_dict(reloaded["state_dict"], strict=True)
            replayed = predict(replay, values[1], CONFIGURATION["batch_size"])
            error = float(np.abs(probabilities - replayed).max())
            if error > 1e-6:
                raise ValueError("observation checkpoint replay differs")
            prediction = folder / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[1], probabilities)
            record = {
                "policy": policy, "condition": condition, "fit": fit,
                "inner_selection": selection_scores(values[1], probabilities, policies, 4),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": error,
                "original_validation_test_or_excluded_policy_forwards": False,
                "submission_allowed": False,
            }
            records.append(record)
            write_new_json(folder / "diagnostic.json", record)
            print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 15}), flush=True)
        shared = {record["fit"]["shared_initialization_digest"] for record in records
                  if record["policy"] == policy and record["condition"] != "same_input_mlp"}
        if len(shared) != 1:
            raise ValueError("active shared initialization differs")
    if any(sha256_file(path) != receipt_hashes[policy] for policy, path in receipt_paths.items()):
        raise ValueError("original partitions changed")
    write_new_json(output / "summary.json", {
        "protocol_sha256": sha256_file(output / "protocol.json"), "records": records,
        **decision(records), "new_fits_completed": 15, "checkpoint_replays_completed": 15,
        "new_account_generation": False, "gpu_used": False,
        "original_validation_or_test_access": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.source, args.metadata, args.partition_root, args.protocol, args.output)


if __name__ == "__main__":
    main()
