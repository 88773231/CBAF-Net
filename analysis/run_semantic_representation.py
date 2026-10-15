"""Frozen 2x2 representation screen; no final-test or excluded-policy forwards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_observable_schema import training_identities
from analysis.run_encoder_objective_diagnostic import write_predictions
from analysis.run_factor_training_screen import POLICIES, policies_from_partitions
from analysis.run_observable_evolution import METADATA_SHA256, SOURCE_SHA256
from dataset_pipeline.frozen_semantic import (
    DIMENSIONS, DIMENSION, MODEL_ID, REVISION, WEIGHT_SHA256,
    FrozenSentenceEncoder, representation_features,
)
from dataset_pipeline.observable_cues import STEPS
from src.data_loader import apply_column_standardizer, fit_column_standardizer
from src.encoder_objective_diagnostic import validate_populations
from src.evidence_repair import write_new_json
from src.observable_evolution import INPUTS
from src.policy_robust import selection_scores
from src.provenance import sha256_file
from src.semantic_representation import CONDITIONS, make_model, predict, train


CONFIGURATION = {
    "width": 32, "readout": 128, "dropout": .2, "learning_rate": .001,
    "weight_decay": .001, "batch_size": 128, "max_epochs": 30, "patience": 6,
}
SOURCES = (
    "analysis/run_semantic_representation.py", "analysis/fetch_frozen_semantic_encoder.py",
    "analysis/audit_semantic_representation.py",
    "dataset_pipeline/frozen_semantic.py", "src/semantic_representation.py",
    "configs/semantic_representation.json", "requirements-semantic.txt",
    "dataset_pipeline/observable_cues.py", "src/observable_evolution.py",
    "dataset_pipeline/export_quadbot_v3_features.py",
    "dataset_pipeline/export_relative_timing.py", "src/data_loader.py",
    "src/encoder_objective_diagnostic.py", "src/policy_robust.py", "src/evidence_repair.py",
    "analysis/audit_observable_schema.py", "analysis/run_factor_training_screen.py",
    "analysis/run_observable_evolution.py",
)


def validate_protocol(settings):
    expected = {
        "schema": "frozen-text-representation-screen-v1", "dataset": "Quadbot", "seed": 42,
        "conditions": list(CONDITIONS),
        "encoder": {
            "model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
            "license": "apache-2.0", "declared_language": "en", "dimension": DIMENSION,
            "max_length": 256, "batch_size": 64, "pooling": "attention_masked_mean_then_l2",
            "empty_text": "zero_vector", "frozen": True, "trust_remote_code": False,
        },
        "configuration": CONFIGURATION, "new_fits": 12, "checkpoint_replays": 12,
        "minimum_representation_mean_macro_f1_gain": .0025, "positive_folds_required": 2,
        "known_policy_training_partitions_only": True,
        "excluded_policy_feature_extraction_or_forward": False,
        "original_validation_or_test_access": False, "gpu_required": False,
        "new_account_generation": False, "hyperparameter_search": False,
        "external_private_text_api": False, "algorithmic_novelty_established": False,
        "independent_confirmation": False, "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("frozen semantic representation protocol differs")


def selected_rows(source, metadata, receipts, policy):
    if source.name != "train.jsonl" or sha256_file(source) != SOURCE_SHA256:
        raise ValueError("original training source differs")
    if sha256_file(metadata) != METADATA_SHA256:
        raise ValueError("original training metadata differs")
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
        raise ValueError("known-policy partition differs")
    allowed = {rows[index]["sample_id"] for index in selected}
    extracted = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["sample_id"]
            if identity not in allowed:
                continue
            if identity in extracted or (str(row["base_profile_id"]), row["label"]) != identities[identity]:
                raise ValueError("selected training identity differs")
            extracted[identity] = row
    if set(extracted) != allowed:
        raise ValueError("selected training observations missing")
    ordered_indices = indices[0] + indices[1]
    return (
        [extracted[rows[index]["sample_id"]] for index in ordered_indices],
        [[rows[index] for index in part] for part in indices], policies,
    )


def prepare_populations(features, populations, policies):
    fitting_count = len(populations[0])
    output, transforms = {}, {}
    for kind in ("hash", "semantic"):
        output[kind] = [{
            "metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
            **{name: torch.from_numpy(np.stack([
                features[kind][index][name] for index in indices
            ])) for name in INPUTS},
        } for rows, indices in zip(populations, (
            range(fitting_count), range(fitting_count, len(features[kind])),
        ))]
        validate_populations(*output[kind], policies, 4)
        transforms[kind] = {}
        for name in ("numeric", "behavior", "style", "timing"):
            fitting = output[kind][0][name]
            transform = fit_column_standardizer(fitting.reshape(-1, fitting.shape[-1]))
            transforms[kind][name] = {key: value.tolist() for key, value in transform.items()}
            for population in output[kind]:
                value = population[name]
                population[name] = apply_column_standardizer(
                    value.reshape(-1, value.shape[-1]), transform,
                ).reshape_as(value)
    if transforms["hash"] != transforms["semantic"]:
        raise ValueError("nontext normalization differs across representations")
    for index in range(2):
        for name in ("numeric", "behavior", "style", "timing"):
            if not torch.equal(output["hash"][index][name], output["semantic"][index][name]):
                raise ValueError("nontext inputs differ across representations")
    return output, transforms["hash"]


def decision(records):
    identities = [(record["policy"], record["condition"]) for record in records]
    expected = {(policy, condition) for policy in POLICIES for condition in CONDITIONS}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicate representation study")
    scores = {identity: record["inner_selection"]["all_profiles"]["macro_f1"]
              for identity, record in zip(identities, records)}
    if any(not np.isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
        raise ValueError("invalid representation score")
    comparisons = {}
    for architecture in ("mlp", "backbone"):
        differences = np.array([
            scores[(policy, "semantic_" + architecture)] - scores[(policy, "hash_" + architecture)]
            for policy in POLICIES
        ])
        comparisons[architecture] = {
            "paired_fold_differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "positive_folds": int((differences > 0).sum()),
            "representation_gate_passed": bool(differences.mean() >= .0025
                                               and (differences > 0).sum() >= 2),
        }
    return {
        "mean_macro_f1_by_condition": {
            condition: float(np.mean([scores[(policy, condition)] for policy in POLICIES]))
            for condition in CONDITIONS
        },
        "representation_comparisons": comparisons,
        "representation_continuation_gate_passed": all(
            value["representation_gate_passed"] for value in comparisons.values()
        ),
        "folds_overlap_not_independent_replications": True,
        "algorithmic_novelty_established": False, "independent_confirmation": False,
        "threshold_is_not_a_ccfb_acceptance_standard": True,
        "module_ablation_or_architecture_superiority_established": False,
        "submission_allowed": False,
    }


def run(source, metadata, partition_root, protocol_path, model_folder, output):
    if output.exists():
        raise FileExistsError("preserve previous representation evidence")
    settings = json.loads(protocol_path.read_text())
    validate_protocol(settings)
    encoder = FrozenSentenceEncoder(
        model_folder, settings["encoder"]["batch_size"], cache_embeddings=True,
    )
    receipt_paths = {policy: partition_root / policy / "normalization.json" for policy in POLICIES}
    receipt_hashes = {policy: sha256_file(path) for policy, path in receipt_paths.items()}
    receipts = {policy: json.loads(path.read_text()) for policy, path in receipt_paths.items()}
    root = Path(__file__).resolve().parents[1]
    write_new_json(output / "protocol.json", {
        **settings, "source_hashes": {name: sha256_file(root / name) for name in SOURCES},
        "prior_partition_hashes": receipt_hashes,
        "input_hashes": {"training_source": SOURCE_SHA256, "training_metadata": METADATA_SHA256},
        "encoder_snapshot_manifest": encoder.manifest,
        "feature_dimensions": DIMENSIONS, "steps": STEPS,
        "lexical_control_dimension_changed_from_256_to_384": True,
        "both_representations_share_normalized_texts_nontext_inputs_and_objective": True,
        "prior_256_dimension_or_legacy_scores_are_not_matched_controls": True,
        "length_bucketing_restores_original_text_order": True,
        "stateless_frozen_text_cache_shared_only_after_allowlisted_text_requests": True,
        "cache_never_fits_transforms_or_modifies_encoder_weights": True,
    })
    records = []
    for policy in POLICIES:
        def progress(event):
            print(json.dumps({"policy": policy, **event}), flush=True)
        raw, populations, policies = selected_rows(source, metadata, receipts, policy)
        progress({"event": "encoding_allowed_rows", "rows": len(raw)})
        features, statistics = representation_features(raw, encoder, progress)
        values, normalization = prepare_populations(features, populations, policies)
        del raw, features
        cache = output / policy / "known_observations.pt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"populations": values, "normalization": normalization}, cache)
        write_new_json(output / policy / "feature_receipt.json", {
            "cache_sha256": sha256_file(cache), "statistics": statistics,
            "encoder_statistics_cumulative": dict(encoder.statistics),
            "normalization": normalization, "normalization_scope": "inner_fitting_only",
            "fitting_indices": receipts[policy]["fitting_indices"],
            "selection_indices": receipts[policy]["selection_indices"],
            "partition_sha256": receipt_hashes[policy],
            "excluded_policy_encoded_transformed_or_forwarded": False,
        })
        for condition in CONDITIONS:
            kind, architecture = condition.split("_")
            def fit_progress(event):
                progress({"condition": condition, **event})
            model, fitted = train(*values[kind], policies, architecture, CONFIGURATION,
                                  settings["seed"], fit_progress)
            probabilities = predict(model, values[kind][1], CONFIGURATION["batch_size"])
            folder = output / policy / condition
            folder.mkdir()
            checkpoint = folder / "selected_model.pt"
            torch.save({
                "state_dict": model.state_dict(), "condition": condition,
                "configuration": CONFIGURATION, "policy": policy,
                "protocol_sha256": sha256_file(output / "protocol.json"),
            }, checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            replay = make_model(CONFIGURATION, architecture)
            replay.load_state_dict(saved["state_dict"], strict=True)
            replayed = predict(replay, values[kind][1], CONFIGURATION["batch_size"])
            error = float(np.abs(probabilities - replayed).max())
            if error > 1e-6:
                raise ValueError("representation checkpoint replay differs")
            prediction = folder / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[kind][1], probabilities)
            record = {
                "policy": policy, "condition": condition, "fit": fitted,
                "inner_selection": selection_scores(values[kind][1], probabilities, policies, 4),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": error,
                "original_validation_test_or_excluded_policy_forwards": False,
                "submission_allowed": False,
            }
            records.append(record)
            write_new_json(folder / "diagnostic.json", record)
            progress({"event": "cell_complete", "completed": len(records), "planned": 12})
        for architecture in ("mlp", "backbone"):
            paired = [record["fit"]["initialization_digest"] for record in records
                      if record["policy"] == policy and record["condition"].endswith(architecture)]
            if len(paired) != 2 or len(set(paired)) != 1:
                raise ValueError("dimension-matched initialization differs")
    if any(sha256_file(path) != receipt_hashes[policy] for policy, path in receipt_paths.items()):
        raise ValueError("original partitions changed")
    if sha256_file(source) != SOURCE_SHA256 or sha256_file(metadata) != METADATA_SHA256:
        raise ValueError("frozen training inputs changed")
    write_new_json(output / "summary.json", {
        "protocol_sha256": sha256_file(output / "protocol.json"), "records": records,
        **decision(records), "new_fits_completed": 12, "checkpoint_replays_completed": 12,
        "new_account_generation": False, "gpu_used": False,
        "original_validation_or_test_access": False,
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "metadata", "partition-root", "protocol", "model-folder", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.source, args.metadata, args.partition_root, args.protocol, args.model_folder, args.output)


if __name__ == "__main__":
    main()
