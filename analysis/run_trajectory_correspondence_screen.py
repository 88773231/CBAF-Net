"""One bounded, known-policy architecture feasibility check on existing rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_encoder_objective_diagnostic import (
    load_known_fold, source_hashes as diagnostic_sources, write_predictions,
)
from analysis.run_factor_training_screen import POLICIES
from src.evidence_repair import write_new_json
from src.policy_robust import selection_scores
from src.provenance import sha256_file
from src.trajectory_correspondence import CONDITIONS, make_model, predict, train


SOURCES = (
    "src/trajectory_correspondence.py", "analysis/run_trajectory_correspondence_screen.py",
    "configs/trajectory_correspondence_screen.json",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**diagnostic_sources(), **{name: sha256_file(root / name) for name in SOURCES}}


def validate_settings(settings):
    expected = {
        "schema": "known-policy-trajectory-correspondence-screen-v1",
        "dataset": "Quadbot", "seeds": [42], "conditions": list(CONDITIONS),
        "configuration": {
            "block_dim": 64, "readout_dim": 128, "dropout": .2,
            "learning_rate": .001, "weight_decay": .001, "batch_size": 128,
            "max_epochs": 30, "patience": 6,
        },
        "new_fits": 12, "checkpoint_replays": 12, "cpu_threads": 2,
        "selection_population": "known_policy_inner_selection_only",
        "challenge_forward_or_scoring": False, "original_validation_or_test_access": False,
        "no_hyperparameter_search": True, "independent_confirmation": False,
        "algorithmic_novelty_established": False, "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("fixed architecture screen or evidence boundary differs")


def verified_checkpoint(checkpoint, identity):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(saved) != {"state_dict", *identity} or any(
        saved[key] != value for key, value in identity.items()
    ):
        raise ValueError("architecture checkpoint envelope differs")
    return saved["state_dict"]


def run(data_dir, temporal, control_root, protocol_path, output):
    if output.exists():
        raise FileExistsError("architecture screen evidence must not be overwritten")
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings)
    config, records = settings["configuration"], []
    reference_files = {
        str(path): sha256_file(path) for path in (control_root / "Quadbot").rglob("*")
        if path.is_file()
    }
    if not reference_files:
        raise ValueError("reference evidence is missing")
    for policy in POLICIES:
        values, policies, normalization, reference, inputs = load_known_fold(
            data_dir, temporal, control_root, policy,
        )
        if not records:
            write_new_json(output / "protocol.json", {
                **settings, "source_hashes": source_hashes(), "input_hashes": inputs,
                "reference_protocol_sha256": sha256_file(control_root / "Quadbot/protocol.json"),
                "reference_configuration": reference["configuration"],
                "architecture": "historical_cross_modal_discrepancy_and_adjacent_block_transition",
                "raw_hashes_are_not_semantic_embeddings": True,
                "historical_attention_is_not_causal_evidence": True,
                "action_and_text_are_not_independent_modalities": True,
                "file_access": "training_tensor_files_read_then_known_rows_sliced_before_transforms",
                "excluded_row_feature_transforms_or_model_forwards": False,
                "exact_capacity_or_compute_matching_claimed": False,
            })
        write_new_json(output / policy / "normalization.json", normalization)
        for condition in CONDITIONS:
            def progress(state):
                print(json.dumps({"policy": policy, "condition": condition, **state}), flush=True)

            model, fitted = train(*values, policies, condition, 42, config, progress)
            probabilities = predict(model, values[1], config["batch_size"])
            scores = selection_scores(values[1], probabilities, policies, 4)
            destination = output / policy / condition
            destination.mkdir(parents=True)
            checkpoint = destination / "selected_model.pt"
            identity = {
                "condition": condition, "policy": policy,
                "seed": 42, "configuration": config, "normalization": normalization,
                "protocol_sha256": sha256_file(output / "protocol.json"),
            }
            torch.save({"state_dict": model.state_dict(), **identity}, checkpoint)
            replay = make_model(values[0], config, condition)
            replay.load_state_dict(verified_checkpoint(checkpoint, identity), strict=True)
            replayed = predict(replay, values[1], config["batch_size"])
            difference = float(np.abs(replayed - probabilities).max())
            if difference > 1e-6 or not np.array_equal(replayed.argmax(1), probabilities.argmax(1)):
                raise ValueError("architecture checkpoint failed known-policy replay")
            prediction = destination / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[1], probabilities)
            record = {
                "policy": policy, "condition": condition, "seed": 42,
                "fit": fitted, "inner_selection": scores,
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": difference,
                "challenge_forward_or_scoring": False, "submission_allowed": False,
            }
            write_new_json(destination / "diagnostic.json", record)
            records.append(record)
            print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 12}),
                  flush=True)
        if len({row["fit"]["shared_initialization_digest"] for row in records
                if row["policy"] == policy}) != 1:
            raise ValueError("paired architecture shared initialization differs")
    if reference_files != {
        str(path): sha256_file(path) for path in (control_root / "Quadbot").rglob("*")
        if path.is_file()
    }:
        raise ValueError("reference evidence changed")
    write_new_json(output / "summary.json", {
        "scope": "one_seed_known_policy_architecture_feasibility",
        "dataset": "Quadbot", "new_fits": 12, "checkpoint_replays": 12,
        "records": records, "challenge_forward_or_scoring": False,
        "independent_confirmation": False, "architecture_novelty_established": False,
        "reference_files_unchanged": True,
        "submission_allowed": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.data_dir, args.temporal_dir, args.control_root, args.protocol, args.output)


if __name__ == "__main__":
    main()
