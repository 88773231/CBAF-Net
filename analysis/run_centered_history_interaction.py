"""One frozen six-fit check against nine retained exact-partition controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_compact_correspondence_stability import source_hashes as retained_sources
from analysis.run_encoder_objective_diagnostic import load_known_fold, write_predictions
from analysis.run_factor_training_screen import POLICIES
from src.botsscl_adapter import CorruptionAdapter
from src.botsscl_policy_control import predict_control
from src.centered_history_interaction import CONDITIONS as NEW, make_model, predict, train
from src.compact_correspondence import make_model as compact_model
from src.evidence_repair import flattened_inputs, write_new_json
from src.policy_robust import selection_scores
from src.provenance import sha256_file


CONDITIONS = (*NEW, "without_correspondence", "supervised_encoder", "botsscl_mechanism")


def source_hashes():
    root = Path(__file__).parents[1]
    return {**retained_sources(), **{name: sha256_file(root / name) for name in (
        "src/centered_history_interaction.py", "src/correspondence_dependence.py",
        "analysis/run_centered_history_interaction.py", "configs/centered_history_interaction.json")}}


def validate_settings(settings):
    expected = {
        "schema": "known-policy-centered-interaction-screen-v1", "dataset": "Quadbot",
        "partition_seed": 42, "training_seed": 42, "conditions": list(CONDITIONS),
        "configuration": {"block_dim": 64, "readout_dim": 128, "dropout": .2,
                          "learning_rate": .001, "weight_decay": .001,
                          "batch_size": 128, "max_epochs": 30, "patience": 6},
        "new_fits": 6, "reused_exact_partition_checkpoints": 9, "checkpoint_replays": 15,
        "cpu_threads": 2, "selection_population": "known_policy_inner_selection_only",
        "challenge_forward_or_scoring": False, "original_validation_or_test_access": False,
        "no_hyperparameter_search": True, "design_informed_by_prior_development": True,
        "minimum_module_mean_macro_f1_gain": .0025,
        "development_gate": "each_module_gain_at_least_0.0025_and_positive_in_two_folds_plus_positive_mean_and_two_folds_against_each_strong_control",
        "independent_confirmation": False, "algorithmic_novelty_established": False,
        "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("fixed interaction screen settings differ")


def decision(records, minimum_gain=.0025):
    expected = {(policy, condition) for policy in POLICIES for condition in CONDITIONS}
    identities = [(row["policy"], row["condition"]) for row in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicated interaction matrix")
    scores = {identity: row["inner_selection"]["all_profiles"]["macro_f1"]
              for identity, row in zip(identities, records)}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("invalid selection performance")
    full = np.array([scores[(policy, NEW[0])] for policy in POLICIES])
    comparisons = {}
    for condition in CONDITIONS[1:]:
        difference = full - np.array([scores[(policy, condition)] for policy in POLICIES])
        required = minimum_gain if condition in (NEW[1], "without_correspondence") else 0
        comparisons[condition] = {
            "paired_fold_differences": difference.tolist(), "mean_difference": float(difference.mean()),
            "positive_folds": int((difference > 0).sum()), "required_mean_gain": required,
            "gate_passed": bool(difference.mean() >= required and difference.mean() > 0
                                and (difference > 0).sum() >= 2),
        }
    return {
        "mean_macro_f1_by_condition": {
            condition: float(np.mean([scores[(policy, condition)] for policy in POLICIES]))
            for condition in CONDITIONS},
        "comparisons": comparisons,
        "development_continuation_gate_passed": all(row["gate_passed"] for row in comparisons.values()),
        "folds_are_overlapping_not_independent_replications": True,
        "single_training_seed_is_not_stability_confirmation": True,
        "threshold_is_not_a_ccfb_acceptance_standard": True,
        "independent_confirmation": False, "submission_allowed": False,
    }


def reuse(study, protocol, reference_summary, policy, condition, values, normalization):
    folder = study / policy / "seed42" / condition
    record = json.loads((folder / "diagnostic.json").read_text())
    expected = next(row for row in reference_summary["records"]
                    if row["policy"] == policy and row["seed"] == 42 and row["condition"] == condition)
    checkpoint = folder / "selected_model.pt"
    if record != expected or record["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("retained control evidence differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = {
        "condition": condition, "policy": policy, "seed": 42, "partition_seed": 42,
        "normalization": normalization, "protocol_sha256": sha256_file(study / "protocol.json"),
        "configuration": (protocol["configuration"] if condition == "without_correspondence"
                          else protocol["reference_configuration"]),
    }
    if condition != "without_correspondence":
        identity["input_dimension"] = flattened_inputs(values).shape[1]
    if (set(saved) != {"state_dict", *identity}
            or any(saved[key] != value for key, value in identity.items())):
        raise ValueError("retained control checkpoint envelope differs")
    model = (compact_model(values, identity["configuration"], condition)
             if condition == "without_correspondence"
             else CorruptionAdapter(identity["input_dimension"], 4,
                                    identity["configuration"]["mlp_hidden_dim"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, {**record["fit"], "new_fit": False,
                   "source_checkpoint_sha256": sha256_file(checkpoint)}, record["inner_selection"]


def run(data_dir, temporal, control_root, study, protocol_path, output):
    if output.exists():
        raise FileExistsError("do not overwrite interaction screen evidence")
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings)
    retained_protocol = json.loads((study / "protocol.json").read_text())
    retained_summary = json.loads((study / "summary.json").read_text())
    if (retained_protocol["source_hashes"] != retained_sources()
            or retained_protocol["configuration"] != settings["configuration"]
            or retained_summary["challenge_forward_or_scoring"] or retained_summary["submission_allowed"]):
        raise ValueError("retained reference protocol differs")
    records, unchanged = [], {}
    for policy in POLICIES:
        values, policies, normalization, _, input_hashes = load_known_fold(
            data_dir, temporal, control_root, policy)
        if input_hashes != retained_protocol["input_hashes"]:
            raise ValueError("original fitting inputs differ")
        if not records:
            write_new_json(output / "protocol.json", {
                **settings, "source_hashes": source_hashes(), "input_hashes": input_hashes,
                "retained_protocol_sha256": sha256_file(study / "protocol.json"),
                "excluded_row_feature_transforms_or_model_forwards": False,
                "raw_hashes_are_not_semantic_embeddings": True,
                "action_and_text_are_not_independent_modalities": True,
                "temporal_interaction_is_not_causal_evidence": True,
                "exact_capacity_or_compute_matching_claimed": False,
            })
        write_new_json(output / policy / "normalization.json", normalization)
        for condition in CONDITIONS:
            expected_scores = None
            if condition in NEW:
                def progress(state):
                    print(json.dumps({"policy": policy, "condition": condition, **state}), flush=True)
                model, fitted = train(*values, policies, condition, 42, settings["configuration"], progress)
            else:
                model, fitted, expected_scores = reuse(
                    study, retained_protocol, retained_summary, policy, condition, values[0], normalization)
                path = study / policy / "seed42" / condition / "selected_model.pt"
                unchanged[str(path)] = sha256_file(path)
            predictor = predict if condition in (*NEW, "without_correspondence") else predict_control
            probabilities = predictor(model, values[1], settings["configuration"]["batch_size"])
            scores = selection_scores(values[1], probabilities, policies, 4)
            if expected_scores is not None and scores != expected_scores:
                raise ValueError("retained control failed exact-partition replay")
            destination = output / policy / condition
            destination.mkdir(parents=True)
            checkpoint = destination / "selected_model.pt"
            saved = {"state_dict": model.state_dict(), "condition": condition, "policy": policy,
                     "seed": 42, "normalization": normalization,
                     "protocol_sha256": sha256_file(output / "protocol.json")}
            torch.save(saved, checkpoint)
            replay = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if set(replay) != set(saved) or any(replay[key] != value for key, value in saved.items()
                                               if key != "state_dict"):
                raise ValueError("interaction checkpoint envelope differs")
            if condition in NEW:
                replay_model = make_model(values[0], settings["configuration"], condition)
            elif condition == "without_correspondence":
                replay_model = compact_model(values[0], settings["configuration"], condition)
            else:
                replay_model = CorruptionAdapter(
                    flattened_inputs(values[0]).shape[1], 4,
                    retained_protocol["reference_configuration"]["mlp_hidden_dim"])
            replay_model.load_state_dict(replay["state_dict"], strict=True)
            replayed = predictor(replay_model, values[1], settings["configuration"]["batch_size"])
            error = float(np.abs(replayed - probabilities).max())
            if error > 1e-6:
                raise ValueError("interaction checkpoint predictions differ")
            prediction = destination / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[1], probabilities)
            records.append({
                "condition": condition, "policy": policy, "seed": 42, "fit": fitted,
                "inner_selection": scores, "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": error,
                "challenge_forward_or_scoring": False, "submission_allowed": False})
            write_new_json(destination / "diagnostic.json", records[-1])
            print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 15}),
                  flush=True)
        digests = {row["fit"]["shared_initialization_digest"] for row in records
                   if row["policy"] == policy and row["condition"] in (*NEW, "without_correspondence")}
        if len(digests) != 1:
            raise ValueError("shared active initialization differs")
    if any(sha256_file(Path(path)) != digest for path, digest in unchanged.items()):
        raise ValueError("retained control checkpoint changed")
    if (len(records), sum(row["fit"]["new_fit"] for row in records)) != (15, 6):
        raise ValueError("interaction fitting and replay counts differ")
    write_new_json(output / "summary.json", {
        "scope": "one_seed_known_policy_centered_interaction_screen", "dataset": "Quadbot",
        "records": records, "new_fits": 6, "reused_exact_partition_checkpoints": 9,
        "checkpoint_replays": 15,
        "decision": decision(records, settings["minimum_module_mean_macro_f1_gain"]),
        "reference_checkpoints_unchanged": True, "challenge_forward_or_scoring": False,
        "independent_confirmation": False, "algorithmic_novelty_established": False,
        "submission_allowed": False})


def main():
    parser = argparse.ArgumentParser()
    for name in ("data-dir", "temporal-dir", "control-root", "study", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.data_dir, args.temporal_dir, args.control_root, args.study, args.protocol, args.output)


if __name__ == "__main__":
    main()
