"""Fixed training seeds on the same known-policy partitions; no challenge forwards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_encoder_objective_diagnostic import load_known_fold, write_predictions
from analysis.run_factor_training_screen import POLICIES
from analysis.run_trajectory_correspondence_screen import source_hashes as legacy_sources
from src.botsscl_adapter import CorruptionAdapter
from src.botsscl_policy_control import predict_control, train_control
from src.compact_correspondence import CONDITIONS as COMPACT, make_model, predict, train
from src.evidence_repair import flattened_inputs, write_new_json
from src.provenance import sha256_file
from src.policy_robust import selection_scores


CONDITIONS = (*COMPACT, "supervised_encoder", "botsscl_mechanism")
SEEDS = (42, 43, 44)
CONTROL_METHODS = {
    "supervised_encoder": "botsscl_supervised_only", "botsscl_mechanism": "botsscl_flattened",
}
SOURCES = (
    "src/compact_correspondence.py", "analysis/run_compact_correspondence_stability.py",
    "configs/compact_correspondence_stability.json",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**legacy_sources(), **{name: sha256_file(root / name) for name in SOURCES}}


def validate_settings(settings):
    expected = {
        "schema": "known-policy-compact-correspondence-stability-v1",
        "dataset": "Quadbot", "partition_seed": 42, "training_seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "configuration": {
            "block_dim": 64, "readout_dim": 128, "dropout": .2, "learning_rate": .001,
            "weight_decay": .001, "batch_size": 128, "max_epochs": 30, "patience": 6,
        },
        "pretraining": {"epochs": 30, "corruption_rate": .6, "temperature": 1.0},
        "new_fits": 30, "reused_reference_checkpoints": 6, "checkpoint_replays": 36,
        "cpu_threads": 2, "selection_population": "known_policy_inner_selection_only",
        "challenge_forward_or_scoring": False, "original_validation_or_test_access": False,
        "no_hyperparameter_search": True, "design_informed_by_prior_development": True,
        "development_gate": "positive_mean_and_at_least_two_positive_seed_means_against_each_control",
        "independent_confirmation": False, "algorithmic_novelty_established": False,
        "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("fixed compact stability settings or evidence boundary differs")


def cached_control(control_root, policy, condition, normalization, config, values, pretraining):
    method = CONTROL_METHODS[condition]
    folder = control_root / "Quadbot" / "seed42" / policy / method
    record = json.loads((folder / "metrics.json").read_text())
    checkpoint = folder / "selected_model.pt"
    identity = {"dataset": "Quadbot", "seed": 42, "heldout_policy": policy, "method": method}
    if (any(record.get(key) != value for key, value in identity.items())
            or record["checkpoint_sha256"] != sha256_file(checkpoint)):
        raise ValueError("cached control record or checkpoint identity differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        **identity, "configuration": config, "normalization": normalization,
        "protocol_sha256": sha256_file(control_root / "Quadbot/protocol.json"),
        "input_dimension": flattened_inputs(values).shape[1],
    }
    if (any(saved.get(key) != value for key, value in expected.items())
            or saved.get("pretraining", pretraining) != pretraining):
        raise ValueError("cached control checkpoint envelope differs")
    model = CorruptionAdapter(expected["input_dimension"], 4, config["mlp_hidden_dim"])
    model.load_state_dict(saved["state_dict"], strict=True)
    fitted = {
        **record["fit"], "new_fit": False,
        "source_checkpoint_sha256": sha256_file(checkpoint),
    }
    selected = next(row["inner_selection"] for row in fitted["history"]
                    if row["epoch"] == fitted["best_epoch"])
    return model, fitted, selected


def decision(records):
    expected = {(seed, policy, condition)
                for seed in SEEDS for policy in POLICIES for condition in CONDITIONS}
    identities = [(row["seed"], row["policy"], row["condition"]) for row in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("stability comparison is incomplete or duplicated")
    scores = {key: row["inner_selection"]["all_profiles"]["macro_f1"]
              for key, row in zip(identities, records)}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("invalid inner-selection score")
    by_seed = {
        condition: [float(np.mean([scores[(seed, policy, condition)] for policy in POLICIES]))
                    for seed in SEEDS]
        for condition in CONDITIONS
    }
    full = np.array(by_seed[CONDITIONS[0]])
    comparisons = {}
    for condition in CONDITIONS[1:]:
        difference = full - np.array(by_seed[condition])
        comparisons[condition] = {
            "paired_seed_mean_differences": difference.tolist(),
            "mean_difference": float(difference.mean()),
            "positive_seed_means": int((difference > 0).sum()),
            "gate_passed": bool(difference.mean() > 0 and (difference > 0).sum() >= 2),
        }
    return {
        "means_by_condition": {
            condition: {
                "values_by_training_seed": values, "mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)),
            } for condition, values in by_seed.items()
        },
        "comparisons": comparisons,
        "development_continuation_gate_passed": all(row["gate_passed"] for row in comparisons.values()),
        "folds_are_overlapping_not_independent_replications": True,
        "training_seed_repetition_is_not_independent_data_confirmation": True,
        "inner_selection_is_not_test_performance": True,
        "independent_confirmation": False, "submission_allowed": False,
    }


def run(data_dir, temporal, control_root, protocol_path, output):
    if output.exists():
        raise FileExistsError("stability evidence must not be overwritten")
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings)
    config, records = settings["configuration"], []
    reference_hashes = {
        str(path): sha256_file(path) for path in (control_root / "Quadbot").rglob("*")
        if path.is_file()
    }
    if not reference_hashes:
        raise ValueError("control evidence missing")
    new_fits, reused = 0, 0
    for policy in POLICIES:
        values, policies, normalization, reference, inputs = load_known_fold(
            data_dir, temporal, control_root, policy,
        )
        control_config = reference["configuration"]
        common = ("learning_rate", "weight_decay", "batch_size", "max_epochs", "patience")
        if (any(control_config[key] != config[key] for key in common)
                or control_config["mlp_hidden_dim"] != 128
                or reference.get("pretraining", settings["pretraining"]) != settings["pretraining"]):
            raise ValueError("control supervised budget or pretraining differs")
        if not records:
            write_new_json(output / "protocol.json", {
                **settings, "source_hashes": source_hashes(), "input_hashes": inputs,
                "reference_protocol_sha256": sha256_file(control_root / "Quadbot/protocol.json"),
                "reference_configuration": control_config,
                "file_access": "training_tensor_files_read_then_known_rows_sliced_before_transforms",
                "excluded_row_feature_transforms_or_model_forwards": False,
                "exact_capacity_or_compute_matching_claimed": False,
                "raw_hashes_are_not_semantic_embeddings": True,
                "action_and_text_are_not_independent_modalities": True,
                "historical_attention_is_not_causal_evidence": True,
                "cached_checkpoint_challenge_fields_used_for_model_selection": False,
            })
        write_new_json(output / policy / "normalization.json", normalization)
        for seed in SEEDS:
            for condition in CONDITIONS:
                def progress(state):
                    print(json.dumps({
                        "policy": policy, "seed": seed, "condition": condition, **state,
                    }), flush=True)

                expected_scores = None
                compact = condition in COMPACT
                if compact:
                    model, fitted = train(*values, policies, condition, seed, config, progress)
                    fitted["new_fit"] = True
                elif seed == 42:
                    model, fitted, expected_scores = cached_control(
                        control_root, policy, condition, normalization, control_config,
                        values[0], settings["pretraining"],
                    )
                    reused += 1
                else:
                    model, fitted = train_control(
                        *values, policies, CONTROL_METHODS[condition], 4, seed, control_config,
                        settings["pretraining"], progress,
                    )
                    fitted["new_fit"] = True
                new_fits += int(fitted["new_fit"])
                predictor = predict if compact else predict_control
                probabilities = predictor(model, values[1], config["batch_size"])
                scores = selection_scores(values[1], probabilities, policies, 4)
                if expected_scores is not None and scores != expected_scores:
                    raise ValueError("cached control failed known-policy selection replay")
                destination = output / policy / f"seed{seed}" / condition
                destination.mkdir(parents=True)
                checkpoint = destination / "selected_model.pt"
                identity = {
                    "condition": condition, "policy": policy, "seed": seed, "partition_seed": 42,
                    "configuration": config if compact else control_config,
                    "normalization": normalization,
                    "protocol_sha256": sha256_file(output / "protocol.json"),
                }
                if not compact:
                    identity["input_dimension"] = flattened_inputs(values[0]).shape[1]
                torch.save({"state_dict": model.state_dict(), **identity}, checkpoint)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (set(saved) != {"state_dict", *identity}
                        or any(saved[key] != value for key, value in identity.items())):
                    raise ValueError("stability checkpoint envelope differs")
                replay = (make_model(values[0], config, condition) if compact else
                          CorruptionAdapter(identity["input_dimension"], 4, control_config["mlp_hidden_dim"]))
                replay.load_state_dict(saved["state_dict"], strict=True)
                replayed = predictor(replay, values[1], config["batch_size"])
                difference = float(np.abs(replayed - probabilities).max())
                if difference > 1e-6 or not np.array_equal(replayed.argmax(1), probabilities.argmax(1)):
                    raise ValueError("stability checkpoint replay differs")
                prediction = destination / "inner_selection_predictions.jsonl"
                write_predictions(prediction, values[1], probabilities)
                record = {
                    "policy": policy, "condition": condition, "seed": seed, "partition_seed": 42,
                    "fit": fitted, "inner_selection": scores,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "predictions_sha256": sha256_file(prediction),
                    "maximum_replay_probability_difference": difference,
                    "challenge_forward_or_scoring": False, "submission_allowed": False,
                }
                write_new_json(destination / "diagnostic.json", record)
                records.append(record)
                print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 36}),
                      flush=True)
            digests = {row["fit"]["shared_initialization_digest"] for row in records
                       if row["policy"] == policy and row["seed"] == seed and row["condition"] in COMPACT}
            if len(digests) != 1:
                raise ValueError("paired compact shared initialization differs")
    if reference_hashes != {
        str(path): sha256_file(path) for path in (control_root / "Quadbot").rglob("*") if path.is_file()
    }:
        raise ValueError("reference evidence changed")
    if (new_fits, reused, len(records)) != (30, 6, 36):
        raise ValueError("stability fit/reuse counts differ")
    write_new_json(output / "summary.json", {
        "scope": "fixed_partition_three_training_seed_known_policy_stability", "dataset": "Quadbot",
        "new_fits": new_fits, "reused_reference_checkpoints": reused, "checkpoint_replays": 36,
        "records": records, "decision": decision(records), "reference_files_unchanged": True,
        "challenge_forward_or_scoring": False, "independent_confirmation": False,
        "architecture_novelty_established": False, "submission_allowed": False,
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("data-dir", "temporal-dir", "control-root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.data_dir, args.temporal_dir, args.control_root, args.protocol, args.output)


if __name__ == "__main__":
    main()
