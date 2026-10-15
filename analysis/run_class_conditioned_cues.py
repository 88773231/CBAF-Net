"""Frozen class-conditioned cue check on previously exposed training folds."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_observable_evolution import audit as audit_reference
from analysis.run_encoder_objective_diagnostic import write_predictions
from analysis.run_observable_evolution import (
    CONFIGURATION as REFERENCE_CONFIGURATION, METADATA_SHA256, POLICIES, SOURCE_SHA256, prepare_fold,
)
from src.class_conditioned_cues import CONDITIONS as NEW_CONDITIONS, make_model, train
from src.evidence_repair import write_new_json
from src.observable_evolution import make_model as reference_model, predict
from src.policy_robust import selection_scores
from src.provenance import sha256_file


CACHED_CONDITIONS = {"clean_backbone_control": "without_both", "same_input_mlp": "same_input_mlp"}
CONDITIONS = (*NEW_CONDITIONS, *CACHED_CONDITIONS)
REFERENCE_SHA256 = "f884060fa7090d848abd5354fa7eafe4c1fc63f12f2234f246505091506b7c95"
CONFIGURATION = {
    "seed": 42, "width": 32, "readout": 128, "dropout": .2, "learning_rate": .001,
    "weight_decay": .001, "batch_size": 128, "max_epochs": 30, "patience": 6,
    "cue_supervision_weight": .1, "pair_margin_weight": .1, "pair_margin": 1.,
}


def settings():
    return {
        "schema": "known-policy-class-conditioned-cues-v1", "dataset": "Quadbot",
        "conditions": list(CONDITIONS), "configuration": CONFIGURATION,
        "new_fits": 12, "checkpoint_replays": 18, "cached_exact_partition_controls": 6,
        "minimum_module_mean_macro_f1_gain": .0025,
        "same_original_observation_inputs_and_partitions": True,
        "ordinary_scalar_gate_control_required": True, "no_hyperparameter_search": True,
        "pair_margin_is_training_objective_not_architectural_module": True,
        "previously_exposed_known_policy_selection_only": True,
        "original_validation_test_or_excluded_policy_forwards": False,
        "compute_and_parameter_matching_claimed": False,
        "algorithmic_novelty_established": False, "independent_confirmation": False,
        "gpu_required": False, "submission_allowed": False,
    }


def validate_settings(configuration):
    if configuration != settings():
        raise ValueError("frozen class-conditioned cue settings differ")


def decision(records):
    expected = {(policy, condition) for policy in POLICIES for condition in CONDITIONS}
    identities = [(row["policy"], row["condition"]) for row in records]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("incomplete or duplicate class-conditioned cue matrix")
    scores = {identity: row["inner_selection"]["all_profiles"]["macro_f1"]
              for identity, row in zip(identities, records)}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("invalid class-conditioned cue scores")
    full = np.asarray([scores[(policy, CONDITIONS[0])] for policy in POLICIES])
    comparisons = {}
    for condition in CONDITIONS[1:]:
        difference = full - np.asarray([scores[(policy, condition)] for policy in POLICIES])
        minimum = .0025 if condition in ("without_pair_margin", "without_class_routing") else 0
        comparisons[condition] = {
            "paired_fold_differences": difference.tolist(), "mean_difference": float(difference.mean()),
            "positive_folds": int((difference > 0).sum()), "required_mean_gain": minimum,
            "gate_passed": bool(difference.mean() >= minimum and difference.mean() > 0
                                and (difference > 0).sum() >= 2),
        }
    return {
        "mean_macro_f1_by_condition": {
            condition: float(np.mean([scores[(policy, condition)] for policy in POLICIES]))
            for condition in CONDITIONS},
        "comparisons": comparisons,
        "development_continuation_gate_passed": all(row["gate_passed"] for row in comparisons.values()),
        "folds_overlap_and_selection_was_previously_exposed": True,
        "single_seed_not_stability_confirmation": True, "threshold_is_not_a_ccfb_standard": True,
        "legacy_botdmm_scores_are_not_matched_comparisons": True,
        "algorithmic_novelty_established": False, "independent_confirmation": False,
        "submission_allowed": False,
    }


def run(source, metadata, partitions, reference, protocol, output):
    if output.exists():
        raise FileExistsError("preserve class-conditioned cue study")
    frozen = json.loads(protocol.read_text())
    validate_settings(frozen)
    audit_reference(reference)
    if sha256_file(reference / "summary.json") != REFERENCE_SHA256:
        raise ValueError("class-conditioned reference changed")
    receipt_paths = {policy: partitions / policy / "normalization.json" for policy in POLICIES}
    receipt_hashes = {policy: sha256_file(path) for policy, path in receipt_paths.items()}
    receipts = {policy: json.loads(path.read_text()) for policy, path in receipt_paths.items()}
    root = Path(__file__).resolve().parents[1]
    sources = ("src/class_conditioned_cues.py", "analysis/run_class_conditioned_cues.py",
               "analysis/audit_class_conditioned_cues.py", "configs/class_conditioned_cues.json")
    write_new_json(output / "protocol.json", {
        **frozen, "source_hashes": {name: sha256_file(root / name) for name in sources},
        "input_hashes": {"training_source": SOURCE_SHA256, "training_metadata": METADATA_SHA256},
        "reference_summary_sha256": REFERENCE_SHA256, "prior_partition_hashes": receipt_hashes,
        "no_independent_modality_assumption": True,
        "lexical_hashes_not_semantic_embeddings": True,
        "not_faithful_botchf_ecml_dbf_cdrbot_or_shortcut_reproduction": True,
    })
    records = []
    for policy in POLICIES:
        print(json.dumps({"event": "extracting_known_observations", "policy": policy}), flush=True)
        values, policies, normalization = prepare_fold(source, metadata, receipts, policy)
        expected_normalizer = json.loads((reference / policy / "normalization.json").read_text())
        if (normalization != expected_normalizer["transforms"]
                or receipt_hashes[policy] != expected_normalizer["partition_sha256"]):
            raise ValueError("reference matching inputs or normalization differs")
        write_new_json(output / policy / "normalization.json", expected_normalizer)
        for condition in CONDITIONS:
            folder = output / policy / condition
            folder.mkdir(parents=True)
            checkpoint = folder / "selected_model.pt"
            if condition in CACHED_CONDITIONS:
                prior = reference / policy / CACHED_CONDITIONS[condition]
                saved = torch.load(prior / "selected_model.pt", map_location="cpu", weights_only=True)
                if (saved["condition"] != CACHED_CONDITIONS[condition] or saved["policy"] != policy
                        or saved["configuration"] != REFERENCE_CONFIGURATION
                        or saved["normalization"] != normalization
                        or saved["protocol_sha256"] != sha256_file(reference / "protocol.json")):
                    raise ValueError("cached class-conditioned control differs")
                model = reference_model(REFERENCE_CONFIGURATION, CACHED_CONDITIONS[condition])
                model.load_state_dict(saved["state_dict"], strict=True)
                probabilities = predict(model, values[1], CONFIGURATION["batch_size"])
                previous = [json.loads(line) for line in
                            (prior / "inner_selection_predictions.jsonl").read_text().splitlines()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in previous] != [
                    (row["sample_id"], row["base_profile_id"], row["label"]) for row in values[1]["metadata"]
                ]:
                    raise ValueError("cached class-conditioned prediction identities differ")
                replay_error = float(np.abs(probabilities - np.asarray([
                    row["probabilities"] for row in previous])).max())
                shutil.copy2(prior / "selected_model.pt", checkpoint)
                fit = {"new_fit": False, "reference_condition": CACHED_CONDITIONS[condition],
                       "reference_summary_sha256": REFERENCE_SHA256}
            else:
                def progress(event):
                    print(json.dumps({"policy": policy, "condition": condition, **event}), flush=True)
                model, fit = train(*values, policies, condition, CONFIGURATION, progress)
                probabilities = predict(model, values[1], CONFIGURATION["batch_size"])
                torch.save({
                    "state_dict": model.state_dict(), "condition": condition, "policy": policy,
                    "configuration": CONFIGURATION, "normalization": normalization,
                    "protocol_sha256": sha256_file(output / "protocol.json"),
                }, checkpoint)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (saved["condition"] != condition or saved["policy"] != policy
                        or saved["configuration"] != CONFIGURATION
                        or saved["normalization"] != normalization
                        or saved["protocol_sha256"] != sha256_file(output / "protocol.json")):
                    raise ValueError("class-conditioned checkpoint envelope differs")
                replay = make_model(CONFIGURATION, condition)
                replay.load_state_dict(saved["state_dict"], strict=True)
                replay_error = float(np.abs(probabilities - predict(
                    replay, values[1], CONFIGURATION["batch_size"])).max())
                fit["new_fit"] = True
            if replay_error > 1e-6:
                raise ValueError("class-conditioned checkpoint replay differs")
            prediction = folder / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[1], probabilities)
            record = {
                "policy": policy, "condition": condition, "fit": fit,
                "inner_selection": selection_scores(values[1], probabilities, policies, 4),
                "checkpoint_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": replay_error,
                "original_validation_test_or_excluded_policy_forwards": False, "submission_allowed": False,
            }
            records.append(record)
            write_new_json(folder / "diagnostic.json", record)
            print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 18}), flush=True)
        digests = {row["fit"]["shared_backbone_initialization_digest"] for row in records
                   if row["policy"] == policy and row["fit"]["new_fit"]}
        if len(digests) != 1:
            raise ValueError("shared cue backbone initialization differs")
    if any(sha256_file(path) != receipt_hashes[policy] for policy, path in receipt_paths.items()):
        raise ValueError("frozen class-conditioned partitions changed")
    write_new_json(output / "summary.json", {
        "protocol_sha256": sha256_file(output / "protocol.json"), "records": records, **decision(records),
        "new_fits_completed": 12, "checkpoint_replays_completed": 18, "cached_controls_replayed": 6,
        "gpu_used": False, "new_account_generation": False, "original_validation_or_test_access": False,
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "metadata", "partitions", "reference", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.source, args.metadata, args.partitions, args.reference, args.protocol, args.output)


if __name__ == "__main__":
    main()
