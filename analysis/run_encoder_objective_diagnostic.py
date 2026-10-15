"""Reuse known-policy partitions; never construct a challenge prediction path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_botsscl_policy_control import source_hashes as control_sources
from analysis.run_factor_training_screen import POLICIES, policies_from_partitions
from src.botsscl_adapter import CorruptionAdapter
from src.botsscl_policy_control import predict_control
from src.data_loader import apply_column_standardizer, fit_column_standardizer, preprocess_tensor
from src.encoder_objective_diagnostic import (
    CONDITIONS, gradient_probe, train_objective, validate_populations,
)
from src.evidence_repair import flattened_inputs, input_files, write_new_json
from src.policy_robust import profile_batches, selection_scores
from src.provenance import sha256_file


SOURCES = (
    "src/encoder_objective_diagnostic.py", "analysis/run_encoder_objective_diagnostic.py",
    "configs/encoder_objective_diagnostic.json",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {**control_sources(), **{name: sha256_file(root / name) for name in SOURCES}}


def validate_settings(settings):
    expected = {
        "schema": "known-policy-encoder-objective-diagnostic-v1", "dataset": "Quadbot",
        "seeds": [42], "conditions": list(CONDITIONS),
        "regularization": {"contrast_weight": 0.1, "temperature": 0.2, "sam_radius": 0.05},
        "new_fits": 9, "reused_reference_checkpoints": 3, "cpu_threads": 2,
        "selection_population": "known_policy_inner_selection_only",
        "challenge_forward_or_scoring": False, "original_validation_or_test_access": False,
        "independent_confirmation": False, "no_hyperparameter_search": True,
        "submission_allowed": False,
    }
    if settings != expected:
        raise ValueError("fixed diagnostic settings or access boundary changed")


def load_known_fold(data_dir, temporal, control_root, policy):
    reference = control_root / "Quadbot"
    frozen = json.loads((reference / "protocol.json").read_text())
    if frozen["source_hashes"] != control_sources():
        raise ValueError("control training sources changed")
    if frozen["original_val_paths_opened"] or frozen["test_paths_opened"]:
        raise ValueError("reference is not training-only")
    paths = [
        *input_files(data_dir, "train", 5), data_dir / "graph/train_node_features.pt",
        *[temporal / name for name in ("train_metadata.jsonl", "train_global.pt", "train_steps.pt")],
    ]
    hashes = {str(path): sha256_file(path) for path in paths}
    if hashes != frozen["input_hashes"]:
        raise ValueError("original training inputs differ from the reference")
    rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    timing_rows = [json.loads(line) for line in (temporal / "train_metadata.jsonl").read_text().splitlines()]
    if rows != timing_rows:
        raise ValueError("training timing identity differs")
    receipts = {
        name: json.loads((reference / "seed42" / name / "normalization.json").read_text())
        for name in POLICIES
    }
    policies = policies_from_partitions(rows, receipts)
    normalization = receipts[policy]
    indices = [normalization["fitting_indices"], normalization["selection_indices"]]
    known = set(indices[0] + indices[1])
    if (len(known) != sum(map(len, indices)) or not known
            or any(policies[rows[index]["base_profile_id"]] == policy for index in known)):
        raise ValueError("known-policy indices overlap or contain excluded rows")
    excluded = set(range(len(rows))) - known
    if excluded != set(normalization["challenge_indices"]):
        raise ValueError("reference exclusion identity differs")
    if any(policies[rows[index]["base_profile_id"]] != policy for index in excluded):
        raise ValueError("reference policy exclusion differs")
    values = [{"metadata": [rows[index] for index in part]} for part in indices]

    def selected_tensor(path, trailing_shape=None):
        # The serialized training file contains all rows; slice before transforms.
        full = torch.load(path, map_location="cpu", weights_only=True)
        if full.shape[0] != len(rows):
            raise ValueError("training tensor row count differs")
        selected = [full[part].clone() for part in indices]
        if trailing_shape is not None and tuple(full.shape[1:]) != trailing_shape:
            raise ValueError("training tensor shape differs")
        del full
        if any(not torch.isfinite(block).all() for block in selected):
            raise ValueError("nonfinite known-policy tensor")
        return selected

    labels = selected_tensor(paths[1], ())
    for population, label in zip(values, labels):
        population["labels"] = label.long()
    for population, block in zip(values, selected_tensor(paths[2])):
        population["description"] = preprocess_tensor(block.float())
    numerical, style = selected_tensor(paths[3], (6,)), selected_tensor(paths[4], (11,))
    global_timing = selected_tensor(temporal / "train_global.pt", (7,))
    raw_steps = selected_tensor(temporal / "train_steps.pt", (5, 7))
    for index, population in enumerate(values):
        population["raw_tabular"] = torch.cat([
            numerical[index].float(), style[index].float(), global_timing[index].float(),
        ], dim=1)
    for kind, offset in (("text", 5), ("action", 6)):
        blocks = [selected_tensor(paths[offset + step * 2]) for step in range(5)]
        for index, population in enumerate(values):
            population[kind] = torch.stack([
                preprocess_tensor(block[index].float()) for block in blocks
            ], dim=1)
    table_transform = fit_column_standardizer(values[0]["raw_tabular"])
    step_transform = fit_column_standardizer(raw_steps[0].float().flatten(0, 1))
    computed = {
        "fit_scope": "inner_fitting_profiles_only",
        "table_mean": table_transform["mean"].tolist(), "table_std": table_transform["std"].tolist(),
        "step_mean": step_transform["mean"].tolist(), "step_std": step_transform["std"].tolist(),
    }
    if any(normalization[name] != value for name, value in computed.items()):
        raise ValueError("known-policy normalization differs from the cached control")
    for population, steps in zip(values, raw_steps):
        population["tabular"] = apply_column_standardizer(population["raw_tabular"], table_transform)
        scaled_steps = apply_column_standardizer(
            steps.float().flatten(0, 1), step_transform,
        ).reshape_as(steps)
        population["action"] = torch.cat([population["action"], scaled_steps], dim=2)
    validate_populations(*values, policies, 4)
    return values, policies, normalization, frozen, hashes


def write_predictions(path, population, probabilities):
    with path.open("x", encoding="utf-8") as stream:
        for row, probability in zip(population["metadata"], probabilities):
            stream.write(json.dumps({
                "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                "label": int(row["label"]), "probabilities": probability.tolist(),
            }) + "\n")


def run(data_dir, temporal, control_root, protocol_path, output):
    if output.exists():
        raise FileExistsError("diagnostic evidence must not be overwritten")
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings)
    records, control_hashes = [], {}
    for policy in POLICIES:
        values, policies, normalization, reference, inputs = load_known_fold(
            data_dir, temporal, control_root, policy,
        )
        config = reference["configuration"]
        source_folder = control_root / "Quadbot" / "seed42" / policy / "botsscl_supervised_only"
        source_record = json.loads((source_folder / "metrics.json").read_text())
        checkpoint = source_folder / "selected_model.pt"
        if (source_record["checkpoint_sha256"] != sha256_file(checkpoint)
                or source_record["seed"] != 42 or source_record["dataset"] != "Quadbot"
                or source_record["heldout_policy"] != policy
                or source_record["method"] != "botsscl_supervised_only"):
            raise ValueError("cached supervised reference identity changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["configuration"] != config or saved["normalization"] != normalization
                or saved["protocol_sha256"] != sha256_file(control_root / "Quadbot" / "protocol.json")
                or any(saved[key] != source_record[key] for key in
                       ("dataset", "seed", "heldout_policy", "method"))):
            raise ValueError("cached checkpoint envelope differs")
        source_selection = next(
            entry["inner_selection"] for entry in source_record["fit"]["history"]
            if entry["epoch"] == source_record["fit"]["best_epoch"]
        )
        folder = output / policy
        write_new_json(folder / "normalization.json", normalization)
        if len(records) == 0:
            write_new_json(output / "protocol.json", {
                **settings, "source_hashes": source_hashes(), "reference_protocol": reference,
                "reference_protocol_sha256": saved["protocol_sha256"],
                "input_hashes": inputs,
                "file_access": "training_tensor_files_read_then_known_rows_sliced_before_transforms",
                "excluded_row_feature_transforms_or_model_forwards": False,
                "cached_checkpoint_challenge_fields_used_for_model_selection": False,
            })
        control_hashes[str(checkpoint)] = sha256_file(checkpoint)
        control_hashes[str(source_folder / "metrics.json")] = sha256_file(source_folder / "metrics.json")
        for condition in CONDITIONS:
            destination = folder / condition
            if condition == "supervised_reference":
                model = CorruptionAdapter(saved["input_dimension"], 4, config["mlp_hidden_dim"])
                model.load_state_dict(saved["state_dict"], strict=True)
                model.projection.requires_grad_(False)
                fitted = {
                    "condition": condition, "reused_reference": True,
                    "best_epoch": source_record["fit"]["best_epoch"],
                    "inference_parameters": source_record["fit"]["inference_parameter_count"],
                    "source_checkpoint_sha256": source_record["checkpoint_sha256"],
                    "new_fit": False, "pretraining_used": False,
                }
                batch = next(profile_batches(
                    values[0]["metadata"], 4, config["batch_size"],
                    torch.Generator().manual_seed(42),
                ))
                known_policies = sorted({policies[row["base_profile_id"]] for row in values[0]["metadata"]})
                ids = torch.tensor([
                    known_policies.index(policies[row["base_profile_id"]])
                    for row in values[0]["metadata"]
                ])
                fitted["selected_probe"] = gradient_probe(
                    model, torch.from_numpy(flattened_inputs(values[0]))[batch],
                    values[0]["labels"][batch], ids[batch], settings["regularization"],
                )
            else:
                def progress(state):
                    print(json.dumps({"policy": policy, "condition": condition, **state}), flush=True)
                model, fitted = train_objective(
                    *values, policies, condition, 4, 42, config, settings["regularization"], progress,
                )
                fitted["new_fit"] = True
            probabilities = predict_control(model, values[1], config["batch_size"])
            scores = selection_scores(values[1], probabilities, policies, 4)
            if condition == "supervised_reference" and scores != source_selection:
                raise ValueError("cached reference failed known-policy selection replay")
            destination.mkdir(parents=True)
            selected_model = destination / "selected_model.pt"
            torch.save({"state_dict": model.state_dict(), "input_dimension": flattened_inputs(values[0]).shape[1],
                        "condition": condition, "policy": policy, "configuration": config,
                        "normalization": normalization,
                        "protocol_sha256": sha256_file(output / "protocol.json")}, selected_model)
            replay_model = CorruptionAdapter(saved["input_dimension"], 4, config["mlp_hidden_dim"])
            replay_model.load_state_dict(torch.load(
                selected_model, map_location="cpu", weights_only=True,
            )["state_dict"], strict=True)
            replayed = predict_control(replay_model, values[1], config["batch_size"])
            difference = float(np.abs(replayed - probabilities).max())
            if difference > 1e-6:
                raise ValueError("known-policy checkpoint replay differs")
            prediction = destination / "inner_selection_predictions.jsonl"
            write_predictions(prediction, values[1], probabilities)
            record = {
                "policy": policy, "condition": condition, "seed": 42, "fit": fitted,
                "inner_selection": scores, "checkpoint_sha256": sha256_file(selected_model),
                "predictions_sha256": sha256_file(prediction),
                "maximum_replay_probability_difference": difference,
                "challenge_forward_or_scoring": False, "submission_allowed": False,
            }
            write_new_json(destination / "diagnostic.json", record)
            records.append(record)
            print(json.dumps({"event": "cell_complete", "completed": len(records), "planned": 12}), flush=True)
        initializations = {row["fit"]["initialization_digest"] for row in records
                           if row["policy"] == policy and row["condition"] != "supervised_reference"}
        if len(initializations) != 1:
            raise ValueError("paired objective initialization differs")
    if any(sha256_file(Path(path)) != value for path, value in control_hashes.items()):
        raise ValueError("cached control evidence changed")
    write_new_json(output / "summary.json", {
        "scope": "one_seed_known_policy_objective_diagnostic", "dataset": "Quadbot",
        "new_fits": 9, "reused_reference_checkpoints": 3, "inner_selection_replays": 12,
        "records": records, "reference_files_unchanged": True,
        "challenge_forward_or_scoring": False, "independent_confirmation": False,
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
