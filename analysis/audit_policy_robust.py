"""Replay saved challenge predictions from unchanged development checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition
from analysis.summarize_policy_robust import verified_records
from src.data_loader import fit_column_standardizer, fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, load_split, make_model, predict, write_new_json,
)
from src.provenance import sha256_file


def standardizer_payload(transform):
    return {
        name: transform[name].tolist()
        for name in ("mean", "std", "observed_std", "variable_mask")
    }


def audit(data_dir, temporal, master, root, out, dataset):
    if out.exists():
        raise FileExistsError("audit records may not be overwritten")
    frozen, records = verified_records(root, dataset)
    code = Path(__file__).resolve().parents[1]
    for relative, expected in frozen["source_hashes"].items():
        if sha256_file(code / relative) != expected:
            raise ValueError("frozen training implementation changed: " + relative)
    if sha256_file(master) != frozen["master_sha256"]:
        raise ValueError("main account archive changed")
    for relative, expected in frozen["input_hashes"].items():
        if sha256_file(data_dir / relative) != expected:
            raise ValueError("frozen input changed: " + relative)
    config, class_count = frozen["configuration"], frozen["class_count"]
    transforms = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", config["steps"], transforms, class_count)
    val = load_split(data_dir, "val", config["steps"], transforms, class_count)
    timing = attach_relative_timing(train, val, temporal)
    if timing != frozen["timing"]:
        raise ValueError("timing inputs changed")
    train_timing = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    val_timing = torch.load(temporal / "val_steps.pt", map_location="cpu", weights_only=True)
    policies, _ = collect_provenance(master)
    out.mkdir(parents=True)
    replay = []
    checkpoint_hashes = {}
    for seed in frozen["seeds"]:
        for heldout in sorted(set(policies.values())):
            fit_indices, select_indices = inner_profile_partition(
                train["metadata"], policies, heldout, seed
            )
            challenge_indices = [
                index for index, row in enumerate(val["metadata"])
                if policies[row["base_profile_id"]] == heldout
            ]
            fitting, selection, challenge = (
                subset(train, fit_indices), subset(train, select_indices),
                subset(val, challenge_indices),
            )
            assert_group_disjoint(fitting, selection)
            assert_group_disjoint(fitting, challenge)
            assert_group_disjoint(selection, challenge)
            if any(policies[row["base_profile_id"]] == heldout
                   for values in (fitting, selection) for row in values["metadata"]):
                raise ValueError("held-out policy leaked into fitting/selection")
            table_transform = fit_column_standardizer(fitting["raw_tabular"])
            step_transform = fit_column_standardizer(train_timing[fit_indices].flatten(0, 1))
            normalization = fit_fold_transforms(
                fitting, selection, challenge,
                (train_timing[fit_indices], train_timing[select_indices],
                 val_timing[challenge_indices]),
            )
            write_new_json(out / f"seed{seed}_{heldout}_normalization.json", {
                "fit_scope": "inner_fitting_profiles_only",
                "global_table": standardizer_payload(table_transform),
                "step_timing": standardizer_payload(step_transform),
                "clamp": [-5.0, 5.0],
                "description_text_action_hash_normalization":
                    "deterministic_per_row_preprocess_tensor_before_timing_append",
                "fitting_indices": fit_indices, "selection_indices": select_indices,
                "challenge_indices": challenge_indices,
                "test_access": False, "checkpoint_bytes_unchanged": True,
            })
            for method in frozen["methods"]:
                if method == "hgb_statistics":
                    continue
                path = root / dataset / f"seed{seed}" / heldout / method
                checkpoint_path = path / "selected_model.pt"
                checkpoint_hash = sha256_file(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if checkpoint["normalization"] != normalization:
                    raise ValueError("checkpoint normalization reconstruction differs")
                if checkpoint["configuration"] != config or checkpoint["method"] != method:
                    raise ValueError("checkpoint method/configuration differs")
                row = records[(seed, heldout, method)]
                model = make_model(challenge, "mlp_matched", class_count, config)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if sum(parameter.numel() for parameter in model.parameters()) != row["fit"]["parameter_count"]:
                    raise ValueError("parameter count differs")
                probabilities, _ = predict(model, challenge, "mlp_matched", config["batch_size"])
                saved = [
                    json.loads(line) for line in
                    (path / "challenge_predictions.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                saved_keys = [(entry["base_profile_id"], entry["label"]) for entry in saved]
                actual_keys = [(entry["base_profile_id"], entry["label"])
                               for entry in challenge["metadata"]]
                if saved_keys != actual_keys:
                    raise ValueError("prediction row order differs")
                expected = np.asarray([entry["probabilities"] for entry in saved])
                difference = float(np.max(np.abs(expected - probabilities)))
                if not np.allclose(expected, probabilities, atol=1e-6, rtol=0):
                    raise ValueError("checkpoint predictions cannot be replayed")
                if sha256_file(checkpoint_path) != checkpoint_hash:
                    raise ValueError("audit modified checkpoint")
                relative = checkpoint_path.relative_to(root).as_posix()
                checkpoint_hashes[relative] = checkpoint_hash
                replay.append({
                    "method": method, "seed": seed, "heldout_policy": heldout,
                    "maximum_probability_difference": difference,
                    "inference_requires_policy_metadata": False,
                })
    result = {
        "dataset": dataset, "checkpoint_replays": replay,
        "checkpoint_sha256": checkpoint_hashes, "all_replayed": True,
        "test_access": False, "submission_allowed": False,
        "main_account_data_unchanged": True,
        "normalization_sidecars_include_variable_masks": True,
        "audit_source_sha256": sha256_file(Path(__file__)),
    }
    write_new_json(out / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result = audit(
            args.data_dir, args.temporal_dir, args.master, args.root,
            args.out_dir, args.dataset,
        )
    print(json.dumps({"dataset": args.dataset, "all_replayed": result["all_replayed"],
                      "replayed_checkpoints": len(result["checkpoint_replays"])}))


if __name__ == "__main__":
    main()
