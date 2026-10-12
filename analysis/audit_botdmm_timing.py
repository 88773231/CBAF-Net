"""Replay every fixed BotDMM timing comparison checkpoint without test access."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from analysis.audit_policy_robust import standardizer_payload
from analysis.run_botdmm_timing import graph_inputs
from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition
from analysis.summarize_policy_robust import verified_records
from src.botdmm_development import DevelopmentGraphView, make_backbone, predict_backbone
from src.data_loader import fit_column_standardizer, fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, load_split, write_new_json,
)
from src.provenance import sha256_file


def audit(data_dir, temporal, master, root, out, dataset):
    if out.exists():
        raise FileExistsError("audit records may not be overwritten")
    frozen, records = verified_records(root, dataset)
    code = Path(__file__).resolve().parents[1]
    for relative, expected in frozen["source_hashes"].items():
        if sha256_file(code / relative) != expected:
            raise ValueError("training implementation changed: " + relative)
    if sha256_file(master) != frozen["master_sha256"]:
        raise ValueError("account archive changed")
    for relative, expected in {**frozen["input_hashes"], **frozen["graph_input_hashes"]}.items():
        if sha256_file(data_dir / relative) != expected:
            raise ValueError("frozen input changed: " + relative)
    config, class_count = frozen["configuration"], frozen["class_count"]
    transforms = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", 5, transforms, class_count)
    val = load_split(data_dir, "val", 5, transforms, class_count)
    assert_group_disjoint(train, val)
    if attach_relative_timing(train, val, temporal) != frozen["timing"]:
        raise ValueError("timing sidecars changed")
    train_steps = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    val_steps = torch.load(temporal / "val_steps.pt", map_location="cpu", weights_only=True)
    train_graph, _ = graph_inputs(data_dir, "train", len(train["labels"]))
    val_graph, _ = graph_inputs(data_dir, "val", len(val["labels"]))
    policies, _ = collect_provenance(master)
    replays = []
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
            for left, right in ((fitting, selection), (fitting, challenge), (selection, challenge)):
                assert_group_disjoint(left, right)
            table_transform = fit_column_standardizer(fitting["raw_tabular"])
            step_transform = fit_column_standardizer(train_steps[fit_indices].flatten(0, 1))
            fit_fold_transforms(fitting, selection, challenge, (
                train_steps[fit_indices], train_steps[select_indices], val_steps[challenge_indices],
            ))
            normalization = {
                "global_table": standardizer_payload(table_transform),
                "step_timing": standardizer_payload(step_transform), "clamp": [-5.0, 5.0],
                "fitting_indices": fit_indices, "selection_indices": select_indices,
                "challenge_indices": challenge_indices, "fit_scope": "inner_fitting_profiles_only",
            }
            fold = root / dataset / f"seed{seed}" / heldout
            if json.loads((fold / "normalization.json").read_text()) != normalization:
                raise ValueError("stored normalization differs from fitting-only reconstruction")
            edge_hashes = []
            for method in frozen["methods"]:
                run_dir = fold / method
                checkpoint_path = run_dir / "selected_model.pt"
                checkpoint_hash = sha256_file(checkpoint_path)
                record = records[(seed, heldout, method)]
                if checkpoint_hash != record["checkpoint_sha256"]:
                    raise ValueError("selected checkpoint bytes changed")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                identities = {
                    "method": method, "seed": seed, "dataset": dataset, "heldout_policy": heldout,
                    "configuration": config, "normalization": normalization,
                    "protocol_sha256": record["protocol_sha256"],
                }
                if any(checkpoint[key] != value for key, value in identities.items()):
                    raise ValueError("checkpoint identity or normalization mismatch")
                fit_view = DevelopmentGraphView(
                    fitting, train_graph[fit_indices], method, config["knn_k"]
                )
                select_view = DevelopmentGraphView(
                    selection, train_graph[select_indices], method, config["knn_k"]
                )
                challenge_view = DevelopmentGraphView(
                    challenge, val_graph[challenge_indices], method, config["knn_k"]
                )
                if (fit_view.graph_receipt != record["fit"]["graph_fit"]
                        or select_view.graph_receipt != record["fit"]["graph_inner_selection"]
                        or challenge_view.graph_receipt != record["graph_challenge"]
                        or challenge_view.graph_receipt != checkpoint["graph_challenge"]):
                    raise ValueError("fold-local graph cannot be reconstructed")
                edge_hashes.append(tuple(view.graph_receipt["edge_pairs_sha256"]
                                         for view in (fit_view, select_view, challenge_view)))
                model = make_backbone(challenge_view, class_count, config)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if sum(p.numel() for p in model.parameters()) != record["fit"]["parameter_count"]:
                    raise ValueError("parameter count differs")
                probabilities = predict_backbone(
                    model, challenge_view, config["batch_size"], config["knn_k"]
                )
                saved = [json.loads(line) for line in
                         (run_dir / "challenge_predictions.jsonl").read_text().splitlines()
                         if line.strip()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in saved] != [
                    (row["sample_id"], row["base_profile_id"], row["label"])
                    for row in challenge["metadata"]
                ]:
                    raise ValueError("prediction identities or row order differ")
                expected = np.asarray([row["probabilities"] for row in saved])
                difference = float(np.max(np.abs(expected - probabilities)))
                if not np.allclose(expected, probabilities, atol=1e-6, rtol=0):
                    raise ValueError("checkpoint predictions cannot be replayed")
                if sha256_file(checkpoint_path) != checkpoint_hash:
                    raise ValueError("audit changed checkpoint bytes")
                replays.append({
                    "seed": seed, "heldout_policy": heldout, "method": method,
                    "checkpoint_sha256": checkpoint_hash,
                    "maximum_probability_difference": difference,
                })
            if len(set(edge_hashes)) != 1:
                raise ValueError("timing and no-timing methods have different graph inputs")
    result = {
        "dataset": dataset, "checkpoint_replays": replays, "all_replayed": True,
        "test_access": False, "submission_allowed": False,
        "normalization_scope": "inner_fitting_profiles_only",
        "same_graph_for_timing_and_without_timing": True,
        "audit_source_sha256": sha256_file(Path(__file__)),
    }
    out.mkdir(parents=True)
    write_new_json(out / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("data-dir", "temporal-dir", "master", "root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result = audit(args.data_dir, args.temporal_dir, args.master, args.root,
                       args.out_dir, args.dataset)
    print(json.dumps({"dataset": args.dataset, "all_replayed": result["all_replayed"],
                      "replayed_checkpoints": len(result["checkpoint_replays"])}), flush=True)


if __name__ == "__main__":
    main()
