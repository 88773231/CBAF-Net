"""One-seed unseen-scheduler screen with a separate in-training selection split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from src.bse_config import release_bse_parameters
from src.data_loader import apply_column_standardizer, fit_column_standardizer, fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, load_split, metrics,
    predict, train_neural, write_new_json,
)
from src.provenance import sha256_file


def inner_profile_partition(rows, policies, heldout, seed):
    groups = sorted({
        row["base_profile_id"] for row in rows
        if policies[row["base_profile_id"]] != heldout
    }, key=lambda group: hashlib.sha256(f"policy-screen:{seed}:{group}".encode()).digest())
    selected_count = max(1, len(groups) // 6)
    selection_groups = set(groups[:selected_count])
    fitting_groups = set(groups[selected_count:])
    fitting = [index for index, row in enumerate(rows) if row["base_profile_id"] in fitting_groups]
    selection = [index for index, row in enumerate(rows) if row["base_profile_id"] in selection_groups]
    if not fitting or not selection or fitting_groups & selection_groups:
        raise ValueError("invalid inner profile partition")
    return fitting, selection


def fit_fold_transforms(fitting, selection, challenge, step_timing):
    table_transform = fit_column_standardizer(fitting["raw_tabular"])
    step_transform = fit_column_standardizer(step_timing[0].flatten(0, 1))
    for values, raw_steps in zip((fitting, selection, challenge), step_timing):
        normalized = apply_column_standardizer(values["raw_tabular"], table_transform)
        values["tabular"] = normalized
        values["numerical"] = torch.cat([normalized[:, :6], normalized[:, 17:]], dim=1)
        normalized_steps = apply_column_standardizer(
            raw_steps.flatten(0, 1), step_transform
        ).reshape_as(raw_steps)
        values["action"] = torch.cat([values["action"][:, :, :-7], normalized_steps], dim=2)
    return {
        "fit_scope": "inner_fitting_profiles_only",
        "table_mean": table_transform["mean"].tolist(),
        "table_std": table_transform["std"].tolist(),
        "step_mean": step_transform["mean"].tolist(),
        "step_std": step_transform["std"].tolist(),
    }


def run(data_dir, temporal, master, output, dataset, config):
    if output.exists():
        raise FileExistsError("refusing to overwrite the policy screen")
    class_count, seed = (3 if dataset == "Twibot22" else 4), 42
    transforms = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", 5, transforms, class_count)
    val = load_split(data_dir, "val", 5, transforms, class_count)
    assert_group_disjoint(train, val)
    attach_relative_timing(train, val, temporal)
    train_timing = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    val_timing = torch.load(temporal / "val_steps.pt", map_location="cpu", weights_only=True)
    policies, provenance = collect_provenance(master)
    protocol = {
        "dataset": dataset, "seed": seed, "configuration": config,
        "methods": ["hgb_statistics", "mlp_matched", "temporal_evidence"],
        "policy_scope": "traditional_scheduler_policy_only_not_cross_generator",
        "inner_selection": "one_sixth_of_remaining_original_training_profiles",
        "challenge": "original_validation_profiles_of_excluded_policy",
        "challenge_used_for_checkpoint_selection": False,
        "old_test_paths_opened": False,
        "claim_scope": "posthoc_one_seed_development_screen",
        "implementation_sha256": sha256_file(Path(__file__)),
        "provenance": provenance,
    }
    protocol_path = output.with_name(output.stem + "_protocol.json")
    write_new_json(protocol_path, protocol)
    records = []
    for heldout in sorted(set(policies.values())):
        fit_indices, select_indices = inner_profile_partition(
            train["metadata"], policies, heldout, seed
        )
        challenge_indices = [index for index, row in enumerate(val["metadata"])
                             if policies[row["base_profile_id"]] == heldout]
        fitting, selection, challenge = (
            subset(train, fit_indices), subset(train, select_indices),
            subset(val, challenge_indices),
        )
        assert_group_disjoint(fitting, selection)
        assert_group_disjoint(fitting, challenge)
        assert_group_disjoint(selection, challenge)
        normalization = fit_fold_transforms(
            fitting, selection, challenge,
            (train_timing[fit_indices], train_timing[select_indices], val_timing[challenge_indices]),
        )
        for method in protocol["methods"]:
            artifacts = output.parent / f"{dataset}_policy_screen" / heldout / method
            artifacts.mkdir(parents=True, exist_ok=False)
            if method == "hgb_statistics":
                estimator = HistGradientBoostingClassifier(**release_bse_parameters(seed))
                estimator.fit(fitting["raw_tabular"].numpy(), fitting["labels"].numpy())
                probabilities = estimator.predict_proba(challenge["raw_tabular"].numpy())
                selected_epoch = None
                import joblib
                joblib.dump(estimator, artifacts / "selected_model.joblib")
            else:
                model, _, fitted = train_neural(
                    fitting, selection, method, class_count, seed, config
                )
                # The excluded policy's probabilities/metrics are computed
                # only after checkpoint selection on the inner profiles.
                probabilities, _ = predict(model, challenge, method, config["batch_size"])
                selected_epoch = fitted["best_epoch"]
                torch.save({
                    "state_dict": model.state_dict(), "method": method,
                    "configuration": config, "normalization": normalization,
                    "protocol_sha256": sha256_file(protocol_path),
                    "heldout_policy": heldout, "seed": seed,
                }, artifacts / "selected_model.pt")
                write_new_json(artifacts / "inner_selection_history.json", fitted)
            with (artifacts / "challenge_predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row, probability in zip(challenge["metadata"], probabilities):
                    stream.write(json.dumps({
                        "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                        "label": int(row["label"]), "probabilities": probability.tolist(),
                        "prediction": int(probability.argmax()),
                    }) + "\n")
            record = {
                "heldout_policy": heldout, "method": method, "seed": seed,
                "fitting_profiles": len({row["base_profile_id"] for row in fitting["metadata"]}),
                "selection_profiles": len({row["base_profile_id"] for row in selection["metadata"]}),
                "challenge_profiles": len({row["base_profile_id"] for row in challenge["metadata"]}),
                "selected_epoch": selected_epoch, "normalization": normalization,
                "challenge": metrics(challenge["labels"].numpy(), probabilities, class_count),
            }
            records.append(record)
            write_new_json(artifacts / "metrics.json", record)
            print(json.dumps({key: record[key] for key in
                              ("heldout_policy", "method", "seed", "challenge")}), flush=True)
    result = {"protocol_sha256": sha256_file(protocol_path), **protocol, "runs": records}
    write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads(args.configuration.read_text(encoding="utf-8"))["configuration"]
    with threadpool_limits(limits=2):
        run(args.data_dir, args.temporal_dir, args.master, args.out, args.dataset, config)


if __name__ == "__main__":
    main()
