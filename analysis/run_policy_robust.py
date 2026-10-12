"""Bounded three-seed development screen; no old test path is constructed."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition
from src.bse_config import release_bse_parameters
from src.data_loader import fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, input_files, load_split,
    metrics, predict, write_new_json,
)
from src.policy_robust import NEURAL_METHODS, train_policy_model
from src.provenance import sha256_file


def run(data_dir, temporal, master, output, dataset, protocol_path):
    if output.exists():
        raise FileExistsError("refusing to overwrite a development round")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds, methods, config = protocol["seeds"], protocol["methods"], protocol["configuration"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct seeds are required")
    if len(set(methods)) != len(methods) or any(
        method not in ("hgb_statistics", *NEURAL_METHODS) for method in methods
    ):
        raise ValueError("invalid methods")
    class_count = 3 if dataset == "Twibot22" else 4
    standardizers = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", config["steps"], standardizers, class_count)
    val = load_split(data_dir, "val", config["steps"], standardizers, class_count)
    assert_group_disjoint(train, val)
    timing = attach_relative_timing(train, val, temporal)
    train_timing = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    val_timing = torch.load(temporal / "val_steps.pt", map_location="cpu", weights_only=True)
    policies, provenance = collect_provenance(master)
    output.mkdir(parents=True)
    code = Path(__file__).resolve().parents[1]
    source_names = (
        "analysis/run_policy_robust.py", "src/policy_robust.py", "src/evidence_model.py",
        "src/evidence_repair.py", "analysis/screen_unseen_policy.py", "src/data_loader.py",
    )
    frozen = {
        **protocol, "dataset": dataset, "class_count": class_count,
        "input_hashes": {path.relative_to(data_dir).as_posix(): sha256_file(path)
                         for split in ("train", "val")
                         for path in input_files(data_dir, split, config["steps"])},
        "timing": timing, "master_sha256": sha256_file(master),
        "source_hashes": {name: sha256_file(code / name) for name in source_names},
        "configuration_sha256": sha256_file(protocol_path),
        "provenance": provenance, "test_paths_opened": False,
        "outer_challenge_used_for_selection": False, "device": "cpu",
    }
    write_new_json(output / "protocol.json", frozen)
    protocol_hash = sha256_file(output / "protocol.json")
    records = []
    for seed in seeds:
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
            normalization = fit_fold_transforms(
                fitting, selection, challenge,
                (train_timing[fit_indices], train_timing[select_indices],
                 val_timing[challenge_indices]),
            )
            for method in methods:
                run_dir = output / f"seed{seed}" / heldout / method
                run_dir.mkdir(parents=True)
                if method == "hgb_statistics":
                    estimator = HistGradientBoostingClassifier(**release_bse_parameters(seed))
                    estimator.fit(fitting["raw_tabular"].numpy(), fitting["labels"].numpy())
                    probabilities = estimator.predict_proba(challenge["raw_tabular"].numpy())
                    fit = {"configuration": estimator.get_params(deep=False),
                           "selection_rule": "fixed_configuration_no_challenge_selection"}
                    import joblib
                    joblib.dump(estimator, run_dir / "selected_model.joblib")
                else:
                    model, fit = train_policy_model(
                        fitting, selection, policies, method, class_count, seed, config
                    )
                    probabilities, _ = predict(
                        model, challenge, "mlp_matched", config["batch_size"]
                    )
                    torch.save({
                        "state_dict": model.state_dict(), "method": method, "seed": seed,
                        "configuration": config, "normalization": normalization,
                        "heldout_policy": heldout, "protocol_sha256": protocol_hash,
                        "inference_model": "MatchedFeatureMLP",
                    }, run_dir / "selected_model.pt")
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": heldout, "method": method,
                    "fit": fit, "protocol_sha256": protocol_hash,
                    "fitting_profiles": len({row["base_profile_id"] for row in fitting["metadata"]}),
                    "selection_profiles": len({row["base_profile_id"] for row in selection["metadata"]}),
                    "challenge_profiles": len({row["base_profile_id"] for row in challenge["metadata"]}),
                    "challenge": metrics(challenge["labels"].numpy(), probabilities, class_count),
                    "test_state": "NOT_OPENED", "claim_scope": "posthoc_development_only",
                    "outer_challenge_used_for_selection": False,
                }
                write_new_json(run_dir / "metrics.json", record)
                with (run_dir / "challenge_predictions.jsonl").open("x", encoding="utf-8") as stream:
                    for row, probability in zip(challenge["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "probabilities": probability.tolist(),
                            "prediction": int(probability.argmax()),
                        }) + "\n")
                records.append(record)
                print(json.dumps({
                    "dataset": dataset, "seed": seed, "heldout_policy": heldout,
                    "method": method, "challenge": record["challenge"],
                    "epochs": len(fit.get("history", [])),
                    "elapsed_seconds": fit.get("elapsed_seconds"),
                }), flush=True)
    aggregates = {}
    for heldout in sorted(set(policies.values())):
        aggregates[heldout] = {}
        for method in methods:
            values = [row["challenge"]["macro_f1"] for row in records
                      if row["method"] == method and row["heldout_policy"] == heldout]
            if len(values) != len(seeds):
                raise ValueError("incomplete seed comparison")
            aggregates[heldout][method] = {
                "mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    result = {
        "dataset": dataset, "protocol_sha256": protocol_hash, "runs": records,
        "aggregates": aggregates, "test_state": "NOT_OPENED",
        "claim_scope": "posthoc_development_only", "submission_allowed": False,
    }
    write_new_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    with threadpool_limits(limits=config["cpu_threads"]):
        run(args.data_dir, args.temporal_dir, args.master, args.out_dir,
            args.dataset, args.protocol)


if __name__ == "__main__":
    main()
