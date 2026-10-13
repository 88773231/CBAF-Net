"""Fit and replay a fixed graph-free mechanism control on existing training folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import describe
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from analysis.run_flat_policy_screen import pool, source_hashes as reference_sources
from src.botsscl_adapter import CorruptionAdapter
from src.botsscl_policy_control import METHODS, SOURCE_COMMIT, predict_control, train_control
from src.evidence_repair import metrics, write_new_json
from src.provenance import sha256_file


DATASETS = ("Twibot22", "Quadbot")
SEEDS = (42, 43, 44)
SOURCES = (
    "src/botsscl_adapter.py", "src/botsscl_policy_control.py",
    "analysis/run_botsscl_policy_control.py", "configs/botsscl_policy_control.json",
)
SCOPE = "posthoc_training_only_botsscl_mechanism_control"


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {**reference_sources(), **{name: sha256_file(code / name) for name in SOURCES}}


def validate_settings(settings, reference):
    expected = {
        "schema": "training-only-botsscl-policy-control-v1",
        "seeds": list(SEEDS), "methods": list(METHODS),
        "configuration": reference["configuration"],
        "pretraining": {"epochs": 30, "corruption_rate": .6, "temperature": 1.0},
        "source_commit": SOURCE_COMMIT, "new_fits": 36, "new_checkpoint_replays": 36,
        "primary_task": "Quadbot", "transfer_task": "Twibot22",
        "primary_metric": "pooled_three_excluded_policy_macro_f1",
        "adaptation_status": "matched_input_mechanism_control_not_faithful_reproduction",
        "original_val_or_test_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False, "no_hyperparameter_search": True,
        "no_label_timing_or_account_augmentation": True,
        "matched_input_not_matched_architecture_or_total_compute": True,
        "report_all_conditions_classes_policies_and_seeds": True,
        "independent_confirmation": False, "submission_allowed": False, "cpu_threads": 2,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed control configuration or reporting boundary differs")
    if (reference["source_hashes"] != reference_sources()
            or reference["original_val_paths_opened"] or reference["test_paths_opened"]
            or reference["challenge_used_for_checkpoint_selection"]):
        raise ValueError("reference training-only recipe or source differs")


def expected_conditions():
    return {(seed, policy, method) for seed in SEEDS for policy in POLICIES for method in METHODS}


def identity(row):
    return row["seed"], row["heldout_policy"], row["method"]


def run_folder(root, dataset, row):
    return root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]


def run(data_dir, temporal, partitions, dataset, protocol_path, reference_root, output):
    if dataset not in DATASETS:
        raise ValueError("unknown controlled view")
    if output.exists():
        raise FileExistsError("control evidence cannot be overwritten")
    reference_path = reference_root / dataset / "protocol.json"
    reference = json.loads(reference_path.read_text())
    settings = json.loads(protocol_path.read_text())
    validate_settings(settings, reference)
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != reference[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("reference inputs or policy partitions differ")
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"], "device": "cpu",
        "input_hashes": material["input_hashes"], "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "reference_protocol_sha256": sha256_file(reference_path),
        "test_paths_opened": False, "original_val_paths_opened": False,
        "input_aggregation": "complete_flattened_history_not_temporal_mean",
        "architecture": "two_layer_relu_encoder_linear_classifier_training_only_projection",
        "adapter_has_dropout": False,
    }
    write_new_json(output / "protocol.json", frozen)
    protocol_hash, config = sha256_file(output / "protocol.json"), settings["configuration"]
    rows = []
    for seed in SEEDS:
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            prior_normalization = reference_root / dataset / "main" / f"init{seed}_fold{seed}" / policy / "normalization.json"
            if normalization != json.loads(prior_normalization.read_text()):
                raise ValueError("reference fitting-only normalization differs")
            write_new_json(output / f"seed{seed}" / policy / "normalization.json", normalization)
            for method in METHODS:
                destination = output / f"seed{seed}" / policy / method
                destination.mkdir()

                def progress(state):
                    print(json.dumps({"dataset": dataset, "seed": seed, "policy": policy,
                                      "method": method, **state}), flush=True)

                model, fitted = train_control(
                    values[0], values[1], material["policies"], method,
                    material["class_count"], seed, config, settings["pretraining"], progress,
                )
                probabilities = predict_control(model, values[2], config["batch_size"])
                checkpoint = destination / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "dataset": dataset, "seed": seed,
                    "heldout_policy": policy, "method": method, "configuration": config,
                    "pretraining": settings["pretraining"], "normalization": normalization,
                    "protocol_sha256": protocol_hash, "input_dimension": fitted["input_dimension"],
                }, checkpoint)
                prediction = destination / "challenge_predictions.jsonl"
                with prediction.open("x", encoding="utf-8") as stream:
                    for row, probability in zip(values[2]["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "prediction": int(probability.argmax()),
                            "probabilities": probability.tolist(),
                        }) + "\n")
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": policy, "method": method,
                    "fit": fitted, "challenge": metrics(
                        values[2]["labels"].numpy(), probabilities, material["class_count"],
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(prediction),
                    "protocol_sha256": protocol_hash, "original_val_or_test_access": False,
                    "challenge_used_for_checkpoint_selection": False, "claim_scope": SCOPE,
                }
                write_new_json(destination / "metrics.json", record)
                rows.append(record)
                print(json.dumps({"event": "fit_complete", "dataset": dataset, "completed": len(rows),
                                  "planned": 18}), flush=True)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": rows, "submission_allowed": False,
        "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("control replay evidence cannot be overwritten")
    protocol_path = root / dataset / "protocol.json"
    frozen = json.loads(protocol_path.read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("frozen control source changed")
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != frozen[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("control replay inputs changed")
    summary = json.loads((root / dataset / "summary.json").read_text())
    rows, protocol_hash = summary["runs"], sha256_file(protocol_path)
    if len(rows) != 18 or {identity(row) for row in rows} != expected_conditions():
        raise ValueError("incomplete control conditions")
    config, cache, audited = frozen["configuration"], {}, []
    for row in rows:
        key = row["seed"], row["heldout_policy"]
        if key not in cache:
            cache[key] = fold(material, *key, config["knn_k"])
        values, _, normalization = cache[key]
        folder = run_folder(root, dataset, row)
        checkpoint, prediction = folder / "selected_model.pt", folder / "challenge_predictions.jsonl"
        if (sha256_file(checkpoint) != row["checkpoint_sha256"]
                or sha256_file(prediction) != row["predictions_sha256"]):
            raise ValueError("control checkpoint or predictions changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = {
            **{name: row[name] for name in ("dataset", "seed", "heldout_policy", "method")},
            "configuration": config, "pretraining": frozen["pretraining"],
            "normalization": normalization, "protocol_sha256": protocol_hash,
            "input_dimension": row["fit"]["input_dimension"],
        }
        if any(saved[name] != value for name, value in expected.items()):
            raise ValueError("control checkpoint identity differs")
        model = CorruptionAdapter(saved["input_dimension"], material["class_count"], config["mlp_hidden_dim"])
        model.load_state_dict(saved["state_dict"], strict=True)
        probabilities = predict_control(model, values[2], config["batch_size"])
        predictions = [json.loads(line) for line in prediction.read_text().splitlines()]
        if [(value["sample_id"], value["base_profile_id"], value["label"]) for value in predictions] != [
            (value["sample_id"], value["base_profile_id"], value["label"]) for value in values[2]["metadata"]
        ]:
            raise ValueError("control replay row identity differs")
        difference = float(np.abs(probabilities - np.asarray(
            [value["probabilities"] for value in predictions],
        )).max())
        if (difference > 1e-6 or probabilities.argmax(1).tolist() != [
            value["prediction"] for value in predictions
        ] or metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]) != row["challenge"]):
            raise ValueError("control selected checkpoint failed replay")
        audited.append({**{name: row[name] for name in ("seed", "heldout_policy", "method")},
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "predictions_sha256": row["predictions_sha256"],
                        "maximum_probability_difference": difference})
    write_new_json(output / "audit.json", {
        "all_replayed": True, "checkpoint_replays": audited,
        "original_val_or_test_access": False, "submission_allowed": False,
    })


def verified(root, dataset):
    protocol_path = root / dataset / "protocol.json"
    frozen = json.loads(protocol_path.read_text())
    summary = json.loads((root / dataset / "summary.json").read_text())
    audit = json.loads((root / (dataset + "_audit") / "audit.json").read_text())
    rows, protocol_hash = summary["runs"], sha256_file(protocol_path)
    replays = {identity(row): row for row in audit["checkpoint_replays"]}
    if (len(rows) != 18 or {identity(row) for row in rows} != expected_conditions()
            or len(audit["checkpoint_replays"]) != 18 or set(replays) != expected_conditions()
            or not audit["all_replayed"] or audit["original_val_or_test_access"]
            or audit["submission_allowed"] or summary["submission_allowed"]
            or summary["protocol_sha256"] != protocol_hash or frozen["source_hashes"] != source_hashes()
            or frozen["original_val_paths_opened"] or frozen["test_paths_opened"]):
        raise ValueError("incomplete or changed control evidence")
    for row in rows:
        folder, selected = run_folder(root, dataset, row), replays[identity(row)]
        if (row != json.loads((folder / "metrics.json").read_text())
                or row["protocol_sha256"] != protocol_hash or row["original_val_or_test_access"]
                or row["challenge_used_for_checkpoint_selection"]
                or sha256_file(folder / "selected_model.pt") != row["checkpoint_sha256"]
                or sha256_file(folder / "challenge_predictions.jsonl") != row["predictions_sha256"]
                or selected["checkpoint_sha256"] != row["checkpoint_sha256"]
                or selected["predictions_sha256"] != row["predictions_sha256"]
                or selected["maximum_probability_difference"] > 1e-6):
            raise ValueError("control per-run provenance differs")
    return frozen, rows


def summarize(root, output):
    if output.exists():
        raise FileExistsError("control aggregate cannot be overwritten")
    report = {
        "scope": SCOPE, "new_fits": 36, "new_checkpoint_replays": 36,
        "submission_allowed": False, "independent_confirmation": False,
        "adaptation_status": "matched_input_mechanism_control_not_faithful_reproduction",
        "datasets": {},
    }
    for dataset in DATASETS:
        frozen, rows = verified(root, dataset)
        identities, pooled = {}, {}
        for method in METHODS:
            scores = []
            for seed in SEEDS:
                selected = [row for row in rows if row["method"] == method and row["seed"] == seed]
                paths = [run_folder(root, dataset, row) / "challenge_predictions.jsonl" for row in selected]
                scores.append(pool(selected, paths, frozen["class_count"], identities))
            pooled[method] = {
                "macro_f1": describe([row["macro_f1"] for row in scores]),
                "metrics_by_seed": scores,
            }
        left = pooled[METHODS[0]]["macro_f1"]["values"]
        right = pooled[METHODS[1]]["macro_f1"]["values"]
        difference = np.asarray(left) - np.asarray(right)
        report["datasets"][dataset] = {
            "runs_verified": len(rows), "primary_pooled": pooled,
            "pretraining_minus_supervised": {
                **describe(difference), "positive_seed_count": int((difference > 0).sum()),
            },
            "per_policy": {
                policy: {
                    method: describe([row["challenge"]["macro_f1"] for row in rows
                                      if row["method"] == method and row["heldout_policy"] == policy])
                    for method in METHODS
                } for policy in POLICIES
            },
            "complete_runs": rows,
        }
    write_new_json(output / "development_summary.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("run", "replay", "summarize"))
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--temporal-dir", type=Path)
    parser.add_argument("--partition-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("configs/botsscl_policy_control.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.operation == "summarize":
            summarize(args.study_root, args.output or args.study_root / "summary")
            return
        if any(value is None for value in (
            args.dataset, args.data_dir, args.temporal_dir, args.partition_root,
        )):
            parser.error("run/replay requires dataset, training inputs and prior partitions")
        if args.operation == "run":
            if args.reference_root is None:
                parser.error("run requires the preserved flat-policy study")
            run(args.data_dir, args.temporal_dir, args.partition_root, args.dataset,
                args.protocol, args.reference_root, args.output or args.study_root / args.dataset)
        else:
            replay(args.data_dir, args.temporal_dir, args.partition_root, args.dataset,
                   args.study_root, args.output or args.study_root / (args.dataset + "_audit"))


if __name__ == "__main__":
    main()
