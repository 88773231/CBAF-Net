"""Fixed training-only objective screen; the both-removed condition is ERM."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import describe
from analysis.run_conditional_screen import CONTROLS, DATASETS, SEEDS, cached_controls
from analysis.run_factor_training_screen import POLICIES, fold, prepare, source_hashes as legacy_hashes
from src.evidence_repair import metrics, write_new_json
from src.paired_policy import METHODS, make_paired_model, predict_paired, train_paired
from src.provenance import sha256_file


SOURCES = (
    "src/paired_policy.py", "analysis/run_paired_policy_screen.py",
    "configs/paired_policy_screen.json", "analysis/run_conditional_screen.py",
    "src/conditional_evidence.py", "analysis/replicate_factor_training_screen.py",
)
SCOPE = "posthoc_training_only_paired_policy_objective_development"


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {**legacy_hashes(), **{name: sha256_file(code / name) for name in SOURCES}}


def validate_settings(settings, prior):
    if (settings["schema"] != "training-only-paired-policy-screen-v1"
            or tuple(settings["seeds"]) != SEEDS or tuple(settings["methods"]) != METHODS
            or tuple(settings["reused_controls"]) != CONTROLS
            or settings["both_objectives_removed_control"] != "erm_mlp"
            or settings["primary_metric"] != "pooled_three_excluded_policy_macro_f1"
            or settings["every_class_or_policy_improvement_required"]
            or settings["configuration"] != prior["configuration"]
            or settings["cpu_threads"] != 2 or settings["new_fits"] != 54
            or settings["reused_control_fits"] != 54 or settings["submission_allowed"]
            or settings["novelty_established"] or not settings["no_hyperparameter_search"]
            or settings["original_val_or_test_paths_opened"]
            or not settings["no_label_timing_or_account_augmentation"]):
        raise ValueError("all fixed paired objectives and the matched control recipe are required")
    if prior["source_hashes"] != legacy_hashes():
        raise ValueError("preserved control implementation changed")


def run(data_dir, temporal, partitions, dataset, protocol, prior_root, replication_root, output):
    if output.exists():
        raise FileExistsError("paired study cannot be overwritten")
    settings = json.loads(protocol.read_text())
    prior, _, provenance = cached_controls(prior_root, replication_root, dataset)
    validate_settings(settings, prior)
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != prior[key]:
            raise ValueError("matched training inputs differ")
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"],
        "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "cached_control_provenance": provenance,
        "device": "cpu", "original_val_paths_opened": False, "test_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False,
    }
    write_new_json(output / "protocol.json", frozen)
    config, protocol_hash = settings["configuration"], sha256_file(output / "protocol.json")
    records = []
    for seed in SEEDS:
        control_root = prior_root if seed == 42 else replication_root
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            control_folder = control_root / dataset / f"seed{seed}" / policy
            if normalization != json.loads((control_folder / "normalization.json").read_text()):
                raise ValueError("paired/control folds differ")
            folder = output / f"seed{seed}" / policy
            write_new_json(folder / "normalization.json", normalization)
            for method in METHODS:
                destination = folder / method
                destination.mkdir()

                def progress(state):
                    print(json.dumps({"dataset": dataset, "seed": seed, "policy": policy,
                                      "method": method, **state}), flush=True)

                model, fitted = train_paired(
                    values[0], values[1], material["policies"], method,
                    material["class_count"], seed, config, settings["regularization"], progress,
                )
                probabilities = predict_paired(model, values[2], config["batch_size"])
                checkpoint = destination / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "dataset": dataset, "method": method,
                    "seed": seed, "heldout_policy": policy, "configuration": config,
                    "regularization": settings["regularization"], "normalization": normalization,
                    "protocol_sha256": protocol_hash,
                }, checkpoint)
                predictions = destination / "challenge_predictions.jsonl"
                with predictions.open("x", encoding="utf-8") as stream:
                    for row, probability in zip(values[2]["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "prediction": int(probability.argmax()),
                            "probabilities": probability.tolist(),
                        }) + "\n")
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": policy, "method": method,
                    "fit": fitted,
                    "challenge": metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "predictions_sha256": sha256_file(predictions), "protocol_sha256": protocol_hash,
                    "original_val_or_test_access": False, "challenge_used_for_checkpoint_selection": False,
                    "claim_scope": SCOPE,
                }
                write_new_json(destination / "metrics.json", record)
                records.append(record)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": records, "submission_allowed": False, "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("paired replay cannot be overwritten")
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("paired training source changed")
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != frozen[key]:
            raise ValueError("paired replay inputs differ")
    protocol_hash, rows = sha256_file(root / dataset / "protocol.json"), []
    config = frozen["configuration"]
    for seed in SEEDS:
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            folder = root / dataset / f"seed{seed}" / policy
            if normalization != json.loads((folder / "normalization.json").read_text()):
                raise ValueError("paired replay normalization differs")
            for method in METHODS:
                destination = folder / method
                record = json.loads((destination / "metrics.json").read_text())
                checkpoint, predictions_path = destination / "selected_model.pt", destination / "challenge_predictions.jsonl"
                if (sha256_file(checkpoint) != record["checkpoint_sha256"]
                        or sha256_file(predictions_path) != record["predictions_sha256"]):
                    raise ValueError("checkpoint/prediction identity differs")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                for key, expected in (
                    ("dataset", dataset), ("method", method), ("seed", seed), ("heldout_policy", policy),
                    ("configuration", config), ("normalization", normalization),
                    ("regularization", frozen["regularization"]), ("protocol_sha256", protocol_hash),
                ):
                    if saved[key] != expected:
                        raise ValueError("paired checkpoint identity differs: " + key)
                model = make_paired_model(values[2], material["class_count"], config)
                model.load_state_dict(saved["state_dict"], strict=True)
                probabilities = predict_paired(model, values[2], config["batch_size"])
                predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in predictions] != [
                    (row["sample_id"], row["base_profile_id"], row["label"]) for row in values[2]["metadata"]
                ]:
                    raise ValueError("paired prediction rows differ")
                difference = float(np.abs(
                    probabilities - np.asarray([row["probabilities"] for row in predictions]),
                ).max())
                if (difference > 1e-6 or probabilities.argmax(1).tolist() != [row["prediction"] for row in predictions]
                        or metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]) != record["challenge"]):
                    raise ValueError("paired checkpoint did not replay")
                rows.append({"seed": seed, "heldout_policy": policy, "method": method,
                             "maximum_probability_difference": difference,
                             "checkpoint_sha256": record["checkpoint_sha256"],
                             "predictions_sha256": record["predictions_sha256"]})
    write_new_json(output / "audit.json", {
        "all_replayed": True, "checkpoint_replays": rows,
        "original_val_or_test_access": False, "submission_allowed": False,
    })


def verified_candidates(root, dataset):
    frozen_path = root / dataset / "protocol.json"
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads((root / dataset / "summary.json").read_text())
    audit = json.loads((root / (dataset + "_audit") / "audit.json").read_text())
    rows, protocol_hash = summary["runs"], sha256_file(frozen_path)
    expected = {(seed, policy, method) for seed in SEEDS for policy in POLICIES for method in METHODS}
    observed = {(row["seed"], row["heldout_policy"], row["method"]) for row in rows}
    replayed = {(row["seed"], row["heldout_policy"], row["method"]): row for row in audit["checkpoint_replays"]}
    if (observed != expected or len(rows) != 27 or set(replayed) != expected or len(replayed) != 27
            or len(audit["checkpoint_replays"]) != 27 or not audit["all_replayed"]
            or audit["original_val_or_test_access"] or summary["submission_allowed"]
            or frozen["original_val_paths_opened"] or frozen["test_paths_opened"]
            or summary["protocol_sha256"] != protocol_hash or frozen["source_hashes"] != source_hashes()):
        raise ValueError("incomplete or changed paired evidence")
    for row in rows:
        folder = root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
        replayed_row = replayed[(row["seed"], row["heldout_policy"], row["method"])]
        if (row != json.loads((folder / "metrics.json").read_text())
                or row["protocol_sha256"] != protocol_hash or row["original_val_or_test_access"]
                or row["challenge_used_for_checkpoint_selection"]
                or sha256_file(folder / "selected_model.pt") != row["checkpoint_sha256"]
                or sha256_file(folder / "challenge_predictions.jsonl") != row["predictions_sha256"]
                or replayed_row["checkpoint_sha256"] != row["checkpoint_sha256"]
                or replayed_row["predictions_sha256"] != row["predictions_sha256"]
                or replayed_row["maximum_probability_difference"] > 1e-6):
            raise ValueError("paired per-run provenance differs")
    return frozen, rows


def summarize(prior_root, replication_root, root, output):
    if output.exists():
        raise FileExistsError("paired summary cannot be overwritten")
    report = {"scope": SCOPE, "new_fits": 54, "new_checkpoint_replays": 54,
              "reused_control_fits": 54, "submission_allowed": False,
              "original_val_or_test_access": False, "both_removed_control": "erm_mlp",
              "primary_metric": "pooled_three_excluded_policy_macro_f1", "datasets": {}}
    lines = ["# Paired Policy Objective Screen", "",
             "Post-hoc training-only development; not independent submission evidence.",
             "ERM MLP is the exact both-objectives-removed architecture and objective.",
             "These are training-objective removals, not two inference architecture modules.",
             "Macro-F1 is percent; SD is sample seed SD. Each seed also changes the inner profile partition.", "",
             "| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        frozen, new_rows = verified_candidates(root, dataset)
        prior, controls, provenance = cached_controls(prior_root, replication_root, dataset)
        validate_settings(frozen, prior)
        if (provenance != frozen["cached_control_provenance"]
                or any(prior[key] != frozen[key] for key in ("input_hashes", "prior_partition_hashes"))):
            raise ValueError("paired controls changed")
        indexed = {(row["seed"], row["heldout_policy"], row["method"]): row for row in new_rows + controls}
        tables, comparisons = {}, {}
        for policy in POLICIES:
            tables[policy], comparisons[policy] = {}, {}
            for method in (*METHODS, *CONTROLS):
                selected = [indexed[(seed, policy, method)] for seed in SEEDS]
                scores = describe([row["challenge"]["macro_f1"] for row in selected])
                counts = {row["fit"]["parameter_count"] for row in selected}
                if len(counts) != 1:
                    raise ValueError("capacity differs between seeds")
                tables[policy][method] = {"macro_f1": scores, "parameter_count": counts.pop(),
                                         "per_class_f1_by_seed": [row["challenge"]["per_class_f1"] for row in selected]}
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, policy, method, " | ".join(f"{value * 100:.2f}" for value in scores["values"]),
                    scores["mean"] * 100, scores["seed_sd"] * 100,
                ))
            if any(tables[policy][method]["parameter_count"] != tables[policy]["erm_mlp"]["parameter_count"]
                   for method in METHODS):
                raise ValueError("paired objectives must retain exact MLP capacity")
            for method in (*METHODS[1:], *CONTROLS):
                delta = describe([
                    indexed[(seed, policy, METHODS[0])]["challenge"]["macro_f1"]
                    - indexed[(seed, policy, method)]["challenge"]["macro_f1"] for seed in SEEDS
                ])
                comparisons[policy][method] = {**delta, "higher_seed_count": sum(value > 0 for value in delta["values"])}
        primary = {}
        for method in (*METHODS, *CONTROLS):
            scores, prediction_hashes = [], {}
            for seed in SEEDS:
                source = root if method in METHODS else (prior_root if seed == 42 else replication_root)
                predictions = []
                for policy in POLICIES:
                    path = source / dataset / f"seed{seed}" / policy / method / "challenge_predictions.jsonl"
                    expected = indexed[(seed, policy, method)]
                    current_hash = sha256_file(path)
                    if method in METHODS and current_hash != expected["predictions_sha256"]:
                        raise ValueError("pooled prediction identity differs")
                    policy_rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if metrics(
                        [row["label"] for row in policy_rows],
                        [row["probabilities"] for row in policy_rows], frozen["class_count"],
                    ) != expected["challenge"]:
                        raise ValueError("pooled policy metrics differ from verified checkpoint evidence")
                    prediction_hashes[f"seed{seed}/{policy}"] = current_hash
                    predictions.extend(policy_rows)
                keys = [(row["base_profile_id"], int(row["label"])) for row in predictions]
                if len(keys) != len(set(keys)):
                    raise ValueError("pooled policies repeat a profile/class row")
                scores.append(metrics(
                    [row["label"] for row in predictions],
                    [row["probabilities"] for row in predictions], frozen["class_count"],
                ))
            primary[method] = {
                "macro_f1": describe([row["macro_f1"] for row in scores]),
                "metrics_by_seed": scores, "prediction_hashes": prediction_hashes,
            }
        primary_differences = {}
        for method in (*METHODS[1:], *CONTROLS):
            delta = describe([
                left - right for left, right in zip(
                    primary[METHODS[0]]["macro_f1"]["values"], primary[method]["macro_f1"]["values"],
                )
            ])
            primary_differences[method] = {**delta, "higher_seed_count": sum(value > 0 for value in delta["values"])}
        report["datasets"][dataset] = {
            "new_runs_verified": 27, "cached_controls_verified": 27, "tables": tables,
            "full_minus_comparator": comparisons, "primary_pooled": primary,
            "primary_pooled_full_minus_comparator": primary_differences,
        }
    lines.extend(["", "## Primary Overall Metric", "",
                  "Pool all three excluded-policy predictions once per profile/class, separately for each seed.",
                  "This outcome and success target were fixed before the new scores were inspected.",
                  "Per-class or every-policy improvement is not required; those scores remain diagnostic.",
                  "Each view needs overall gains against controls and both individual objective removals.", "",
                  "| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for method, entry in data["primary_pooled"].items():
            scores = entry["macro_f1"]
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} |".format(
                dataset, method, " | ".join(f"{value * 100:.2f}" for value in scores["values"]),
                scores["mean"] * 100, scores["seed_sd"] * 100,
            ))
    lines.extend(["", "| View | Overall Full Minus | Mean Delta | Delta SD | Positive Seeds |",
                  "|---|---|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for method, delta in data["primary_pooled_full_minus_comparator"].items():
            lines.append(f"| {dataset} | {method} | {delta['mean'] * 100:+.2f} | "
                         f"{delta['seed_sd'] * 100:.2f} | {delta['higher_seed_count']}/3 |")
    lines.extend(["", "## Per-Policy Paired Differences", ""])
    lines.extend(["", "| View | Policy | Full Minus | Mean Delta | Delta SD | Positive Seeds |",
                  "|---|---|---|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for policy, entries in data["full_minus_comparator"].items():
            for method, delta in entries.items():
                lines.append(f"| {dataset} | {policy} | {method} | {delta['mean'] * 100:+.2f} | "
                             f"{delta['seed_sd'] * 100:.2f} | {delta['higher_seed_count']}/3 |")
    lines.extend(["", "All planned conditions/seeds are retained. No candidate is promoted automatically.",
                  "The two views share profiles and construction labels; they are not independent real-world datasets.",
                  "HGB has global summaries; adapted BotDMM has extra unlabelled feature-kNN context.",
                  "Original datasets, published results and manuscript remain unchanged.", ""])
    write_new_json(output / "development_summary.json", report)
    (output / "development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "replay", "summarize"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    for name in ("data-dir", "temporal-dir", "partition-root", "protocol",
                 "prior-root", "replication-root", "root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=name == "out-dir")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "summarize":
            if not all((args.prior_root, args.replication_root, args.root)):
                parser.error("summarize requires both cached roots and candidate root")
            summarize(args.prior_root, args.replication_root, args.root, args.out_dir)
        else:
            if not all((args.data_dir, args.temporal_dir, args.partition_root, args.dataset)):
                parser.error("train/replay require the frozen training inputs")
            common = args.data_dir, args.temporal_dir, args.partition_root, args.dataset
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay requires candidate root")
                replay(*common, args.root, args.out_dir)
            else:
                if not all((args.protocol, args.prior_root, args.replication_root)):
                    parser.error("train requires fixed protocol and both cached roots")
                run(*common, args.protocol, args.prior_root, args.replication_root, args.out_dir)


if __name__ == "__main__":
    main()
