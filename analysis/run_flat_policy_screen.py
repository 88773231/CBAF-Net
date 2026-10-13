"""Frozen SAM/contrast study with separate training-seed and partition probes."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import describe
from analysis.run_conditional_screen import CONTROLS, DATASETS, SEEDS
from analysis.run_cosine_policy_screen import cached as cached_linear
from analysis.run_cosine_policy_screen import source_hashes as prior_source_hashes
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from src.evidence_repair import metrics, write_new_json
from src.flat_policy import METHODS, train_flat
from src.paired_policy import make_paired_model, predict_paired
from src.policy_robust import train_policy_model
from src.provenance import sha256_file


ALL_METHODS = (*METHODS, "without_flatness", *CONTROLS)
SCOPE = "posthoc_training_only_flat_policy_development"
SOURCES = ("src/flat_policy.py", "analysis/run_flat_policy_screen.py", "configs/flat_policy_screen.json")
AXES = {
    "fixed_partition": ((43, 42), (44, 42)),
    "fixed_training_seed": ((42, 43), (42, 44)),
}


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {**prior_source_hashes(), **{name: sha256_file(code / name) for name in SOURCES}}


def cached(prior_root, replication_root, paired_root, dataset):
    prior, rows, provenance = cached_linear(prior_root, replication_root, paired_root, dataset)
    rows = copy.deepcopy(rows)
    for row in rows:
        if row["method"] == "without_cosine_head":
            row["method"] = "without_flatness"
    return prior, rows, provenance


def validate_settings(settings, prior):
    expected = {
        "schema": "training-only-flat-policy-screen-v1", "seeds": list(SEEDS),
        "new_methods": list(METHODS), "configuration": prior["configuration"],
        "regularization": {"contrast_weight": .1, "temperature": .2, "sam_radius": .05},
        "cached_flatness_removal": "paired_policy_screen/without_pair_margin",
        "both_removed_control": "erm_mlp", "new_main_fits": 54, "new_axis_fits": 48,
        "new_fits": 102, "cached_fits": 72, "cpu_threads": 2,
        "primary_task": "Quadbot", "transfer_task": "Twibot22",
        "primary_metric": "pooled_three_excluded_policy_macro_f1",
        "axis_probe_is_independent_confirmation": False, "novelty_established": False,
        "submission_allowed": False, "no_hyperparameter_search": True,
        "original_val_or_test_paths_opened": False,
        "no_label_timing_or_account_augmentation": True,
        "report_all_conditions_classes_policies_seeds_and_axes": True,
        "same_epoch_not_same_compute": True,
        "gradient_evaluations_per_batch": 2,
        "compute_control": "two_pass_erm",
        "desired_control_gain_points": 1.0, "desired_positive_seed_count": 2,
        "diagnostic_sha256": "7781433d0f5489f90014ce4a07e09b11867a12366d166d785644c6e9ea329bda",
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed flat-policy recipe, controls and transparent reporting required")
    if settings["axis_pairs"] != {
        name: [list(pair) for pair in pairs] for name, pairs in AXES.items()
    }:
        raise ValueError("training seed and partition axis probes differ")


def expected_conditions():
    main = {("main", seed, seed, policy, method)
            for seed in SEEDS for policy in POLICIES for method in METHODS}
    axes = {(axis, seed, partition, policy, method)
            for axis, pairs in AXES.items() for seed, partition in pairs
            for policy in POLICIES for method in ("flat_policy", "erm_mlp")}
    return main | axes


def identity(row):
    return row["scheme"], row["seed"], row["partition_seed"], row["heldout_policy"], row["method"]


def run_folder(root, dataset, row):
    return (root / dataset / row["scheme"] /
            f"init{row['seed']}_fold{row['partition_seed']}" /
            row["heldout_policy"] / row["method"])


def run(data_dir, temporal, partitions, dataset, protocol, prior_root, replication_root, paired_root, output):
    if output.exists():
        raise FileExistsError("flat-policy study cannot be overwritten")
    settings = json.loads(protocol.read_text())
    prior, _, provenance = cached(prior_root, replication_root, paired_root, dataset)
    validate_settings(settings, prior)
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != prior[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("immutable development inputs differ")
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"], "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "cached_provenance": provenance,
        "device": "cpu", "original_val_paths_opened": False, "test_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False,
    }
    write_new_json(output / "protocol.json", frozen)
    config, protocol_hash = settings["configuration"], sha256_file(output / "protocol.json")
    schemes = {"main": tuple((seed, seed) for seed in SEEDS), **AXES}
    rows = []
    for scheme, pairs in schemes.items():
        methods = METHODS if scheme == "main" else ("flat_policy", "erm_mlp")
        for seed, partition_seed in pairs:
            for policy in POLICIES:
                values, _, normalization = fold(material, partition_seed, policy, config["knn_k"])
                old = paired_root / dataset / f"seed{partition_seed}" / policy / "normalization.json"
                if normalization != json.loads(old.read_text()):
                    raise ValueError("fixed prior fold normalization differs")
                folder = output / scheme / f"init{seed}_fold{partition_seed}" / policy
                write_new_json(folder / "normalization.json", normalization)
                for method in methods:
                    destination = folder / method
                    destination.mkdir()

                    def progress(state):
                        print(json.dumps({"dataset": dataset, "scheme": scheme, "seed": seed,
                                          "partition_seed": partition_seed, "policy": policy,
                                          "method": method, **state}), flush=True)

                    if method == "erm_mlp":
                        model, fitted = train_policy_model(
                            values[0], values[1], material["policies"], method, material["class_count"],
                            seed, config,
                        )
                    else:
                        model, fitted = train_flat(
                            values[0], values[1], material["policies"], method, material["class_count"],
                            seed, config, settings["regularization"], progress,
                        )
                    probabilities = predict_paired(model, values[2], config["batch_size"])
                    checkpoint = destination / "selected_model.pt"
                    torch.save({
                        "state_dict": model.state_dict(), "dataset": dataset, "scheme": scheme,
                        "method": method, "seed": seed, "partition_seed": partition_seed,
                        "heldout_policy": policy, "configuration": config,
                        "regularization": settings["regularization"], "normalization": normalization,
                        "protocol_sha256": protocol_hash,
                    }, checkpoint)
                    path = destination / "challenge_predictions.jsonl"
                    with path.open("x", encoding="utf-8") as stream:
                        for row, probability in zip(values[2]["metadata"], probabilities):
                            stream.write(json.dumps({
                                "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                                "label": int(row["label"]), "prediction": int(probability.argmax()),
                                "probabilities": probability.tolist(),
                            }) + "\n")
                    record = {
                        "dataset": dataset, "scheme": scheme, "seed": seed, "partition_seed": partition_seed,
                        "heldout_policy": policy, "method": method, "fit": fitted,
                        "challenge": metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]),
                        "checkpoint_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(path),
                        "protocol_sha256": protocol_hash, "original_val_or_test_access": False,
                        "challenge_used_for_checkpoint_selection": False, "claim_scope": SCOPE,
                    }
                    write_new_json(destination / "metrics.json", record)
                    rows.append(record)
                    print(json.dumps({"event": "fit_complete", "dataset": dataset, "completed": len(rows),
                                      "planned": 51, "scheme": scheme, "method": method}), flush=True)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": rows, "submission_allowed": False, "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("flat-policy replay cannot be overwritten")
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("frozen flat-policy source changed")
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != frozen[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("flat-policy replay inputs changed")
    summary = json.loads((root / dataset / "summary.json").read_text())
    if {identity(row) for row in summary["runs"]} != expected_conditions() or len(summary["runs"]) != 51:
        raise ValueError("incomplete main or seed/partition-axis fits")
    config, protocol_hash = frozen["configuration"], sha256_file(root / dataset / "protocol.json")
    partitions_cache, audited = {}, []
    for record in summary["runs"]:
        key = record["partition_seed"], record["heldout_policy"]
        if key not in partitions_cache:
            partitions_cache[key] = fold(material, *key, config["knn_k"])
        values, _, normalization = partitions_cache[key]
        destination = run_folder(root, dataset, record)
        checkpoint, prediction = destination / "selected_model.pt", destination / "challenge_predictions.jsonl"
        if (sha256_file(checkpoint) != record["checkpoint_sha256"]
                or sha256_file(prediction) != record["predictions_sha256"]):
            raise ValueError("flat-policy checkpoint or predictions changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = {
            **{key: record[key] for key in ("dataset", "scheme", "seed", "partition_seed", "heldout_policy", "method")},
            "configuration": config, "regularization": frozen["regularization"],
            "normalization": normalization, "protocol_sha256": protocol_hash,
        }
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError("flat-policy checkpoint identity differs")
        model = make_paired_model(values[2], material["class_count"], config)
        model.load_state_dict(saved["state_dict"], strict=True)
        probabilities = predict_paired(model, values[2], config["batch_size"])
        rows = [json.loads(line) for line in prediction.read_text().splitlines()]
        if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows] != [
            (row["sample_id"], row["base_profile_id"], row["label"]) for row in values[2]["metadata"]
        ]:
            raise ValueError("flat-policy replay row identities differ")
        difference = float(np.abs(probabilities - np.asarray([row["probabilities"] for row in rows])).max())
        if (difference > 1e-6 or probabilities.argmax(1).tolist() != [row["prediction"] for row in rows]
                or metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]) != record["challenge"]):
            raise ValueError("flat-policy selected checkpoint failed replay")
        audited.append({
            **{key: record[key] for key in ("scheme", "seed", "partition_seed", "heldout_policy", "method")},
            "checkpoint_sha256": record["checkpoint_sha256"],
            "predictions_sha256": record["predictions_sha256"], "maximum_probability_difference": difference,
        })
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
    replayed = {identity(row): row for row in audit["checkpoint_replays"]}
    if (len(rows) != 51 or {identity(row) for row in rows} != expected_conditions()
            or len(replayed) != 51 or len(audit["checkpoint_replays"]) != 51
            or set(replayed) != expected_conditions() or not audit["all_replayed"]
            or audit["original_val_or_test_access"] or summary["submission_allowed"]
            or summary["protocol_sha256"] != protocol_hash or frozen["source_hashes"] != source_hashes()
            or frozen["original_val_paths_opened"] or frozen["test_paths_opened"]):
        raise ValueError("incomplete flat-policy evidence")
    for row in rows:
        destination = run_folder(root, dataset, row)
        selected = replayed[identity(row)]
        if (row != json.loads((destination / "metrics.json").read_text())
                or row["protocol_sha256"] != protocol_hash or row["original_val_or_test_access"]
                or row["challenge_used_for_checkpoint_selection"]
                or sha256_file(destination / "selected_model.pt") != row["checkpoint_sha256"]
                or sha256_file(destination / "challenge_predictions.jsonl") != row["predictions_sha256"]
                or selected["checkpoint_sha256"] != row["checkpoint_sha256"]
                or selected["predictions_sha256"] != row["predictions_sha256"]
                or selected["maximum_probability_difference"] > 1e-6):
            raise ValueError("flat-policy per-run provenance differs")
    return frozen, rows


def pool(rows, paths, class_count, identities):
    combined = []
    for row, path in zip(rows, paths):
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        current = [(value["sample_id"], value["base_profile_id"], value["label"]) for value in predictions]
        policy = row["heldout_policy"]
        if policy not in identities:
            identities[policy] = current
        if (current != identities[policy] or metrics(
            [value["label"] for value in predictions], [value["probabilities"] for value in predictions], class_count,
        ) != row["challenge"]):
            raise ValueError("pooled flat-policy method predictions differ")
        combined.extend(predictions)
    keys = [(row["base_profile_id"], row["label"]) for row in combined]
    if len(keys) != len(set(keys)):
        raise ValueError("pooled flat-policy challenges repeat profile/class pairs")
    return metrics([row["label"] for row in combined], [row["probabilities"] for row in combined], class_count)


def summarize(prior_root, replication_root, paired_root, root, output):
    if output.exists():
        raise FileExistsError("flat-policy aggregate cannot be overwritten")
    report = {
        "scope": SCOPE, "new_fits": 102, "new_checkpoint_replays": 102, "cached_fits": 72,
        "primary_task": "Quadbot", "transfer_task": "Twibot22",
        "submission_allowed": False, "original_val_or_test_access": False, "datasets": {},
    }
    lines = [
        "# Flat Policy Development Study", "",
        "All main conditions and both seed/partition probes are exposed development evidence.",
        "The axes are not independent confirmation. Training seed includes initialization, batches and dropout.",
        "Two-pass ERM matches two gradient evaluations per batch, not optimizer updates or selected epoch counts.",
        "SAM is established technology, not a new algorithm. The rejected cosine/margin methods are not promoted.",
        "Macro-F1 is percent; SD is sample SD. Quadbot is primary; Twibot22 is transfer.", "",
        "| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        frozen, new_rows = verified(root, dataset)
        prior, old_rows, provenance = cached(prior_root, replication_root, paired_root, dataset)
        validate_settings(frozen, prior)
        if provenance != frozen["cached_provenance"]:
            raise ValueError("cached flat-policy provenance changed")
        main = {(row["seed"], row["heldout_policy"], row["method"]): row
                for row in new_rows if row["scheme"] == "main"}
        main.update({(row["seed"], row["heldout_policy"], row["method"]): row for row in old_rows})
        identities = {}

        def path(row):
            return (Path(row["prediction_path"]) if "prediction_path" in row
                    else run_folder(root, dataset, row) / "challenge_predictions.jsonl")

        primary, tables = {}, {}
        for method in ALL_METHODS:
            per_seed, hashes = [], {}
            for seed in SEEDS:
                selected = [main[(seed, policy, method)] for policy in POLICIES]
                paths = [path(row) for row in selected]
                per_seed.append(pool(selected, paths, frozen["class_count"], identities))
                hashes.update({f"seed{seed}/{row['heldout_policy']}": sha256_file(item)
                               for row, item in zip(selected, paths)})
            primary[method] = {
                "macro_f1": describe([row["macro_f1"] for row in per_seed]),
                "metrics_by_seed": per_seed, "prediction_hashes": hashes,
            }
            score = primary[method]["macro_f1"]
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} |".format(
                dataset, method, " | ".join(f"{value * 100:.2f}" for value in score["values"]),
                score["mean"] * 100, score["seed_sd"] * 100,
            ))
        for policy in POLICIES:
            tables[policy] = {}
            for method in ALL_METHODS:
                selected = [main[(seed, policy, method)] for seed in SEEDS]
                tables[policy][method] = {
                    "macro_f1": describe([row["challenge"]["macro_f1"] for row in selected]),
                    "per_class_f1_by_seed": [row["challenge"]["per_class_f1"] for row in selected],
                    "parameter_count": selected[0]["fit"]["parameter_count"],
                }
        differences = {}
        for method in ALL_METHODS[1:]:
            delta = describe([left - right for left, right in zip(
                primary["flat_policy"]["macro_f1"]["values"], primary[method]["macro_f1"]["values"],
            )])
            differences[method] = {**delta, "higher_seed_count": sum(value > 0 for value in delta["values"])}
        axes = {}
        indexed = {identity(row): row for row in new_rows}
        for axis, pairs in AXES.items():
            pairs = ((42, 42), *pairs)
            axes[axis] = {}
            for method in ("flat_policy", "erm_mlp"):
                scores, hashes = [], {}
                for seed, partition in pairs:
                    selected = [
                        main[(42, policy, method)] if seed == partition == 42
                        else indexed[(axis, seed, partition, policy, method)]
                        for policy in POLICIES
                    ]
                    paths = [path(row) for row in selected]
                    scores.append(pool(selected, paths, frozen["class_count"], identities))
                    hashes.update({f"init{seed}_fold{partition}/{row['heldout_policy']}": sha256_file(item)
                                   for row, item in zip(selected, paths)})
                axes[axis][method] = {
                    "training_seed_partition_pairs": [list(pair) for pair in pairs],
                    "macro_f1": describe([row["macro_f1"] for row in scores]),
                    "metrics_by_pair": scores, "prediction_hashes": hashes,
                }
            full = axes[axis]["flat_policy"]["macro_f1"]["values"]
            erm = axes[axis]["erm_mlp"]["macro_f1"]["values"]
            delta = describe([left - right for left, right in zip(full, erm)])
            axes[axis]["full_minus_erm"] = {
                **delta, "higher_pair_count": sum(value > 0 for value in delta["values"]),
            }
        report["datasets"][dataset] = {
            "new_runs_verified": 51, "cached_runs_verified": 36, "primary_pooled": primary,
            "primary_pooled_full_minus_comparator": differences, "tables": tables,
            "axis_probes": axes, "protocol_sha256": sha256_file(root / dataset / "protocol.json"),
        }
    lines.extend(["", "| View | Full Minus | Mean Delta | Delta SD | Positive Seeds |",
                  "|---|---|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for method, delta in data["primary_pooled_full_minus_comparator"].items():
            lines.append(f"| {dataset} | {method} | {delta['mean'] * 100:+.2f} | "
                         f"{delta['seed_sd'] * 100:.2f} | {delta['higher_seed_count']}/3 |")
    lines.extend(["", "## Seed and Partition Axis Probes", "",
                  "| View | Axis | Method | Pair 1 | Pair 2 | Pair 3 | Mean | SD |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for axis, methods in data["axis_probes"].items():
            for method in ("flat_policy", "erm_mlp"):
                score = methods[method]["macro_f1"]
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, axis, method, " | ".join(f"{value * 100:.2f}" for value in score["values"]),
                    score["mean"] * 100, score["seed_sd"] * 100,
                ))
    lines.extend(["", "## Every Main Policy Condition", "",
                  "| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for policy, entries in data["tables"].items():
            for method, entry in entries.items():
                score = entry["macro_f1"]
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, policy, method, " | ".join(f"{value * 100:.2f}" for value in score["values"]),
                    score["mean"] * 100, score["seed_sd"] * 100,
                ))
    lines.extend(["", "Positive counts are descriptive, not significance tests.",
                  "No original validation/test evaluation or automatic manuscript promotion is authorized.",
                  "Recent bot-specific fair controls and genuinely independent confirmation remain gaps.", ""])
    write_new_json(output / "development_summary.json", report)
    (output / "development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "replay", "summarize"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    for name in ("data-dir", "temporal-dir", "partition-root", "protocol",
                 "prior-root", "replication-root", "paired-root", "root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=name == "out-dir")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "summarize":
            if not all((args.prior_root, args.replication_root, args.paired_root, args.root)):
                parser.error("summarize needs all evidence roots")
            summarize(args.prior_root, args.replication_root, args.paired_root, args.root, args.out_dir)
        else:
            if not all((args.data_dir, args.temporal_dir, args.partition_root, args.dataset)):
                parser.error("train/replay need original training inputs")
            common = args.data_dir, args.temporal_dir, args.partition_root, args.dataset
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay needs the candidate study root")
                replay(*common, args.root, args.out_dir)
            else:
                if not all((args.protocol, args.prior_root, args.replication_root, args.paired_root)):
                    parser.error("train needs fixed settings and cached evidence roots")
                run(*common, args.protocol, args.prior_root, args.replication_root, args.paired_root, args.out_dir)


if __name__ == "__main__":
    main()
