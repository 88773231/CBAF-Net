"""Fixed normalized-readout screen with verified cached linear-head removals."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import describe
from analysis.run_conditional_screen import CONTROLS, DATASETS, SEEDS, cached_controls
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from analysis.run_paired_policy_screen import source_hashes as paired_hashes, verified_candidates as verified_paired
from src.cosine_policy import METHODS, make_cosine_model, predict_cosine, train_cosine
from src.evidence_repair import metrics, write_new_json
from src.provenance import sha256_file


ALL_METHODS = (*METHODS, "without_cosine_head", *CONTROLS)
SOURCES = ("src/cosine_policy.py", "analysis/run_cosine_policy_screen.py", "configs/cosine_policy_screen.json")
SCOPE = "posthoc_training_only_cosine_policy_development"


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {**paired_hashes(), **{name: sha256_file(code / name) for name in SOURCES}}


def cached(prior_root, replication_root, paired_root, dataset):
    prior, controls, provenance = cached_controls(prior_root, replication_root, dataset)
    paired, old = verified_paired(paired_root, dataset)
    if (paired["configuration"] != prior["configuration"]
            or paired["input_hashes"] != prior["input_hashes"]
            or paired["prior_partition_hashes"] != prior["prior_partition_hashes"]
            or paired["regularization"]["contrast_weight"] != .1
            or paired["regularization"]["temperature"] != .2):
        raise ValueError("cached contrast-only recipe or inputs differ")
    selected = [copy.deepcopy(row) for row in old if row["method"] == "without_pair_margin"]
    if len(selected) != 9:
        raise ValueError("nine cached linear-head removals are required")
    rows = []
    for row in controls:
        cloned = copy.deepcopy(row)
        source = prior_root if row["seed"] == 42 else replication_root
        cloned["prediction_path"] = str(
            source / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"] / "challenge_predictions.jsonl"
        )
        cloned["original_method"] = row["method"]
        rows.append(cloned)
    for row in selected:
        if row["fit"]["active_pair_weight"] != 0 or row["fit"]["active_contrast_weight"] != .1:
            raise ValueError("cached removal still uses a different objective")
        row["original_method"] = row["method"]
        row["prediction_path"] = str(
            paired_root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"] / "challenge_predictions.jsonl"
        )
        row["method"] = "without_cosine_head"
        rows.append(row)
    provenance["paired_protocol"] = sha256_file(paired_root / dataset / "protocol.json")
    provenance["paired_summary"] = sha256_file(paired_root / dataset / "summary.json")
    provenance["paired_audit"] = sha256_file(paired_root / (dataset + "_audit") / "audit.json")
    return prior, rows, provenance


def validate_settings(settings, prior):
    if (settings["schema"] != "training-only-cosine-policy-screen-v1"
            or tuple(settings["seeds"]) != SEEDS or tuple(settings["new_methods"]) != METHODS
            or settings["configuration"] != prior["configuration"]
            or settings["regularization"] != {"contrast_weight": .1, "temperature": .2, "cosine_scale": 16.0}
            or settings["cached_cosine_removal"] != "paired_policy_screen/without_pair_margin"
            or settings["both_removed_control"] != "erm_mlp"
            or settings["primary_task"] != "Quadbot" or settings["transfer_task"] != "Twibot22"
            or settings["primary_metric"] != "pooled_three_excluded_policy_macro_f1"
            or settings["new_fits"] != 36 or settings["reused_fits"] != 72 or settings["cpu_threads"] != 2
            or settings["submission_allowed"] or settings["novelty_established"]
            or not settings["no_hyperparameter_search"]
            or settings["original_val_or_test_paths_opened"]
            or not settings["report_transfer_and_all_class_policy_seed_results"]
            or settings["every_class_or_policy_improvement_required"]
            or not settings["no_label_timing_or_account_augmentation"]):
        raise ValueError("fixed cosine hypothesis and transparent transfer reporting required")


def run(data_dir, temporal, partitions, dataset, protocol, prior_root, replication_root, paired_root, output):
    if output.exists():
        raise FileExistsError("cosine study cannot be overwritten")
    settings = json.loads(protocol.read_text())
    prior, _, provenance = cached(prior_root, replication_root, paired_root, dataset)
    validate_settings(settings, prior)
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != prior[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("immutable training inputs differ")
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"], "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "cached_provenance": provenance, "device": "cpu",
        "original_val_paths_opened": False, "test_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False,
    }
    write_new_json(output / "protocol.json", frozen)
    config, protocol_hash = settings["configuration"], sha256_file(output / "protocol.json")
    rows = []
    for seed in SEEDS:
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            old_folder = paired_root / dataset / f"seed{seed}" / policy
            if normalization != json.loads((old_folder / "normalization.json").read_text()):
                raise ValueError("cached removal and new folds differ")
            folder = output / f"seed{seed}" / policy
            write_new_json(folder / "normalization.json", normalization)
            for method in METHODS:
                destination = folder / method
                destination.mkdir()

                def progress(state):
                    print(json.dumps({"dataset": dataset, "seed": seed, "policy": policy,
                                      "method": method, **state}), flush=True)

                model, fitted = train_cosine(
                    values[0], values[1], material["policies"], method, material["class_count"],
                    seed, config, settings["regularization"], progress,
                )
                probabilities = predict_cosine(model, values[2], config["batch_size"])
                checkpoint = destination / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "dataset": dataset, "method": method,
                    "seed": seed, "heldout_policy": policy, "configuration": config,
                    "regularization": settings["regularization"], "normalization": normalization,
                    "protocol_sha256": protocol_hash,
                }, checkpoint)
                predictions_path = destination / "challenge_predictions.jsonl"
                with predictions_path.open("x", encoding="utf-8") as stream:
                    for row, probability in zip(values[2]["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "prediction": int(probability.argmax()),
                            "probabilities": probability.tolist(),
                        }) + "\n")
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": policy, "method": method, "fit": fitted,
                    "challenge": metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]),
                    "checkpoint_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(predictions_path),
                    "protocol_sha256": protocol_hash, "original_val_or_test_access": False,
                    "challenge_used_for_checkpoint_selection": False, "claim_scope": SCOPE,
                }
                write_new_json(destination / "metrics.json", record)
                rows.append(record)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": rows, "submission_allowed": False, "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("cosine audit cannot be overwritten")
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("frozen cosine source changed")
    material = prepare(data_dir, temporal, partitions, dataset)
    if any(material[key] != frozen[key] for key in ("input_hashes", "prior_partition_hashes")):
        raise ValueError("cosine replay input changed")
    config, protocol_hash = frozen["configuration"], sha256_file(root / dataset / "protocol.json")
    rows = []
    for seed in SEEDS:
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            folder = root / dataset / f"seed{seed}" / policy
            if normalization != json.loads((folder / "normalization.json").read_text()):
                raise ValueError("cosine replay normalization differs")
            for method in METHODS:
                destination = folder / method
                record = json.loads((destination / "metrics.json").read_text())
                checkpoint, predictions_path = destination / "selected_model.pt", destination / "challenge_predictions.jsonl"
                if (sha256_file(checkpoint) != record["checkpoint_sha256"]
                        or sha256_file(predictions_path) != record["predictions_sha256"]):
                    raise ValueError("cosine checkpoint or predictions changed")
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                for key, expected in (
                    ("dataset", dataset), ("method", method), ("seed", seed), ("heldout_policy", policy),
                    ("configuration", config), ("regularization", frozen["regularization"]),
                    ("normalization", normalization), ("protocol_sha256", protocol_hash),
                ):
                    if saved[key] != expected:
                        raise ValueError("cosine checkpoint identity differs")
                model = make_cosine_model(values[2], material["class_count"], config, frozen["regularization"]["cosine_scale"])
                model.load_state_dict(saved["state_dict"], strict=True)
                probabilities = predict_cosine(model, values[2], config["batch_size"])
                predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in predictions] != [
                    (row["sample_id"], row["base_profile_id"], row["label"]) for row in values[2]["metadata"]
                ]:
                    raise ValueError("cosine prediction rows differ")
                difference = float(np.abs(probabilities - np.asarray(
                    [row["probabilities"] for row in predictions],
                )).max())
                if (difference > 1e-6 or probabilities.argmax(1).tolist() != [row["prediction"] for row in predictions]
                        or metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]) != record["challenge"]):
                    raise ValueError("cosine checkpoint failed replay")
                rows.append({"seed": seed, "heldout_policy": policy, "method": method,
                             "checkpoint_sha256": record["checkpoint_sha256"],
                             "predictions_sha256": record["predictions_sha256"],
                             "maximum_probability_difference": difference})
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
    replayed = {(row["seed"], row["heldout_policy"], row["method"]): row for row in audit["checkpoint_replays"]}
    if ({(row["seed"], row["heldout_policy"], row["method"]) for row in rows} != expected
            or len(rows) != 18 or set(replayed) != expected or len(audit["checkpoint_replays"]) != 18
            or not audit["all_replayed"] or audit["original_val_or_test_access"]
            or summary["submission_allowed"] or summary["protocol_sha256"] != protocol_hash
            or frozen["source_hashes"] != source_hashes()
            or frozen["original_val_paths_opened"] or frozen["test_paths_opened"]):
        raise ValueError("incomplete cosine evidence")
    for row in rows:
        folder = root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
        audited = replayed[(row["seed"], row["heldout_policy"], row["method"])]
        if (row != json.loads((folder / "metrics.json").read_text())
                or row["protocol_sha256"] != protocol_hash or row["original_val_or_test_access"]
                or row["challenge_used_for_checkpoint_selection"]
                or sha256_file(folder / "selected_model.pt") != row["checkpoint_sha256"]
                or sha256_file(folder / "challenge_predictions.jsonl") != row["predictions_sha256"]
                or audited["checkpoint_sha256"] != row["checkpoint_sha256"]
                or audited["predictions_sha256"] != row["predictions_sha256"]
                or audited["maximum_probability_difference"] > 1e-6):
            raise ValueError("cosine per-run provenance differs")
    return frozen, rows


def summarize(prior_root, replication_root, paired_root, root, output):
    if output.exists():
        raise FileExistsError("cosine aggregate cannot be overwritten")
    report = {
        "scope": SCOPE, "new_fits": 36, "new_checkpoint_replays": 36, "cached_fits": 72,
        "primary_task": "Quadbot", "transfer_task": "Twibot22",
        "submission_allowed": False, "original_val_or_test_access": False, "datasets": {},
    }
    lines = [
        "# Cosine Policy Development Screen", "",
        "Quadbot is the core task; Twibot22 is transfer evidence. Both views share profiles.",
        "This is post-hoc training-only development, not independent submission confirmation.",
        "The ordinary linear-head removal is reused from the preceding contrast-only condition.",
        "No failed paired-margin model is renamed or promoted. ERM removes both proposed components.",
        "Macro-F1 is percent; SD is sample SD across seeds, which also change inner selection folds.", "",
        "| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        frozen, new_rows = verified_candidates(root, dataset)
        prior, old_rows, provenance = cached(prior_root, replication_root, paired_root, dataset)
        validate_settings(frozen, prior)
        if provenance != frozen["cached_provenance"] or any(
            frozen[key] != prior[key] for key in ("input_hashes", "prior_partition_hashes")
        ):
            raise ValueError("cosine cached evidence changed")
        indexed = {(row["seed"], row["heldout_policy"], row["method"]): row for row in new_rows + old_rows}
        tables = {}
        for policy in POLICIES:
            tables[policy] = {}
            for method in ALL_METHODS:
                selected = [indexed[(seed, policy, method)] for seed in SEEDS]
                scores = describe([row["challenge"]["macro_f1"] for row in selected])
                counts = {row["fit"]["parameter_count"] for row in selected}
                if len(counts) != 1:
                    raise ValueError("cosine capacity changed across seeds")
                tables[policy][method] = {
                    "macro_f1": scores, "parameter_count": counts.pop(),
                    "per_class_f1_by_seed": [row["challenge"]["per_class_f1"] for row in selected],
                }
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, policy, method, " | ".join(f"{x * 100:.2f}" for x in scores["values"]),
                    scores["mean"] * 100, scores["seed_sd"] * 100,
                ))
        primary = {}
        expected_identities = {}
        for method in ALL_METHODS:
            per_seed, prediction_hashes = [], {}
            for seed in SEEDS:
                pooled = []
                for policy in POLICIES:
                    record = indexed[(seed, policy, method)]
                    path = (
                        root / dataset / f"seed{seed}" / policy / method / "challenge_predictions.jsonl"
                        if method in METHODS else Path(record["prediction_path"])
                    )
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    identity = [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows]
                    key = seed, policy
                    if key not in expected_identities:
                        expected_identities[key] = identity
                    if identity != expected_identities[key]:
                        raise ValueError("pooled method row identities differ")
                    if metrics([row["label"] for row in rows], [row["probabilities"] for row in rows],
                               frozen["class_count"]) != record["challenge"]:
                        raise ValueError("pooled metrics differ from verified checkpoints")
                    prediction_hashes[f"seed{seed}/{policy}"] = sha256_file(path)
                    pooled.extend(rows)
                identity = [(row["base_profile_id"], row["label"]) for row in pooled]
                if len(identity) != len(set(identity)):
                    raise ValueError("pooled cosine challenges repeat a profile/class")
                per_seed.append(metrics([row["label"] for row in pooled],
                                        [row["probabilities"] for row in pooled], frozen["class_count"]))
            primary[method] = {
                "macro_f1": describe([row["macro_f1"] for row in per_seed]),
                "metrics_by_seed": per_seed, "prediction_hashes": prediction_hashes,
            }
        differences = {}
        for method in ALL_METHODS[1:]:
            delta = describe([left - right for left, right in zip(
                primary["cosine_policy"]["macro_f1"]["values"], primary[method]["macro_f1"]["values"],
            )])
            differences[method] = {
                **delta, "higher_seed_count": sum(value > 0 for value in delta["values"]),
                "at_least_one_point_higher_seed_count": sum(value >= .01 for value in delta["values"]),
            }
        report["datasets"][dataset] = {
            "new_runs_verified": 18, "cached_runs_verified": 36, "tables": tables,
            "primary_pooled": primary, "primary_pooled_full_minus_comparator": differences,
            "protocol_sha256": sha256_file(root / dataset / "protocol.json"),
        }
    lines.extend(["", "## Overall Pooled Macro-F1", "",
                  "| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for method, entry in data["primary_pooled"].items():
            score = entry["macro_f1"]
            lines.append("| {} | {} | {} | {:.2f} | {:.2f} |".format(
                dataset, method, " | ".join(f"{x * 100:.2f}" for x in score["values"]),
                score["mean"] * 100, score["seed_sd"] * 100,
            ))
    lines.extend(["", "| View | Full Minus | Mean Delta | Delta SD | Positive Seeds |",
                  "|---|---|---:|---:|---:|"])
    for dataset, data in report["datasets"].items():
        for method, delta in data["primary_pooled_full_minus_comparator"].items():
            lines.append(f"| {dataset} | {method} | {delta['mean'] * 100:+.2f} | "
                         f"{delta['seed_sd'] * 100:.2f} | {delta['higher_seed_count']}/3 |")
    lines.extend(["", "Positive counts are descriptive, not significance tests.",
                  "Cosine readout omits the C class biases, so capacity is not exactly matched.",
                  "HGB has global summaries; adapted BotDMM has extra unlabelled feature-kNN context.",
                  "The cosine formula is established technology, not a new general algorithm.",
                  "Do not promote a removal, omit transfer failures, tune scale after scoring or claim SOTA.", ""])
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
                parser.error("summarize needs all fixed evidence roots")
            summarize(args.prior_root, args.replication_root, args.paired_root, args.root, args.out_dir)
        else:
            if not all((args.data_dir, args.temporal_dir, args.partition_root, args.dataset)):
                parser.error("train/replay require original training-only inputs")
            common = args.data_dir, args.temporal_dir, args.partition_root, args.dataset
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay needs cosine evidence root")
                replay(*common, args.root, args.out_dir)
            else:
                if not all((args.protocol, args.prior_root, args.replication_root, args.paired_root)):
                    parser.error("train needs fixed protocol and cached evidence roots")
                run(*common, args.protocol, args.prior_root, args.replication_root, args.paired_root, args.out_dir)


if __name__ == "__main__":
    main()
