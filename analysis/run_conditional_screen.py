"""Fixed, training-only 2x2 screen of conditional routing and content bridging."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import describe, verified_runs
from analysis.run_factor_training_screen import POLICIES, fold, prepare, source_hashes as legacy_hashes
from src.conditional_evidence import METHODS, make_conditional_model, predict_conditional, train_conditional
from src.evidence_repair import metrics, write_new_json
from src.provenance import sha256_file


SEEDS = (42, 43, 44)
CONTROLS = ("erm_mlp", "hgb_statistics", "botdmm_timing")
DATASETS = ("Twibot22", "Quadbot")
SCOPE = "posthoc_training_only_conditional_bridge_development"
SOURCES = (
    "src/conditional_evidence.py", "analysis/run_conditional_screen.py",
    "configs/conditional_screen.json", "analysis/replicate_factor_training_screen.py",
)


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {**legacy_hashes(), **{name: sha256_file(code / name) for name in SOURCES}}


def validate_settings(settings, prior):
    if (settings["schema"] != "training-only-conditional-bridge-screen-v1"
            or tuple(settings["seeds"]) != SEEDS or tuple(settings["methods"]) != METHODS
            or tuple(settings["reused_controls"]) != CONTROLS
            or settings["configuration"] != prior["configuration"]
            or settings["cpu_threads"] != 2 or settings["submission_allowed"]
            or settings["novelty_established"] or not settings["no_hyperparameter_search"]
            or settings["original_val_or_test_paths_opened"]
            or not settings["no_timing_or_label_augmentation"]
            or settings["new_fits"] != 72 or settings["reused_control_fits"] != 54):
        raise ValueError("conditional screen requires all fixed conditions and unchanged recipe")
    if prior["source_hashes"] != legacy_hashes():
        raise ValueError("preserved training framework changed")


def cached_controls(prior_root, replication_root, dataset):
    prior, old = verified_runs(prior_root, dataset, (42,))
    extension, added = verified_runs(replication_root, dataset, (43, 44))
    if (prior["configuration"] != extension["configuration"]
            or prior["input_hashes"] != extension["input_hashes"]
            or prior["prior_partition_hashes"] != extension["prior_partition_hashes"]):
        raise ValueError("cached controls use different recipes or inputs")
    provenance = {
        "seed42_protocol": sha256_file(prior_root / dataset / "protocol.json"),
        "seed42_summary": sha256_file(prior_root / dataset / "summary.json"),
        "seed43_44_protocol": sha256_file(replication_root / dataset / "protocol.json"),
        "seed43_44_summary": sha256_file(replication_root / dataset / "summary.json"),
    }
    selected = [row for row in old + added if row["method"] in CONTROLS]
    if len(selected) != 27:
        raise ValueError("all three controls, seeds and policies are required")
    return prior, selected, provenance


def run(data_dir, temporal, partitions, dataset, protocol, prior_root, replication_root, output):
    if output.exists():
        raise FileExistsError("conditional screen evidence cannot be overwritten")
    settings = json.loads(protocol.read_text())
    prior, _, provenance = cached_controls(prior_root, replication_root, dataset)
    validate_settings(settings, prior)
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != prior[key]:
            raise ValueError("preserved training inputs differ")
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"],
        "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "cached_control_provenance": provenance,
        "device": "cpu", "test_paths_opened": False, "original_val_paths_opened": False,
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
                raise ValueError("new candidates are not paired with cached control folds")
            folder = output / f"seed{seed}" / policy
            write_new_json(folder / "normalization.json", normalization)
            for method in METHODS:
                destination = folder / method
                destination.mkdir()

                def progress(state):
                    print(json.dumps({
                        "event": "epoch", "dataset": dataset, "seed": seed,
                        "heldout_policy": policy, "method": method, **state,
                    }), flush=True)

                model, fitted = train_conditional(
                    values[0], values[1], material["policies"], method,
                    material["class_count"], seed, config, progress,
                )
                probabilities = predict_conditional(model, values[2], config["batch_size"])
                path = destination / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "dataset": dataset, "method": method,
                    "seed": seed, "heldout_policy": policy, "configuration": config,
                    "normalization": normalization, "protocol_sha256": protocol_hash,
                }, path)
                prediction_path = destination / "challenge_predictions.jsonl"
                with prediction_path.open("x", encoding="utf-8") as stream:
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
                    "checkpoint_sha256": sha256_file(path),
                    "predictions_sha256": sha256_file(prediction_path),
                    "protocol_sha256": protocol_hash,
                    **{name + "_profiles": len({row["base_profile_id"] for row in value["metadata"]})
                       for name, value in zip(("fitting", "selection", "challenge"), values)},
                    "original_val_or_test_access": False,
                    "challenge_used_for_checkpoint_selection": False, "claim_scope": SCOPE,
                }
                write_new_json(destination / "metrics.json", record)
                records.append(record)
                print(json.dumps({
                    "event": "run_complete", "dataset": dataset, "seed": seed,
                    "heldout_policy": policy, "method": method,
                }), flush=True)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": records, "submission_allowed": False,
        "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("conditional replay cannot be overwritten")
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("fixed conditional implementation changed")
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != frozen[key]:
            raise ValueError("fixed training inputs changed")
    config, rows = frozen["configuration"], []
    protocol_hash = sha256_file(root / dataset / "protocol.json")
    for seed in SEEDS:
        for policy in POLICIES:
            values, _, normalization = fold(material, seed, policy, config["knn_k"])
            folder = root / dataset / f"seed{seed}" / policy
            if normalization != json.loads((folder / "normalization.json").read_text()):
                raise ValueError("replay transforms differ")
            for method in METHODS:
                destination = folder / method
                record = json.loads((destination / "metrics.json").read_text())
                path = destination / "selected_model.pt"
                prediction_path = destination / "challenge_predictions.jsonl"
                if (sha256_file(path) != record["checkpoint_sha256"]
                        or sha256_file(prediction_path) != record["predictions_sha256"]):
                    raise ValueError("checkpoint or prediction bytes changed")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                for key, expected in (
                    ("dataset", dataset), ("method", method), ("seed", seed),
                    ("heldout_policy", policy), ("configuration", config),
                    ("normalization", normalization), ("protocol_sha256", protocol_hash),
                ):
                    if saved[key] != expected:
                        raise ValueError("checkpoint identity differs: " + key)
                model = make_conditional_model(values[2], material["class_count"], config, method)
                model.load_state_dict(saved["state_dict"], strict=True)
                probabilities = predict_conditional(model, values[2], config["batch_size"])
                predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in predictions] != [
                    (row["sample_id"], row["base_profile_id"], row["label"])
                    for row in values[2]["metadata"]
                ]:
                    raise ValueError("replay row identities differ")
                difference = float(np.max(np.abs(
                    probabilities - np.asarray([row["probabilities"] for row in predictions]),
                )))
                if (difference > 1e-6
                        or [row["prediction"] for row in predictions] != probabilities.argmax(1).tolist()
                        or metrics(values[2]["labels"].numpy(), probabilities, material["class_count"])
                        != record["challenge"]):
                    raise ValueError("checkpoint predictions or metrics did not replay")
                rows.append({
                    "seed": seed, "heldout_policy": policy, "method": method,
                    "maximum_probability_difference": difference,
                    "checkpoint_sha256": sha256_file(path),
                    "predictions_sha256": sha256_file(prediction_path),
                })
    write_new_json(output / "audit.json", {
        "dataset": dataset, "all_replayed": True, "checkpoint_replays": rows,
        "original_val_or_test_access": False, "submission_allowed": False,
    })


def verified_candidates(root, dataset):
    frozen_path = root / dataset / "protocol.json"
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads((root / dataset / "summary.json").read_text())
    audit = json.loads((root / (dataset + "_audit") / "audit.json").read_text())
    rows = summary["runs"]
    expected = {(seed, policy, method) for seed in SEEDS for policy in POLICIES for method in METHODS}
    observed = {(row["seed"], row["heldout_policy"], row["method"]) for row in rows}
    replayed = {
        (row["seed"], row["heldout_policy"], row["method"]): row
        for row in audit["checkpoint_replays"]
    }
    protocol_hash = sha256_file(frozen_path)
    if (observed != expected or len(rows) != len(expected) or set(replayed) != expected
            or len(audit["checkpoint_replays"]) != len(expected) or not audit["all_replayed"]
            or audit["original_val_or_test_access"] or summary["submission_allowed"]
            or frozen["test_paths_opened"] or frozen["original_val_paths_opened"]
            or summary["protocol_sha256"] != protocol_hash
            or frozen["source_hashes"] != source_hashes()):
        raise ValueError("incomplete or changed conditional evidence")
    for row in rows:
        folder = root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
        evidence = replayed[(row["seed"], row["heldout_policy"], row["method"])]
        if (row != json.loads((folder / "metrics.json").read_text())
                or row["protocol_sha256"] != protocol_hash
                or row["original_val_or_test_access"] or row["challenge_used_for_checkpoint_selection"]
                or sha256_file(folder / "selected_model.pt") != row["checkpoint_sha256"]
                or sha256_file(folder / "challenge_predictions.jsonl") != row["predictions_sha256"]
                or evidence["checkpoint_sha256"] != row["checkpoint_sha256"]
                or evidence["predictions_sha256"] != row["predictions_sha256"]
                or evidence["maximum_probability_difference"] > 1e-6):
            raise ValueError("per-run provenance differs")
    return frozen, rows


def summarize(prior_root, replication_root, root, output):
    if output.exists():
        raise FileExistsError("conditional aggregate cannot be overwritten")
    report = {
        "claim_scope": SCOPE, "seeds": list(SEEDS), "new_fits": 72,
        "reused_control_fits": 54, "new_checkpoint_replays": 72,
        "submission_allowed": False, "original_val_or_test_access": False, "datasets": {},
    }
    lines = [
        "# Conditional Routing and Content Bridge Screen", "",
        "This is a fixed post-hoc training-only development screen, not final submission evidence.",
        "All four architectural conditions, three seeds and three policies are retained.",
        "Controls are reused only after their input, recipe, fold and checkpoint identity checks.",
        "Macro-F1 values are percentages; SD is sample SD across the three seeds.", "",
        "| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        frozen, new_rows = verified_candidates(root, dataset)
        prior, controls, provenance = cached_controls(prior_root, replication_root, dataset)
        settings_keys = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/conditional_screen.json").read_text()
        ).keys()
        validate_settings({key: frozen[key] for key in settings_keys}, prior)
        if frozen["cached_control_provenance"] != provenance:
            raise ValueError("cached control artifacts changed")
        for key in ("input_hashes", "prior_partition_hashes"):
            if frozen[key] != prior[key]:
                raise ValueError("candidate/control input identity differs")
        indexed = {
            (row["seed"], row["heldout_policy"], row["method"]): row
            for row in new_rows + controls
        }
        tables, comparisons = {}, {}
        for policy in POLICIES:
            tables[policy], comparisons[policy] = {}, {}
            for method in (*METHODS, *CONTROLS):
                selected = [indexed[(seed, policy, method)] for seed in SEEDS]
                scores = describe([row["challenge"]["macro_f1"] for row in selected])
                counts = {row["fit"]["parameter_count"] for row in selected}
                if len(counts) != 1:
                    raise ValueError("model capacity changed across seeds")
                tables[policy][method] = {
                    "macro_f1": scores, "parameter_count": counts.pop(),
                    "per_class_f1_by_seed": [row["challenge"]["per_class_f1"] for row in selected],
                }
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, policy, method,
                    " | ".join("{:.2f}".format(value * 100) for value in scores["values"]),
                    scores["mean"] * 100, scores["seed_sd"] * 100,
                ))
            for comparator in (*METHODS[1:], *CONTROLS):
                delta = describe([
                    indexed[(seed, policy, METHODS[0])]["challenge"]["macro_f1"]
                    - indexed[(seed, policy, comparator)]["challenge"]["macro_f1"]
                    for seed in SEEDS
                ])
                comparisons[policy][comparator] = {
                    **delta, "higher_seed_count": sum(value > 0 for value in delta["values"]),
                    "at_least_one_point_higher_seed_count": sum(value >= .01 for value in delta["values"]),
                }
        report["datasets"][dataset] = {
            "new_runs_verified": len(new_rows), "cached_controls_verified": len(controls),
            "tables": tables, "full_minus_comparator": comparisons,
            "protocol_sha256": sha256_file(root / dataset / "protocol.json"),
            "cached_control_provenance": provenance,
        }
    lines.extend([
        "", "## Full Minus Removals and Controls", "",
        "Differences are percentage points. Positive seed counts are descriptive, not significance tests.",
        "", "| View | Policy | Comparator | Mean Delta | Delta SD | Positive Seeds |",
        "|---|---|---|---:|---:|---:|",
    ])
    for dataset, evidence in report["datasets"].items():
        for policy, comparisons in evidence["full_minus_comparator"].items():
            for comparator, delta in comparisons.items():
                lines.append("| {} | {} | {} | {:+.2f} | {:.2f} | {}/3 |".format(
                    dataset, policy, comparator, delta["mean"] * 100,
                    delta["seed_sd"] * 100, delta["higher_seed_count"],
                ))
    lines.extend([
        "", "## Boundary", "",
        "Removing the bridge physically omits its projection. Removing conditional routing",
        "replaces hierarchical leaves with a flat joint head but retains both auxiliary heads",
        "and the same masked subtype supervision. Capacity is not matched across removals.",
        "The content/behavior groups overlap; event hashes contain text. No causal separation is claimed.",
        "HGB has global summaries; adapted BotDMM has extra transductive feature-kNN context.",
        "The two views share profiles. More seeds do not establish independent generalization or SOTA.",
        "No candidate is automatically promoted to a final model; negative outcomes remain diagnostic.",
        "Original datasets, existing results and submitted manuscript remain unchanged.", "",
    ])
    write_new_json(output / "development_summary.json", report)
    (output / "development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "replay", "summarize"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    for name in (
        "data-dir", "temporal-dir", "partition-root", "protocol",
        "prior-root", "replication-root", "root", "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=name == "out-dir")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "summarize":
            if not all((args.prior_root, args.replication_root, args.root)):
                parser.error("summarize requires both control roots and the new root")
            summarize(args.prior_root, args.replication_root, args.root, args.out_dir)
        else:
            if not all((args.data_dir, args.temporal_dir, args.partition_root, args.dataset)):
                parser.error("train/replay require dataset and training inputs")
            common = (args.data_dir, args.temporal_dir, args.partition_root, args.dataset)
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay requires root")
                replay(*common, args.root, args.out_dir)
            else:
                if not all((args.protocol, args.prior_root, args.replication_root)):
                    parser.error("train requires protocol and both preserved control roots")
                run(*common, args.protocol, args.prior_root, args.replication_root, args.out_dir)


if __name__ == "__main__":
    main()
