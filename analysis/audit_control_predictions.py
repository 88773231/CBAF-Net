"""Read-only diagnosis of already exposed policy-control predictions.

No raw tensor, checkpoint, fitting, threshold selection, or model selection is
performed. The caller supplies the previously verified private export.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from src.evidence_repair import metrics, write_new_json
from src.provenance import sha256_file


DATASETS = {"Twibot22": 3, "Quadbot": 4}
SEEDS = (42, 43, 44)
POLICIES = ("burst_scheduler", "cyclic_reposter", "uniform_scheduler")
METHODS = ("flat_policy", "botsscl_flattened", "botsscl_supervised_only")
NAMES = {
    "flat_policy": "SAM + policy contrast",
    "botsscl_flattened": "BotSSCL-mechanism control",
    "botsscl_supervised_only": "Same encoder without pretraining",
}
CLASSES = ("Human", "Traditional Bot", "LLM Bot", "Full-stack Agent")
REFERENCE = "flat_policy_screen_20261013"
CONTROL = "botsscl_policy_control_20261014"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "values": values.tolist(), "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
    }


def verify_export(root, receipt_path):
    receipt = read_json(receipt_path)
    manifest = receipt["archive_member_hashes"]
    if not manifest or not receipt["private_archive_not_for_public_release"]:
        raise ValueError("a private exported-evidence manifest is required")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != expected:
            raise ValueError("private export path or digest differs: " + name)
    return manifest


def validate_predictions(rows, classes):
    if not rows:
        raise ValueError("empty predictions")
    identifiers, groups = set(), {}
    for row in rows:
        if (not isinstance(row["sample_id"], str) or not row["sample_id"]
                or not isinstance(row["base_profile_id"], str) or not row["base_profile_id"]
                or type(row["label"]) is not int or type(row["prediction"]) is not int
                or not 0 <= row["label"] < classes or not 0 <= row["prediction"] < classes):
            raise ValueError("invalid prediction identity or taxonomy")
        if row["sample_id"] in identifiers:
            raise ValueError("duplicate sample identifier")
        identifiers.add(row["sample_id"])
        members = groups.setdefault(row["base_profile_id"], set())
        if row["label"] in members:
            raise ValueError("duplicate profile/class pair")
        members.add(row["label"])
    if any(labels != set(range(classes)) for labels in groups.values()):
        raise ValueError("class-incomplete profile")
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
    labels = np.asarray([row["label"] for row in rows])
    scores = metrics(labels, probabilities, classes)
    if probabilities.argmax(1).tolist() != [row["prediction"] for row in rows]:
        raise ValueError("prediction differs from probability argmax")
    return scores


def row_identity(rows):
    return [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows]


def assert_same_rows(left, right):
    if row_identity(left) != row_identity(right):
        raise ValueError("paired ordered prediction identities differ")


def prediction_summary(rows, classes):
    scores = validate_predictions(rows, classes)
    labels = np.asarray([row["label"] for row in rows])
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=float)
    predicted = probabilities.argmax(1)
    correct = labels == predicted
    confidence = probabilities.max(1)
    confusion = confusion_matrix(labels, predicted, labels=list(range(classes)))
    boundary_errors = int(confusion[0, 1] + confusion[1, 0])
    errors = int((~correct).sum())
    return {
        **scores, "rows": len(rows), "profiles": len({row["base_profile_id"] for row in rows}),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "error_count": errors, "human_traditional_mutual_errors": boundary_errors,
        "human_traditional_fraction_of_errors": boundary_errors / errors if errors else None,
        "negative_log_likelihood": float(-np.log(np.maximum(
            probabilities[np.arange(len(rows)), labels], np.finfo(float).tiny,
        )).mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
    }


def disagreement(left, right, classes):
    assert_same_rows(left, right)
    labels = np.asarray([row["label"] for row in left])
    first = np.asarray([row["prediction"] for row in left])
    second = np.asarray([row["prediction"] for row in right])
    first_correct, second_correct = first == labels, second == labels

    def cells(mask):
        return {
            "rows": int(mask.sum()),
            "both_correct": int((mask & first_correct & second_correct).sum()),
            "candidate_only_correct": int((mask & first_correct & ~second_correct).sum()),
            "control_only_correct": int((mask & ~first_correct & second_correct).sum()),
            "both_wrong": int((mask & ~first_correct & ~second_correct).sum()),
            "prediction_disagreements": int((mask & (first != second)).sum()),
        }

    return {
        "all": cells(np.ones(len(labels), dtype=bool)),
        "by_true_class": {CLASSES[label]: cells(labels == label) for label in range(classes)},
        "candidate_predictions_rows_control_predictions_columns": confusion_matrix(
            first, second, labels=list(range(classes)),
        ).tolist(),
    }


def selected_history(record):
    history = record["fit"]["history"]
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("training history epochs differ")
    allowed = set(POLICIES) - {record["heldout_policy"]}
    for row in history:
        score = row["inner_selection"]
        if set(score["by_fitting_policy"]) != allowed:
            raise ValueError("excluded policy entered inner checkpoint selection")
        worst = min(value["macro_f1"] for value in score["by_fitting_policy"].values())
        if not np.isclose(worst, score["worst_policy_macro_f1"], atol=1e-12, rtol=0):
            raise ValueError("worst known-policy selection score differs")

    def key(row):
        score = row["inner_selection"]
        return (score["worst_policy_macro_f1"], score["all_profiles"]["macro_f1"],
                score["all_profiles"]["mcc"])

    chosen = max(history, key=key)
    if chosen["epoch"] != record["fit"]["best_epoch"]:
        raise ValueError("selected epoch differs from frozen inner-selection rule")
    score = chosen["inner_selection"]
    return {
        "selected_epoch": chosen["epoch"], "epochs_completed": len(history),
        "known_policy_selection_macro_f1": score["all_profiles"]["macro_f1"],
        "worst_known_policy_macro_f1": score["worst_policy_macro_f1"],
        "excluded_policy_macro_f1": record["challenge"]["macro_f1"],
        "selection_minus_excluded_policy_macro_f1":
            score["all_profiles"]["macro_f1"] - record["challenge"]["macro_f1"],
        "not_a_same_population_generalization_estimate": True,
    }


def verify_protocols(root, dataset):
    reference_path = root / REFERENCE / dataset / "protocol.json"
    reference = read_json(reference_path)
    control = read_json(root / CONTROL / dataset / "protocol.json")
    for protocol in (reference, control):
        if (protocol["dataset"] != dataset or protocol["class_count"] != DATASETS[dataset]
                or protocol["seeds"] != list(SEEDS) or protocol["submission_allowed"]
                or protocol["test_paths_opened"] or protocol["original_val_paths_opened"]
                or protocol["challenge_used_for_checkpoint_selection"]):
            raise ValueError("protocol taxonomy, population, or access boundary differs")
    for key in ("input_hashes", "prior_partition_hashes", "configuration"):
        if reference[key] != control[key]:
            raise ValueError("candidate/control " + key + " differ")
    if control["reference_protocol_sha256"] != sha256_file(reference_path):
        raise ValueError("control reference protocol digest differs")
    code = Path(__file__).resolve().parents[1]
    for protocol in (reference, control):
        for name, expected in protocol["source_hashes"].items():
            source = (code / name).resolve()
            if not source.is_relative_to(code) or sha256_file(source) != expected:
                raise ValueError("frozen training source differs: " + name)
    return reference, control


def verify_normalization(root, manifest, dataset, seed, policy, prior):
    path = root / CONTROL / dataset / f"seed{seed}" / policy / "normalization.json"
    name = f"{REFERENCE}/{dataset}/main/init{seed}_fold{seed}/{policy}/normalization.json"
    if sha256_file(path) != prior[name]:
        raise ValueError("candidate/control fitting normalization differs")
    if path.relative_to(root).as_posix() not in manifest:
        raise ValueError("normalization absent from export receipt")


def load_run(root, dataset, seed, policy, method, protocol_hash, control_runs, replays):
    base = (
        root / REFERENCE / dataset / "main" / f"init{seed}_fold{seed}" / policy / method
        if method == "flat_policy"
        else root / CONTROL / dataset / f"seed{seed}" / policy / method
    )
    record = read_json(base / "metrics.json")
    prediction = base / "challenge_predictions.jsonl"
    if (any(record[key] != expected for key, expected in (
        ("dataset", dataset), ("seed", seed), ("heldout_policy", policy), ("method", method),
        ("protocol_sha256", protocol_hash),
    )) or record["original_val_or_test_access"] or record["challenge_used_for_checkpoint_selection"]
            or record["predictions_sha256"] != sha256_file(prediction)):
        raise ValueError("run identity, prediction digest, or access boundary differs")
    if method == "flat_policy":
        if record["scheme"] != "main" or record["partition_seed"] != seed:
            raise ValueError("candidate main partition differs")
    else:
        identity = seed, policy, method
        replay = replays[identity]
        if (record != control_runs[identity]
                or record["checkpoint_sha256"] != replay["checkpoint_sha256"]
                or record["predictions_sha256"] != replay["predictions_sha256"]
                or replay["maximum_probability_difference"] > 1e-6):
            raise ValueError("preserved control replay differs")
    rows = [json.loads(line) for line in prediction.read_text(encoding="utf-8").splitlines()]
    actual = validate_predictions(rows, DATASETS[dataset])
    if actual != record["challenge"]:
        raise ValueError("recomputed per-run metrics differ")
    return rows, record


def shared_view_comparison(left, right):
    """Compare shared rows without merging four-class probabilities or labels."""
    first = {(row["base_profile_id"], row["label"]): row for row in left}
    second = {(row["base_profile_id"], row["label"]): row for row in right if row["label"] < 3}
    if set(first) != set(second) or any(
        first[key]["sample_id"] != second[key]["sample_id"] for key in first
    ):
        raise ValueError("nested-view profile/class/sample identities differ")
    cells = {"shared_rows": len(first), "both_correct": 0, "three_class_only_correct": 0,
             "four_class_only_correct": 0, "both_wrong": 0, "four_class_predicts_agent": 0}
    for key, row in first.items():
        other = second[key]
        good, other_good = row["prediction"] == row["label"], other["prediction"] == row["label"]
        name = ("both_correct" if good and other_good else
                "three_class_only_correct" if good else
                "four_class_only_correct" if other_good else "both_wrong")
        cells[name] += 1
        cells["four_class_predicts_agent"] += int(other["prediction"] == 3)
    return {**cells, "different_training_tasks_not_a_causal_head_intervention": True}


def audit(root, receipt_path, output):
    if output.exists():
        raise FileExistsError("preserve existing read-only diagnostic")
    root = root.resolve()
    manifest = verify_export(root, receipt_path)
    prior = read_json(root / CONTROL / "preserved_reference_manifest.json")
    candidate_summary = read_json(root / REFERENCE / "summary/development_summary.json")
    control_summary = read_json(root / CONTROL / "summary/development_summary.json")
    report = {
        "scope": "read_only_diagnosis_of_exposed_training_policy_predictions",
        "new_research_fits": 0, "raw_inputs_opened": False, "checkpoint_weights_opened": False,
        "threshold_or_recipe_search": False, "independent_confirmation": False,
        "submission_allowed": False, "private_prediction_identifiers_exported": False,
        "source_export_receipt_sha256": sha256_file(receipt_path),
        "source_archive_sha256": read_json(receipt_path)["archive_sha256"],
        "archive_member_hashes_verified": len(manifest),
        "aggregation": "metrics_pool_policies_within_seed_then_average_three_seeds",
        "error_counts": "prediction_events_repeated_for_three_selected_models_not_independent_rows",
        "datasets": {}, "shared_view_comparison": {},
    }
    all_predictions = {}
    for dataset, classes in DATASETS.items():
        reference, control = verify_protocols(root, dataset)
        summary = read_json(root / CONTROL / dataset / "summary.json")
        audit_record = read_json(root / CONTROL / (dataset + "_audit") / "audit.json")
        identities = {(seed, policy, method) for seed in SEEDS for policy in POLICIES
                      for method in METHODS[1:]}
        key = lambda row: (row["seed"], row["heldout_policy"], row["method"])
        runs = {key(row): row for row in summary["runs"]}
        replays = {key(row): row for row in audit_record["checkpoint_replays"]}
        if (set(runs) != identities or len(summary["runs"]) != len(identities)
                or set(replays) != identities or len(audit_record["checkpoint_replays"]) != len(identities)
                or not audit_record["all_replayed"] or audit_record["original_val_or_test_access"]
                or audit_record["submission_allowed"] or summary["submission_allowed"]):
            raise ValueError("incomplete control conditions or checkpoint replay records")
        dataset_runs, pooled = [], {method: [] for method in METHODS}
        for seed in SEEDS:
            combined = {method: [] for method in METHODS}
            for policy in POLICIES:
                verify_normalization(root, manifest, dataset, seed, policy, prior)
                predictions, selections = {}, {}
                for method in METHODS:
                    protocol_path = root / (REFERENCE if method == "flat_policy" else CONTROL)
                    protocol_path /= dataset + "/protocol.json"
                    rows, record = load_run(
                        root, dataset, seed, policy, method, sha256_file(protocol_path), runs, replays,
                    )
                    assert_same_rows(rows, rows if method == "flat_policy" else predictions["flat_policy"])
                    predictions[method], selections[method] = rows, selected_history(record)
                    combined[method].extend(rows)
                dataset_runs.append({
                    "seed": seed, "excluded_policy": policy,
                    "conditions": {method: prediction_summary(rows, classes)
                                   for method, rows in predictions.items()},
                    "selected_checkpoints": selections,
                    "candidate_vs_control": {
                        method: disagreement(predictions["flat_policy"], predictions[method], classes)
                        for method in METHODS[1:]
                    },
                })
            for method, rows in combined.items():
                scores = prediction_summary(rows, classes)
                expected = (
                    candidate_summary if method == "flat_policy" else control_summary
                )["datasets"][dataset]["primary_pooled"][method]["metrics_by_seed"][seed - 42]
                if any(scores[name] != value for name, value in expected.items()):
                    raise ValueError("recomputed pooled seed metrics differ")
                pooled[method].append(scores)
                all_predictions[dataset, seed, method] = rows
        aggregates = {}
        for method in METHODS:
            values = pooled[method]
            confusion = np.asarray([
                value["confusion_true_rows_predicted_columns"] for value in values
            ]).sum(0)
            errors = sum(value["error_count"] for value in values)
            mutual = sum(value["human_traditional_mutual_errors"] for value in values)
            aggregates[method] = {
                "macro_f1": describe([value["macro_f1"] for value in values]),
                "per_class_f1": {CLASSES[label]: describe([
                    value["per_class_f1"][label] for value in values
                ]) for label in range(classes)},
                "prediction_events": sum(value["rows"] for value in values),
                "confusion_true_rows_predicted_columns": confusion.tolist(),
                "error_events": errors, "human_traditional_mutual_error_events": mutual,
                "human_traditional_fraction_of_errors": mutual / errors if errors else None,
            }
        report["datasets"][dataset] = {
            "verified_input_file_hashes": len(control["input_hashes"]),
            "verified_normalization_pairs": len(SEEDS) * len(POLICIES),
            "input_partition_configuration_and_frozen_sources_match": True,
            "architecture_or_total_compute_match": False,
            "profiles_per_seed": pooled[METHODS[0]][0]["profiles"],
            "runs": dataset_runs, "pooled_by_seed": pooled, "aggregates": aggregates,
            "candidate_minus_control": {
                method: describe([a["macro_f1"] - b["macro_f1"] for a, b in
                                  zip(pooled["flat_policy"], pooled[method])])
                for method in METHODS[1:]
            },
        }
    for method in METHODS:
        report["shared_view_comparison"][method] = [
            {"seed": seed, **shared_view_comparison(
                all_predictions["Twibot22", seed, method],
                all_predictions["Quadbot", seed, method],
            )} for seed in SEEDS
        ]
    write_new_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-evidence-root", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.private_evidence_root, args.export_receipt, args.output)
    print(json.dumps({
        "new_fits": 0, "verified_export_members": report["archive_member_hashes_verified"],
        "datasets": {
            dataset: {method: value["macro_f1"] for method, value in data["aggregates"].items()}
            for dataset, data in report["datasets"].items()
        }, "submission_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
