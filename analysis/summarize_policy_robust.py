"""Recompute complete policy-study tables and descriptive paired intervals."""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.paired_profile_cluster_bootstrap import aligned_dataset, analyze_dataset
from src.evidence_repair import metrics, write_new_json
from src.provenance import sha256_file


def verified_records(root, dataset):
    source = root / dataset
    frozen = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    protocol_hash = sha256_file(source / "protocol.json")
    if frozen["test_paths_opened"] or frozen["outer_challenge_used_for_selection"]:
        raise ValueError("test or challenge-selection access invalidates study boundary")
    if summary["test_state"] != "NOT_OPENED" or summary["submission_allowed"]:
        raise ValueError("development evidence cannot authorize submission")
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("summary protocol hash differs")
    methods, seeds = frozen["methods"], frozen["seeds"]
    records = {}
    for row in summary["runs"]:
        key = (row["seed"], row["heldout_policy"], row["method"])
        if key in records:
            raise ValueError("duplicate run")
        if row["test_state"] != "NOT_OPENED" or row["outer_challenge_used_for_selection"]:
            raise ValueError("run violates development boundaries")
        if row["dataset"] != dataset or row["protocol_sha256"] != protocol_hash:
            raise ValueError("run identity or protocol mismatch")
        path = source / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
        saved = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if saved != row:
            raise ValueError("saved run differs from summary")
        predictions = [
            json.loads(line) for line in
            (path / "challenge_predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels = np.asarray([entry["label"] for entry in predictions], dtype=np.int64)
        probabilities = np.asarray([entry["probabilities"] for entry in predictions])
        keys = [(entry["base_profile_id"], entry["label"]) for entry in predictions]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate prediction profile/class pair")
        profile_ids = {profile for profile, _ in keys}
        if len(profile_ids) != row["challenge_profiles"] or set(keys) != {
            (profile, label) for profile in profile_ids
            for label in range(frozen["class_count"])
        }:
            raise ValueError("challenge profiles are not class-complete")
        if any(entry["prediction"] != int(probability.argmax())
               for entry, probability in zip(predictions, probabilities)):
            raise ValueError("prediction does not match probabilities")
        recomputed = metrics(labels, probabilities, frozen["class_count"])
        for name in ("macro_f1", "accuracy", "mcc"):
            if not np.isclose(recomputed[name], row["challenge"][name], atol=1e-12):
                raise ValueError("reported metric differs from row predictions")
        if not np.allclose(recomputed["per_class_f1"],
                           row["challenge"]["per_class_f1"], atol=1e-12):
            raise ValueError("reported per-class metric differs from row predictions")
        records[key] = row
    policies = sorted({key[1] for key in records})
    expected = {(seed, policy, method) for seed in seeds
                for policy in policies for method in methods}
    if set(records) != expected or len(policies) != 3:
        raise ValueError("incomplete seeds, policies, or methods")
    return frozen, records


def summarize(root, out, replicates=2000, base_seed=20261013):
    if out.exists():
        raise FileExistsError("summary output is immutable")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    report = {
        "claim_scope": "posthoc_development_only", "test_access": False,
        "submission_allowed": False, "bootstrap_replicates": replicates,
        "bootstrap_scope": "descriptive_profile_resampling_fixed_training_seeds_not_confirmation",
        "datasets": {},
    }
    lines = [
        "# Policy Robustness Development Results", "",
        "This is a post-hoc development screen on previously inspected validation",
        "challenges. It is not an untouched test, an established final method, or",
        "evidence of SOTA. The main account data and old paper results are unchanged.", "",
    ]
    for dataset in ("Twibot22", "Quadbot"):
        frozen, records = verified_records(root, dataset)
        policies = sorted({key[1] for key in records})
        tables, intervals = {}, {}
        lines.extend([f"## {dataset}", ""])
        for index, policy in enumerate(policies):
            tables[policy] = {}
            lines.extend([
                f"### {policy}", "",
                "| Method | Macro-F1 mean +/- sample seed SD |",
                "|---|---:|",
            ])
            block = {"seeds": {}}
            for seed in frozen["seeds"]:
                block["seeds"][str(seed)] = {
                    method: {"path": str(root / dataset / f"seed{seed}" / policy
                                         / method / "challenge_predictions.jsonl"),
                             "pred_key": "prediction"}
                    for method in frozen["methods"]
                }
            data = aligned_dataset(root / "unused_manifest.json", dataset, block)
            if len(data["classes"]) != frozen["class_count"]:
                raise ValueError("challenge classes are incomplete")
            paired = analyze_dataset(
                data, ["erm_mlp", "hgb_statistics"], replicates, base_seed + index
            )
            intervals[policy] = paired
            for method in frozen["methods"]:
                rows = [records[(seed, policy, method)] for seed in frozen["seeds"]]
                table = {
                    name: {"mean": float(np.mean([row["challenge"][name] for row in rows])),
                           "seed_sd": float(np.std(
                               [row["challenge"][name] for row in rows], ddof=1))}
                    for name in ("macro_f1", "accuracy", "mcc")
                }
                tables[policy][method] = table
                macro = table["macro_f1"]
                lines.append(f"| {method} | {macro['mean']:.4f} "
                             f"+/- {macro['seed_sd']:.4f} |")
            difference = paired["aggregate_comparisons"][
                "group_dro_rdrop_mlp_minus_erm_mlp"
            ]["macro_f1"]
            lines.extend([
                "",
                f"- {policy}: combined-minus-ERM descriptive Macro-F1 difference "
                f"{difference['estimate']:.4f}, profile-resampling 95% interval "
                f"[{difference['lower_95']:.4f}, {difference['upper_95']:.4f}].",
                "",
            ])
        overall = {}
        for method in frozen["methods"]:
            seed_scores = [
                np.mean([records[(seed, policy, method)]["challenge"]["macro_f1"]
                         for policy in policies])
                for seed in frozen["seeds"]
            ]
            overall[method] = {
                "equal_policy_mean_macro_f1": float(np.mean(seed_scores)),
                "seed_sd_of_equal_policy_mean": float(np.std(seed_scores, ddof=1)),
            }
        report["datasets"][dataset] = {
            "runs_verified": len(records), "tables": tables,
            "equal_policy_averages_not_pooled_scores": overall,
            "paired_intervals": intervals, "source_protocol_sha256": sha256_file(
                root / dataset / "protocol.json"
            ),
        }
    lines.extend([
        "## Interpretation Boundary", "",
        "All four neural methods have the same inference architecture and inputs.",
        "The comparison changes training objectives, not architectural modules.",
        "R-Drop uses two training forward passes, so epoch budgets are equal but",
        "computational work is not. HGB uses global statistics, not identical",
        "flattened encoding. The current ERM control also uses the newly frozen",
        "weight decay, profile batching, and worst-known-policy checkpoint rule;",
        "its differences from the earlier MLP screen are not objective-only gains.", "",
        "Intervals resample whole paired profiles synchronously across fixed",
        "training seeds. Prior challenge exposure and multiple explored mechanisms",
        "prevent treating them as confirmatory significance or external robustness.", "",
        "No result is filtered because an ablation is better. A failed combination",
        "must not be relabelled as a successful final method.", "",
    ])
    out.mkdir(parents=True)
    write_new_json(out / "development_summary.json", report)
    with (out / "development_summary.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    summarize(args.root, args.out_dir, args.replicates)


if __name__ == "__main__":
    main()
