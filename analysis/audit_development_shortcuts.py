"""Profile-disjoint shortcut probes on train/validation, never held-out test."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from dataset_pipeline.build_dataset_views import opaque_id
from src.bse_config import release_bse_parameters
from src.data_loader import fit_tabular_standardizers
from src.evidence_repair import assert_group_disjoint, load_split, metrics, write_new_json
from src.provenance import sha256_file


STYLE_NAMES = (
    "post_count", "mean_post_length", "post_length_sd", "lexical_diversity",
    "punctuation_ratio", "url_per_post", "hashtag_per_post",
    "uppercase_ratio", "non_ascii_ratio", "repost_ratio", "interaction_ratio",
)
NUM_NAMES = ("followers", "following", "historical_posts", "listed", "posts", "actions")


def collect_provenance(master):
    policies, generator_counts, source_counts = {}, Counter(), Counter()
    with master.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            provenance = row["provenance"]
            label = int(row["label"])
            source_counts[(label, provenance.get("source_dataset", "missing"))] += 1
            if label == 1:
                policies[opaque_id("base_profile", row["base_profile_id"])] = (
                    provenance["generation_policy"]
                )
            if label >= 2:
                generator_counts[(label, provenance.get("generator_family", "missing"))] += 1
    return policies, {
        "source_counts_by_class": [[*key, count] for key, count in sorted(source_counts.items())],
        "generator_counts_by_class": [[*key, count] for key, count in sorted(generator_counts.items())],
        "cross_generator_evaluation_available": len({
            family for (_, family), count in generator_counts.items() if count
        }) > 1,
        "scope": "provenance audit only; never model input",
    }


def evaluate_columns(train, val, columns, class_count, seed=42, stump=False):
    estimator = (
        DecisionTreeClassifier(max_depth=1, random_state=seed)
        if stump else HistGradientBoostingClassifier(**release_bse_parameters(seed))
    )
    estimator.fit(train["raw_tabular"].numpy()[:, columns], train["labels"].numpy())
    probabilities = estimator.predict_proba(val["raw_tabular"].numpy()[:, columns])
    scores = metrics(val["labels"].numpy(), probabilities, class_count)
    return estimator, scores


def subset(values, indices):
    selected = {}
    for key, value in values.items():
        selected[key] = [value[index] for index in indices] if key == "metadata" else value[indices]
    return selected


def run_audit(data_dir, master, output, dataset):
    if output.exists():
        raise FileExistsError(output)
    class_count = 3 if dataset == "Twibot22" else 4
    standardizers = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", 5, standardizers, class_count)
    val = load_split(data_dir, "val", 5, standardizers, class_count)
    groups = assert_group_disjoint(train, val)
    policies, provenance = collect_provenance(master)
    feature_names = [*NUM_NAMES, *STYLE_NAMES]
    dimensions = train["raw_tabular"].shape[1]
    if dimensions != len(feature_names):
        raise ValueError("feature schema does not match audit names")
    probes = {}
    modes = {
        "all_statistics": list(range(dimensions)),
        "numerical_only": list(range(6)),
        "surface_style_only": list(range(7, 15)),
        "action_ratios_only": [15, 16],
        "without_surface_style": [*range(6), 15, 16],
    }
    full_estimator = None
    for mode, columns in modes.items():
        estimator, scores = evaluate_columns(train, val, columns, class_count)
        probes[mode] = {"feature_names": [feature_names[index] for index in columns],
                        "validation": scores}
        if mode == "all_statistics":
            full_estimator = estimator
    stump_scores = {}
    for index, name in enumerate(feature_names):
        _, scores = evaluate_columns(train, val, [index], class_count, stump=True)
        stump_scores[name] = scores
    policy_transfer = {}
    for policy in sorted(set(policies.values())):
        train_indices = [index for index, row in enumerate(train["metadata"])
                         if policies[row["base_profile_id"]] != policy]
        val_indices = [index for index, row in enumerate(val["metadata"])
                       if policies[row["base_profile_id"]] == policy]
        reduced_train, held_val = subset(train, train_indices), subset(val, val_indices)
        counts = assert_group_disjoint(reduced_train, held_val)
        _, scores = evaluate_columns(reduced_train, held_val, list(range(dimensions)), class_count)
        policy_transfer[policy] = {
            "excluded_policy_from_all_training_profile_variants": True,
            "evaluation": "original_validation_groups_with_heldout_traditional_policy",
            "group_counts": counts, "validation": scores,
        }
    labels = train["labels"].numpy()
    permuted = labels.copy()
    rng = np.random.default_rng(20261012)
    group_indices = defaultdict(list)
    for index, row in enumerate(train["metadata"]):
        group_indices[row["base_profile_id"]].append(index)
    for indices in group_indices.values():
        permuted[indices] = rng.permutation(labels[indices])
    placebo = HistGradientBoostingClassifier(**release_bse_parameters(42))
    placebo.fit(train["raw_tabular"].numpy(), permuted)
    placebo_scores = metrics(
        val["labels"].numpy(), placebo.predict_proba(val["raw_tabular"].numpy()), class_count
    )
    # A stress test, not a new training run: neutralize surface-style columns
    # using training medians. Never choose columns based on test outcomes.
    perturbed = val["raw_tabular"].numpy().copy()
    surface = modes["surface_style_only"]
    perturbed[:, surface] = np.median(train["raw_tabular"].numpy()[:, surface], axis=0)
    stress = metrics(val["labels"].numpy(), full_estimator.predict_proba(perturbed), class_count)
    result = {
        "dataset": dataset, "group_counts": groups, "test_paths_opened": False,
        "claim_scope": "development_diagnostics_not_external_generalization",
        "master_sha256": sha256_file(master), "provenance": provenance,
        "feature_probes": probes, "single_feature_stumps": stump_scores,
        "policy_transfer": policy_transfer,
        "within_profile_random_label_negative_control": {
            "seed": 20261012, "validation": placebo_scores,
            "training_labels_permuted_independently_within_each_profile": True,
        },
        "surface_style_neutralization": {
            "intervention": "replace eight surface-style features by training medians",
            "validation": stress,
            "warning": "out-of-distribution stress, not evidence that style is always illegitimate",
        },
    }
    write_new_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_audit(args.data_dir, args.master, args.out, args.dataset)


if __name__ == "__main__":
    main()
