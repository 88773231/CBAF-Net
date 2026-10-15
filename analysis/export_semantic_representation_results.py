"""Export audited aggregate evidence without account data or private paths."""

import argparse
import json
from pathlib import Path

from analysis.audit_semantic_representation import audit
from analysis.run_semantic_representation import CONDITIONS, POLICIES
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def export(study, output):
    if output.exists():
        raise FileExistsError("preserve the previous representation export")
    verified = audit(study)
    summary = json.loads((study / "summary.json").read_text())
    protocol = json.loads((study / "protocol.json").read_text())
    small_gain = all(
        item["mean_difference"] > 0 and item["positive_folds"] >= 2
        for item in verified["representation_comparisons"].values()
    )
    public = {
        "schema": "public-semantic-representation-screen-v1",
        "dataset": "Quadbot", "class_count": 4, "training_seed": 42,
        "conditions": list(CONDITIONS), "folds": list(POLICIES),
        "metric_population": "known_policy_inner_selection_only",
        "folds_overlap_not_independent_replications": True,
        "new_fits": verified["audited_cells"], "checkpoint_replays": 12,
        "verified_prediction_metrics_and_hashes": True,
        "private_summary_sha256": verified["summary_sha256"],
        "private_protocol_sha256": sha256_file(study / "protocol.json"),
        "source_hashes": protocol["source_hashes"],
        "encoder": protocol["encoder"],
        "feature_dimensions": protocol["feature_dimensions"],
        "configuration": protocol["configuration"],
        "records": [{
            "policy": row["policy"], "condition": row["condition"],
            "macro_f1": row["inner_selection"]["all_profiles"]["macro_f1"],
            "best_epoch": row["fit"]["best_epoch"],
            "epochs_run": len(row["fit"]["history"]),
            "trainable_parameter_count": row["fit"]["training_parameter_count"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "private_predictions_sha256": row["predictions_sha256"],
            "maximum_replay_probability_difference": row["maximum_replay_probability_difference"],
        } for row in summary["records"]],
        "mean_macro_f1_by_condition": verified["mean_macro_f1_by_condition"],
        "representation_comparisons": verified["representation_comparisons"],
        "original_025_point_gate_passed": verified["representation_continuation_gate_passed"],
        "prospectively_relaxed_small_gain_rule_passed": small_gain,
        "permission_for_reproducibility_check_only": small_gain,
        "branch_closed_if_small_gain_rule_fails": not small_gain,
        "original_validation_test_or_excluded_policy_scored": False,
        "new_account_generation": False, "gpu_used": False,
        "algorithmic_novelty_established": False, "independent_confirmation": False,
        "module_ablation_or_architecture_superiority_established": False,
        "test_performance_claimed": False, "submission_allowed": False,
        "raw_text_embeddings_or_account_identifiers_included": False,
    }
    write_new_json(output, public)
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = export(args.study, args.output)
    print(json.dumps({
        "exported_cells": len(public["records"]),
        "public_result_sha256": sha256_file(args.output),
        "small_gain_rule_passed": public["prospectively_relaxed_small_gain_rule_passed"],
        "submission_allowed": False,
    }))


if __name__ == "__main__":
    main()
