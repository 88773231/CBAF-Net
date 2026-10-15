"""Export aggregate revision evidence without account identifiers or machine paths."""

import argparse
import json
from pathlib import Path

from analysis.audit_compact_correspondence_stability import audit
from analysis.run_compact_correspondence_stability import CONDITIONS, SEEDS, source_hashes
from analysis.run_factor_training_screen import POLICIES
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def export(archive, summary, receipt, output):
    if output.exists():
        raise FileExistsError("public revision result must not be overwritten")
    verified = audit(archive, summary, receipt)
    original = json.loads(summary.read_text())
    public = {
        "schema": "public-compact-correspondence-development-results-v1",
        "dataset": "Quadbot", "class_count": 4,
        "classes": ["Human", "Traditional Bot", "LLM Bot", "Full-stack Agent"],
        "training_seeds": list(SEEDS), "partition_seed": 42,
        "folds": list(POLICIES), "conditions": list(CONDITIONS),
        "metric_population": "known_policy_inner_selection_only",
        "distinct_selection_profiles": verified["selection_profiles_across_folds"],
        "selection_profile_memberships_per_profile": 2,
        "new_fits": 30, "reused_checkpoints": 6, "checkpoint_replays": 36,
        "source_hashes": source_hashes(),
        "evidence_archive_sha256": sha256_file(archive),
        "private_summary_sha256": sha256_file(summary),
        "verified_all_predictions_and_metrics": True,
        "records": [{
            "training_seed": row["seed"], "partition_seed": row["partition_seed"],
            "excluded_policy": row["policy"], "condition": row["condition"],
            "inner_selection": row["inner_selection"],
            "best_epoch": row["fit"]["best_epoch"],
            "epochs_run": len(row["fit"]["history"]), "new_fit": row["fit"]["new_fit"],
            "trainable_parameter_count": row["fit"].get(
                "parameter_count", row["fit"].get("training_parameter_count"),
            ),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "private_predictions_sha256": row["predictions_sha256"],
        } for row in original["records"]],
        "decision": verified["decision"],
        "challenge_or_original_validation_test_scoring": False,
        "independent_confirmation": False, "algorithmic_novelty_established": False,
        "test_performance_claimed": False, "submission_allowed": False,
        "publication_scope": "aggregate_development_evidence_not_final_method_release",
        "raw_data_or_original_identifiers_included": False,
    }
    write_new_json(output, public)
    return public


def main():
    parser = argparse.ArgumentParser()
    for name in ("archive", "summary", "receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = export(args.archive, args.summary, args.receipt, args.output)
    print(json.dumps({"exported_cells": len(result["records"]),
                      "public_result_sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
