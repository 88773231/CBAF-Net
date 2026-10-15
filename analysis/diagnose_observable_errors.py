"""Aggregate already-exposed known-policy errors; no features or new forwards."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.audit_observable_evolution import audit
from analysis.run_observable_evolution import CONDITIONS, POLICIES
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def diagnose(study):
    audited = audit(study)
    confusion = {condition: np.zeros((4, 4), dtype=np.int64) for condition in CONDITIONS}
    changes = {condition: Counter() for condition in CONDITIONS[1:]}
    confidence_bins = Counter()
    unique_errors, unique_samples, row_events = set(), set(), 0
    for policy in POLICIES:
        predictions = {
            condition: [json.loads(line) for line in
                        (study / policy / condition / "inner_selection_predictions.jsonl").read_text().splitlines()]
            for condition in CONDITIONS
        }
        full = predictions[CONDITIONS[0]]
        row_events += len(full)
        for index, row in enumerate(full):
            identity = row["sample_id"]
            unique_samples.add(identity)
            truth = row["label"]
            probabilities = np.asarray(row["probabilities"])
            predicted = int(probabilities.argmax())
            correct = truth == predicted
            if not correct:
                unique_errors.add(identity)
                confidence_bins[min(int(float(probabilities.max()) * 10), 9)] += 1
            for condition in CONDITIONS:
                reference = predictions[condition][index]
                if (reference["sample_id"], reference["label"]) != (identity, truth):
                    raise ValueError("paired error identities differ")
                guess = int(np.argmax(reference["probabilities"]))
                confusion[condition][truth, guess] += 1
                if condition != CONDITIONS[0]:
                    key = ("both_correct" if correct and guess == truth else
                           "full_corrects_reference" if correct else
                           "full_introduces_error" if guess == truth else "both_wrong")
                    changes[condition][key] += 1
    matrix = confusion[CONDITIONS[0]]
    wrong = int(matrix.sum() - np.trace(matrix))
    human_traditional = int(matrix[0, 1] + matrix[1, 0])
    llm_agent = int(matrix[2, 3] + matrix[3, 2])
    return {
        "scope": "posthoc_previously_exposed_known_policy_inner_selection_predictions_only",
        "summary_sha256": sha256_file(study / "summary.json"),
        "audited_cells": audited["audited_cells"],
        "row_events_in_overlapping_folds": row_events,
        "unique_selection_trajectories": len(unique_samples),
        "unique_selection_profiles": audited["unique_overlapping_selection_profiles"],
        "full_error_events": wrong, "unique_trajectories_with_any_full_error": len(unique_errors),
        "full_human_traditional_error_events": human_traditional,
        "full_llm_agent_error_events": llm_agent,
        "full_other_pair_error_events": wrong - human_traditional - llm_agent,
        "confusion_by_condition": {condition: value.tolist() for condition, value in confusion.items()},
        "paired_classification_changes": {condition: dict(sorted(value.items()))
                                          for condition, value in changes.items()},
        "full_error_confidence_bins": [{"lower_inclusive": index / 10,
                                       "upper": (index + 1) / 10, "events": count}
                                      for index, count in sorted(confidence_bins.items())],
        "independent_evaluation": False, "new_fits_or_model_forwards": 0,
        "raw_identifiers_posts_features_or_prompts_exported": False,
        "classification_confusion_does_not_establish_cross_cue_conflict": True,
        "submission_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.study)
    write_new_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
