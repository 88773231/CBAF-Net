import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.diagnose_observable_errors import diagnose
from analysis.run_observable_evolution import CONDITIONS, POLICIES


class ObservableErrorDiagnosisTests(unittest.TestCase):
    def fixture(self, study):
        (study / "summary.json").write_text("{}", encoding="utf-8")
        for policy in POLICIES:
            for condition in CONDITIONS:
                folder = study / policy / condition
                folder.mkdir(parents=True)
                rows = []
                for label in range(4):
                    guess = label
                    if condition == CONDITIONS[0]:
                        guess = {0: 3, 1: 0}.get(label, label)
                    if condition == "without_distillation":
                        guess = {0: 2, 2: 3}.get(label, label)
                    probabilities = [.04 / 3] * 4
                    probabilities[guess] = .96
                    rows.append({
                        "sample_id": "shared-" + str(label), "base_profile_id": "shared",
                        "label": label, "probabilities": probabilities,
                    })
                (folder / "inner_selection_predictions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_overlapping_errors_are_not_counted_as_unique_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            self.fixture(study)
            with patch("analysis.diagnose_observable_errors.audit", return_value={
                "audited_cells": 15, "unique_overlapping_selection_profiles": 1,
            }):
                result = diagnose(study)
            self.assertEqual(result["row_events_in_overlapping_folds"], 12)
            self.assertEqual(result["unique_selection_trajectories"], 4)
            self.assertEqual(result["full_error_events"], 6)
            self.assertEqual(result["unique_trajectories_with_any_full_error"], 2)
            self.assertEqual(result["full_human_traditional_error_events"], 3)
            self.assertEqual(result["full_other_pair_error_events"], 3)
            self.assertEqual(result["full_error_confidence_bins"], [
                {"lower_inclusive": .9, "upper": 1., "events": 6},
            ])
            self.assertEqual(result["paired_classification_changes"]["without_distillation"], {
                "both_correct": 3, "both_wrong": 3,
                "full_corrects_reference": 3, "full_introduces_error": 3,
            })
            self.assertFalse(result["independent_evaluation"])
            self.assertFalse(result["submission_allowed"])
            self.assertEqual(result["new_fits_or_model_forwards"], 0)
            self.assertNotIn("shared", json.dumps(result))

    def test_paired_identities_must_remain_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            self.fixture(study)
            path = study / POLICIES[0] / CONDITIONS[1] / "inner_selection_predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["sample_id"] = "different"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with patch("analysis.diagnose_observable_errors.audit", return_value={
                "audited_cells": 15, "unique_overlapping_selection_profiles": 1,
            }):
                with self.assertRaisesRegex(ValueError, "identities differ"):
                    diagnose(study)

    def test_existing_evidence_audit_runs_before_diagnosis(self):
        with patch("analysis.diagnose_observable_errors.audit",
                   side_effect=ValueError("evidence changed")):
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                diagnose(Path("unused-fixture"))


if __name__ == "__main__":
    unittest.main()
