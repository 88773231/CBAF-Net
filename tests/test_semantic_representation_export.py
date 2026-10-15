import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.export_semantic_representation_results import export
from analysis.run_semantic_representation import CONDITIONS, POLICIES


class SemanticRepresentationExportTests(unittest.TestCase):
    def export_fixture(self, folder, gain):
        records = [{
            "policy": policy, "condition": condition,
            "inner_selection": {"all_profiles": {"macro_f1": .9}},
            "fit": {
                "best_epoch": 1, "history": [1], "training_parameter_count": 100,
                "private_machine_path": "SECRET_PATH",
            },
            "checkpoint_sha256": "a" * 64, "predictions_sha256": "b" * 64,
            "maximum_replay_probability_difference": 0,
            "private_account_text": "SECRET_TEXT",
        } for policy in POLICIES for condition in CONDITIONS]
        protocol = {
            "source_hashes": {"file.py": "c" * 64},
            "encoder": {"frozen": True},
            "feature_dimensions": {"text": 384}, "configuration": {"seed": 42},
            "private_snapshot_path": "SECRET_PATH",
        }
        (folder / "summary.json").write_text(json.dumps({"records": records}))
        (folder / "protocol.json").write_text(json.dumps(protocol))
        comparisons = {name: {
            "mean_difference": gain, "positive_folds": 2,
            "representation_gate_passed": gain >= .0025,
        } for name in ("mlp", "backbone")}
        verified = {
            "audited_cells": 12, "summary_sha256": "d" * 64,
            "mean_macro_f1_by_condition": {name: .9 for name in CONDITIONS},
            "representation_comparisons": comparisons,
            "representation_continuation_gate_passed": gain >= .0025,
        }
        with patch("analysis.export_semantic_representation_results.audit",
                   return_value=verified) as auditor:
            result = export(folder, folder / "public" / "summary.json")
        auditor.assert_called_once_with(folder)
        self.assertEqual(len(result["records"]), 12)
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertFalse(result["submission_allowed"])
        self.assertFalse(result["test_performance_claimed"])
        return result

    def test_small_gain_does_not_relabel_original_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.export_fixture(Path(temporary), .0001)
        self.assertTrue(result["prospectively_relaxed_small_gain_rule_passed"])
        self.assertFalse(result["original_025_point_gate_passed"])
        self.assertFalse(result["algorithmic_novelty_established"])

    def test_zero_gain_closes_branch_and_omits_private_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.export_fixture(Path(temporary), 0)
        self.assertFalse(result["permission_for_reproducibility_check_only"])
        self.assertTrue(result["branch_closed_if_small_gain_rule_fails"])

    def test_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text("preserved")
            with self.assertRaises(FileExistsError):
                export(Path(temporary), path)
            self.assertEqual(path.read_text(), "preserved")


if __name__ == "__main__":
    unittest.main()
