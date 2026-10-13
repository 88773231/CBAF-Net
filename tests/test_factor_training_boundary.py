import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.run_factor_training_screen import (
    POLICIES, policies_from_partitions, screen_inner_partition, training_input_files,
)


class FactorTrainingBoundaryTests(unittest.TestCase):
    def test_inner_selection_covers_all_and_only_fitting_policies(self):
        rows = [
            {"base_profile_id": f"{policy}_{profile}", "label": label}
            for policy in POLICIES for profile in range(6) for label in range(3)
        ]
        policies = {
            row["base_profile_id"]: next(name for name in POLICIES if row["base_profile_id"].startswith(name))
            for row in rows
        }
        for heldout in POLICIES:
            fitting, selection = screen_inner_partition(rows, policies, heldout, 42)
            for part in (fitting, selection):
                self.assertEqual({policies[rows[index]["base_profile_id"]] for index in part},
                                 set(POLICIES) - {heldout})
            self.assertEqual(len(selection), 6)
            self.assertFalse(set(fitting) & set(selection))
            self.assertEqual((fitting, selection), screen_inner_partition(rows, policies, heldout, 42))

    def rows(self):
        return [
            {"base_profile_id": profile, "label": label}
            for profile in ("a", "b", "c") for label in range(3)
        ]

    def receipts(self):
        return {
            "first": {"fitting_indices": [3, 4, 5], "selection_indices": [6, 7, 8]},
            "second": {"fitting_indices": [0, 1, 2], "selection_indices": [6, 7, 8]},
            "third": {"fitting_indices": [0, 1, 2], "selection_indices": [3, 4, 5]},
        }

    def test_file_inventory_never_constructs_val_or_test_inputs(self):
        with patch("pathlib.Path.is_file", side_effect=AssertionError("file access")):
            paths = training_input_files(Path("/features"), Path("/timing"))
        self.assertTrue(all("train" in path.name for path in paths))
        self.assertFalse(any("val_" in path.name or "test_" in path.name for path in paths))

    def test_policy_recovery_only_uses_training_receipts(self):
        self.assertEqual(
            policies_from_partitions(self.rows(), self.receipts()),
            {"a": "first", "b": "second", "c": "third"},
        )

    def test_partial_profile_and_duplicate_assignment_rejected(self):
        receipts = self.receipts()
        receipts["first"]["selection_indices"].append(0)
        with self.assertRaisesRegex(ValueError, "divides"):
            policies_from_partitions(self.rows(), receipts)
        receipts = self.receipts()
        receipts["third"] = receipts["first"]
        with self.assertRaisesRegex(ValueError, "multiple"):
            policies_from_partitions(self.rows(), receipts)

    def test_invalid_prior_indices_rejected(self):
        for indices in ([6, 7, 8, 9], [6, 7, 8, 8], [3, 6, 7, 8]):
            receipts = self.receipts()
            receipts["first"]["selection_indices"] = indices
            with self.assertRaisesRegex(ValueError, "invalid"):
                policies_from_partitions(self.rows(), receipts)

    def test_incomplete_policy_coverage_rejected(self):
        receipts = self.receipts()
        del receipts["third"]
        with self.assertRaisesRegex(ValueError, "cover"):
            policies_from_partitions(self.rows(), receipts)


if __name__ == "__main__":
    unittest.main()
