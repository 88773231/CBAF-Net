import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis.audit_control_predictions import (
    assert_same_rows, disagreement, prediction_summary, selected_history,
    shared_view_comparison, validate_predictions, verify_export, verify_normalization,
)
from src.provenance import sha256_file


class ControlPredictionAuditTests(unittest.TestCase):
    def rows(self, classes=3, predictions=None):
        predictions = list(range(classes)) if predictions is None else predictions
        result = []
        for label, predicted in enumerate(predictions):
            probability = [0.1 / (classes - 1)] * classes
            probability[predicted] = 0.9
            result.append({
                "sample_id": f"s{label}", "base_profile_id": "p",
                "label": label, "prediction": predicted, "probabilities": probability,
            })
        return result

    def test_confusion_and_mutual_errors_and_disagreement(self):
        candidate, control = self.rows(predictions=[1, 0, 2]), self.rows()
        score = prediction_summary(candidate, 3)
        self.assertEqual(score["error_count"], 2)
        self.assertEqual(score["human_traditional_fraction_of_errors"], 1)
        self.assertAlmostEqual(score["macro_f1"], 1 / 3)
        self.assertEqual(score["confusion_true_rows_predicted_columns"],
                         [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        cells = disagreement(candidate, control, 3)
        self.assertEqual(cells["all"], {
            "rows": 3, "both_correct": 1, "candidate_only_correct": 0,
            "control_only_correct": 2, "both_wrong": 0, "prediction_disagreements": 2,
        })
        self.assertEqual(cells["by_true_class"]["Traditional Bot"]["control_only_correct"], 1)

    def test_no_errors_has_no_conditional_wrong_confidence(self):
        score = prediction_summary(self.rows(), 3)
        self.assertIsNone(score["human_traditional_fraction_of_errors"])
        self.assertIsNone(score["mean_confidence_wrong"])

    def test_invalid_predictions_rejected(self):
        for change in ("argmax", "nonfinite", "duplicate", "incomplete", "out_of_range"):
            rows = self.rows()
            if change == "argmax":
                rows[0]["prediction"] = 1
            elif change == "nonfinite":
                rows[0]["probabilities"][0] = float("nan")
            elif change == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif change == "incomplete":
                rows.pop()
            else:
                rows[0]["label"] = 3
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_predictions(rows, 3)

    def test_pairing_is_order_and_profile_sensitive(self):
        rows = self.rows()
        for other in (rows[::-1], [{**row, "base_profile_id": "other"} for row in rows]):
            with self.assertRaisesRegex(ValueError, "identities"):
                assert_same_rows(rows, other)

    def record(self):
        def scores(f1):
            return {
                "all_profiles": {"macro_f1": f1, "mcc": f1},
                "by_fitting_policy": {
                    "cyclic_reposter": {"macro_f1": f1},
                    "uniform_scheduler": {"macro_f1": f1},
                },
                "worst_policy_macro_f1": f1,
            }

        return {
            "heldout_policy": "burst_scheduler",
            "fit": {"best_epoch": 2, "history": [
                {"epoch": 1, "inner_selection": scores(.8)},
                {"epoch": 2, "inner_selection": scores(.9)},
                {"epoch": 3, "inner_selection": scores(.9)},
            ]},
            "challenge": {"macro_f1": .6},
        }

    def test_only_frozen_inner_selection_selects_checkpoint_and_ties_are_earliest(self):
        score = selected_history(self.record())
        self.assertEqual(score["selected_epoch"], 2)
        self.assertAlmostEqual(score["selection_minus_excluded_policy_macro_f1"], .3)
        self.assertTrue(score["not_a_same_population_generalization_estimate"])
        changed = self.record()
        changed["fit"]["best_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "epoch"):
            selected_history(changed)
        changed = self.record()
        changed["fit"]["history"][0]["inner_selection"]["by_fitting_policy"] = {
            "burst_scheduler": {"macro_f1": .8},
            "uniform_scheduler": {"macro_f1": .8},
        }
        with self.assertRaisesRegex(ValueError, "excluded policy"):
            selected_history(changed)

    def test_shared_views_keep_agent_predictions_wrong_instead_of_remapping(self):
        three = self.rows()
        four = self.rows(4, [0, 3, 2, 3])
        result = shared_view_comparison(three, four)
        self.assertEqual(result["shared_rows"], 3)
        self.assertEqual(result["three_class_only_correct"], 1)
        self.assertEqual(result["four_class_predicts_agent"], 1)
        self.assertTrue(result["different_training_tasks_not_a_causal_head_intervention"])
        four[0]["sample_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "nested-view"):
            shared_view_comparison(three, four)

    def test_export_and_normalization_digest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prediction = root / "member.json"
            prediction.write_text("{}")
            manifest = {"member.json": sha256_file(prediction)}
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "archive_member_hashes": manifest, "private_archive_not_for_public_release": True,
            }))
            self.assertEqual(verify_export(root, receipt), manifest)
            prediction.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "digest"):
                verify_export(root, receipt)
            normalization = (
                root / "botsscl_policy_control_20261014" / "Twibot22" /
                "seed42" / "burst_scheduler" / "normalization.json"
            )
            normalization.parent.mkdir(parents=True)
            normalization.write_text("{}")
            name = (
                "flat_policy_screen_20261013/Twibot22/main/"
                "init42_fold42/burst_scheduler/normalization.json"
            )
            export = {normalization.relative_to(root).as_posix(): sha256_file(normalization)}
            prior = {name: sha256_file(normalization)}
            verify_normalization(root, export, "Twibot22", 42, "burst_scheduler", prior)
            normalization.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "normalization"):
                verify_normalization(root, export, "Twibot22", 42, "burst_scheduler", prior)

    def test_unsafe_export_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            root.mkdir()
            outside = root.parent / "outside.json"
            outside.write_text("{}")
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "archive_member_hashes": {"../outside.json": sha256_file(outside)},
                "private_archive_not_for_public_release": True,
            }))
            with self.assertRaisesRegex(ValueError, "path"):
                verify_export(root, receipt)


if __name__ == "__main__":
    unittest.main()
