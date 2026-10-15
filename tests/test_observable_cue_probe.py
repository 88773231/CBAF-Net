import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.probe_observable_cues import selection_values, summarize
from analysis.run_observable_evolution import METADATA_SHA256, SOURCE_SHA256
from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS
from src.observable_evolution import CUES, INPUTS


class ObservableCueProbeTests(unittest.TestCase):
    def probabilities(self, guesses):
        return np.eye(4)[guesses] * .96 + .01

    def test_disagreement_and_oracle_are_separate_from_fused_accuracy(self):
        labels = np.arange(4)
        fused = self.probabilities([3, 1, 2, 3])
        cues = self.probabilities(np.array([[0, 3, 3, 3], [1, 1, 1, 1],
                                           [2, 2, 2, 2], [3, 3, 3, 3]]))
        report = summarize(labels, fused, cues)
        self.assertEqual(report["fused_errors"], 1)
        self.assertEqual(report["fused_errors_with_at_least_one_correct_cue"], 1)
        self.assertEqual(report["fused_errors_with_all_cues_wrong"], 0)
        self.assertEqual(report["fused_errors_with_unanimous_cues"], 0)
        self.assertAlmostEqual(report["pairwise_disagreement_sum_on_fused_errors"], .5)
        self.assertEqual(report["pairwise_disagreement_sum_on_fused_correct"], 0)
        self.assertGreater(report["jsd_sum_on_fused_errors"], 0)
        self.assertTrue(report["any_correct_cue_is_label_using_oracle_not_an_achievable_score"])

    def test_unanimous_wrong_cues_do_not_establish_recoverability(self):
        report = summarize(np.arange(4), self.probabilities([3, 1, 2, 3]),
                           self.probabilities(np.array([[3] * 4, [1] * 4, [2] * 4, [3] * 4])))
        self.assertEqual(report["fused_errors_with_unanimous_cues"], 1)
        self.assertEqual(report["fused_errors_with_all_cues_wrong"], 1)
        self.assertEqual(report["fused_errors_with_at_least_one_correct_cue"], 0)
        self.assertEqual(report["pairwise_disagreement_sum_on_fused_errors"], 0)

    def test_invalid_or_empty_probabilities_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "four-class"):
            summarize(np.array([], dtype=int), np.empty((0, 4)), np.empty((0, 4, 4)))
        cues = self.probabilities(np.array([[0] * 4, [1] * 4, [2] * 4, [3] * 4]))
        with self.assertRaisesRegex(ValueError, "distributions"):
            summarize(np.arange(4), np.ones((4, 4)), cues)
        cues[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "distributions"):
            summarize(np.arange(4), self.probabilities(np.arange(4)), cues)

    def test_only_selection_observed_and_saved_constant_mask_is_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, metadata = root / "train.jsonl", root / "metadata.jsonl"
            rows = [{"sample_id": str(index), "base_profile_id": str(index // 4),
                     "label": index % 4} for index in range(8)]
            text = "".join(json.dumps(row) + "\n" for row in rows)
            source.write_text(text, encoding="utf-8")
            metadata.write_text(text, encoding="utf-8")
            features = {
                name: np.full((STEPS, FEATURE_DIMENSIONS[name]) if name in CUES else
                              (FEATURE_DIMENSIONS[name],), 2, dtype=np.float32) for name in INPUTS
            }
            transforms = {
                name: {"mean": [1] * FEATURE_DIMENSIONS[name], "std": [1] * FEATURE_DIMENSIONS[name],
                       "observed_std": [1] * FEATURE_DIMENSIONS[name],
                       "variable_mask": [True] + [False] * (FEATURE_DIMENSIONS[name] - 1)}
                for name in ("numeric", "behavior", "style", "timing")
            }
            identities = {row["sample_id"]: (row["base_profile_id"], row["label"]) for row in rows}
            with patch("analysis.probe_observable_cues.sha256_file",
                       side_effect=lambda path: SOURCE_SHA256 if path == source else METADATA_SHA256):
                with patch("analysis.probe_observable_cues.training_identities", return_value=identities):
                    with patch("analysis.probe_observable_cues.observe", return_value=features) as observer:
                        values = selection_values(source, metadata, {
                            "selection_indices": [0, 1, 2, 3], "fitting_indices": [4, 5, 6, 7],
                        }, transforms)
            self.assertEqual(observer.call_count, 4)
            self.assertEqual([call.args[0]["sample_id"] for call in observer.call_args_list],
                             ["0", "1", "2", "3"])
            self.assertTrue((values["numeric"][:, 0] == 1).all())
            self.assertTrue((values["numeric"][:, 1:] == 0).all())


if __name__ == "__main__":
    unittest.main()
