import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.evidence_model import TemporalEvidenceModel
from src.evidence_repair import assert_group_disjoint, input_files, metrics, model_inputs, write_new_json


class EvidenceRepairTests(unittest.TestCase):
    def test_candidate_shapes_probabilities_and_trainable_modules(self):
        torch.manual_seed(42)
        model = TemporalEvidenceModel(8, 8, 6, 4, 3, steps=2, hidden_dim=8, dropout=0)
        inputs = (torch.randn(5, 8), torch.randn(5, 2, 8),
                  torch.randn(5, 2, 6), torch.randn(5, 4))
        outputs = model(*inputs)
        self.assertEqual(outputs["logits"].shape, (5, 3))
        self.assertEqual(outputs["gate_weights"].shape, (5, 3, 2))
        torch.testing.assert_close(outputs["gate_weights"].sum(2), torch.ones(5, 3))
        outputs["logits"].square().mean().backward()
        self.assertGreater(float(model.gate[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.cross_view[0].weight.grad.abs().sum()), 0)
        self.assertIsNot(model.text_encoders[0], model.text_encoders[1])

    def test_ablation_constructs_absent_modules_and_uniform_weights(self):
        model = TemporalEvidenceModel(8, 8, 6, 4, 4, steps=2, hidden_dim=8,
                                      dropout=0, interaction=False, gate=False)
        self.assertIsNone(model.gate)
        self.assertEqual(model.cross_view[0].in_features, 16)
        outputs = model(torch.randn(3, 8), torch.randn(3, 2, 8),
                        torch.randn(3, 2, 6), torch.randn(3, 4))
        torch.testing.assert_close(
            outputs["gate_weights"], torch.full((3, 4, 2), 1 / 2)
        )

    def test_file_enumeration_forbids_test_before_opening_any_path(self):
        with patch("pathlib.Path.is_file", side_effect=AssertionError("file opened")):
            with self.assertRaisesRegex(ValueError, "train and val"):
                input_files(Path("/not/a/dataset"), "test", 5)
        paths = input_files(Path("/dataset"), "val", 5)
        self.assertFalse(any("test" in str(path) for path in paths))

    def test_group_overlap_rejected(self):
        values = {"metadata": [{"base_profile_id": "p", "label": 0},
                               {"base_profile_id": "p", "label": 1}],
                  "labels": torch.tensor([0, 1])}
        with self.assertRaisesRegex(ValueError, "crosses"):
            assert_group_disjoint(values, values)

    def test_metrics_require_valid_multiclass_probabilities(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            metrics(np.array([0, 1]), np.array([[0.1, 0.1], [0.2, 0.2]]), 2)
        result = metrics(np.array([0, 1]), np.array([[0.8, 0.2], [0.1, 0.9]]), 2)
        self.assertEqual(result["macro_f1"], 1)

    def test_selection_artifact_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selection.json"
            write_new_json(path, {"test_state": "NOT_OPENED"})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"test_state": "changed"})

    def test_timing_and_style_removals_are_distinct(self):
        values = {
            "description": torch.randn(4, 8), "text": torch.randn(4, 2, 8),
            "action": torch.randn(4, 2, 15), "tabular": torch.randn(4, 24),
            "numerical": torch.randn(4, 13),
        }
        _, _, no_time_action, no_time_stats = model_inputs(values, "without_timing")
        _, _, no_style_action, no_style_stats = model_inputs(values, "without_style")
        self.assertEqual(no_time_action.shape[-1], 8)
        self.assertEqual(no_time_stats.shape[-1], 17)
        self.assertEqual(no_style_action.shape[-1], 15)
        self.assertEqual(no_style_stats.shape[-1], 13)


if __name__ == "__main__":
    unittest.main()
