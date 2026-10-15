import copy
import unittest

import torch
from threadpoolctl import threadpool_limits

from src.compact_correspondence import make_model
from src.correspondence_dependence import attention_summary, diagnose, fixed_history
from tests import test_trajectory_correspondence as fixture


class CorrespondenceDependenceTests(unittest.TestCase):
    def model(self):
        fitting, _, _, config = fixture.TrajectoryCorrespondenceTests().populations()
        model = make_model(fitting, config, "compact_correspondence")
        model.eval()
        return model, fitting

    def test_fixed_history_and_entropy(self):
        uniform = fixed_history(5, torch.device("cpu"), torch.float32)
        summary = attention_summary(uniform[None].expand(3, -1, -1))
        self.assertEqual(summary["future_mass"], 0)
        self.assertEqual(summary["mean_absolute_uniform_difference"], 0)
        self.assertAlmostEqual(sum(summary["lag_mass_mean_over_steps"]), 1, places=6)
        for value in summary["normalized_entropy_by_nontrivial_step"]:
            self.assertAlmostEqual(value, 1, places=6)
        summary = attention_summary(torch.eye(5)[None])
        self.assertEqual(summary["normalized_entropy_by_nontrivial_step"], [0] * 4)
        with self.assertRaises(ValueError):
            attention_summary(torch.zeros(2, 5, 5))
        with self.assertRaises(ValueError):
            fixed_history(1, torch.device("cpu"), torch.float32)

    def test_probes_read_only_and_batch_independent(self):
        torch.set_num_threads(2)
        model, fitting = self.model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with threadpool_limits(limits=2):
            result = diagnose(model, fitting, 5)
            single = diagnose(model, fitting, 1)
        self.assertFalse(result["selection_or_challenge_or_test_forwards"])
        self.assertFalse(result["submission_allowed"])
        self.assertLessEqual(result["maximum_baseline_replay_error"], 1e-6)
        self.assertLessEqual(result["maximum_row_isolation_error"], 1e-5)
        self.assertLessEqual(result["maximum_linear_decomposition_error"], 1e-5)
        self.assertEqual(result["class_counts"], torch.bincount(fitting["labels"], minlength=4).tolist())
        for probe in result["outputs"]:
            self.assertAlmostEqual(result["outputs"][probe]["fitting_cross_entropy"],
                                   single["outputs"][probe]["fitting_cross_entropy"], places=6)
            self.assertEqual(result["outputs"][probe]["prediction_changes_from_baseline"],
                             single["outputs"][probe]["prediction_changes_from_baseline"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_uniform_queries_remove_adaptive_dependence(self):
        model, fitting = self.model()
        with torch.no_grad():
            for parameter in model.correspondence.parameters():
                parameter.zero_()
        result = diagnose(model, fitting)
        self.assertEqual(result["outputs"]["uniform_history"]["maximum_probability_change"], 0)
        self.assertEqual(result["adaptive_alignment_rms_relative_to_uniform"],
                         {"text_to_action": 0, "action_to_text": 0})

    def test_nonfitting_metadata_not_model_input_and_invalid_mode_rejected(self):
        model, fitting = self.model()
        copied = copy.deepcopy(fitting)
        copied["metadata"] = [{"label": 999, "base_profile_id": "not-an-input"}] * len(fitting["labels"])
        self.assertEqual(diagnose(model, fitting), diagnose(model, copied))
        model.train()
        with self.assertRaises(ValueError):
            diagnose(model, fitting)
        model.eval()
        with self.assertRaises(ValueError):
            diagnose(model, fitting, 0)
        copied["labels"][0] = 4
        with self.assertRaises(ValueError):
            diagnose(model, copied)


if __name__ == "__main__":
    unittest.main()
