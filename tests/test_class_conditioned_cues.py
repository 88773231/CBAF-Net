import copy
import unittest

import torch

from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS
from analysis.run_class_conditioned_cues import CONFIGURATION
from src.class_conditioned_cues import (
    CONDITIONS, ClassConditionedCueModel, objective, paired_profile_margin, shared_digest, train,
)
from src.observable_evolution import CUES, INPUTS


class ClassConditionedCueTests(unittest.TestCase):
    def inputs(self, count=8):
        return [torch.randn((count, STEPS, FEATURE_DIMENSIONS[name]) if name in CUES
                            else (count, FEATURE_DIMENSIONS[name])) for name in INPUTS]

    def test_probability_routes_are_class_conditional_and_scalar_control_is_not(self):
        values = self.inputs()
        full = ClassConditionedCueModel(dropout=0).eval()
        result = full(*values)
        self.assertEqual(result["logits"].shape, (8, 4))
        self.assertTrue(torch.allclose(result["routing_weights"].sum(1), torch.ones(8, 4)))
        self.assertFalse(torch.allclose(result["routing_weights"][:, :, 0],
                                        result["routing_weights"][:, :, 1]))
        scalar = copy.deepcopy(full)
        scalar.condition = "scalar_routing_control"
        weights = scalar(*values)["routing_weights"]
        self.assertTrue(torch.allclose(weights[:, :, 0], weights[:, :, 3]))

    def test_removal_is_exact_context_and_shared_initialization_is_preserved(self):
        values = self.inputs()
        digests = []
        for condition in CONDITIONS:
            torch.manual_seed(42)
            model = ClassConditionedCueModel(dropout=0, condition=condition).eval()
            digests.append(shared_digest(model))
            if condition == "without_class_routing":
                self.assertIsNone(model.routing)
                self.assertIsNone(model(*values)["routing_weights"])
                self.assertTrue(torch.equal(model(*values)["logits"], model.backbone(*values)["logits"]))
        self.assertEqual(len(set(digests)), 1)

    def test_pair_margin_is_group_order_and_logit_shift_invariant(self):
        logits = torch.eye(4).repeat(2, 1) * 3
        labels = torch.arange(4).repeat(2)
        profiles = ["first"] * 4 + ["second"] * 4
        expected = paired_profile_margin(logits, labels, profiles, 1)
        order = torch.tensor([5, 0, 7, 3, 2, 4, 1, 6])
        actual = paired_profile_margin(logits[order], labels[order],
                                       [profiles[index] for index in order.tolist()], 1)
        self.assertTrue(torch.allclose(expected, actual))
        self.assertTrue(torch.allclose(expected, paired_profile_margin(
            logits + torch.arange(8)[:, None], labels, profiles, 1)))
        self.assertLess(expected, paired_profile_margin(torch.zeros_like(logits), labels, profiles, 1))
        with self.assertRaisesRegex(ValueError, "class-complete"):
            paired_profile_margin(logits[:3], labels[:3], profiles[:3], 1)

    def test_actual_objective_removal_and_gradient_reach_retained_components(self):
        values = self.inputs()
        labels, profiles = torch.arange(4).repeat(2), ["a"] * 4 + ["b"] * 4
        config = {"cue_supervision_weight": .1, "pair_margin_weight": .1, "pair_margin": 1}
        model = ClassConditionedCueModel(dropout=0)
        output = model(*values)
        full = objective(output, labels, profiles, CONDITIONS[0], config)
        removed = objective(output, labels, profiles, "without_pair_margin", config)
        self.assertTrue(torch.allclose(full - removed, .1 * paired_profile_margin(
            output["logits"], labels, profiles, 1)))
        full.backward()
        self.assertGreater(model.routing[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(model.projections["text"][0].weight.grad.abs().sum(), 0)
        self.assertGreater(model.backbone.encoders["text"][0].weight.grad.abs().sum(), 0)
        self.assertGreater(model.residual_log_scale.grad.abs(), 0)

    def test_single_account_inference_is_batch_independent(self):
        values = self.inputs()
        model = ClassConditionedCueModel(dropout=0).eval()
        full = model(*values)["logits"]
        single = model(*(value[:1] for value in values))["logits"]
        self.assertTrue(torch.allclose(full[:1], single, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "schema"):
            model(*(values[:-1] + [values[-1][:, :2]]))

    def test_small_fit_updates_retained_parameters_and_rejects_profile_overlap(self):
        populations, policies = [], {}
        for prefix in ("fit", "selection"):
            rows = []
            for policy in ("known-a", "known-b"):
                profile = prefix + policy
                policies[profile] = policy
                rows.extend({"base_profile_id": profile, "sample_id": profile + str(label),
                             "label": label} for label in range(4))
            populations.append({
                "metadata": rows, "labels": torch.arange(4).repeat(2),
                **dict(zip(INPUTS, self.inputs())),
            })
        config = {**CONFIGURATION, "max_epochs": 2, "patience": 1}
        model, fitted = train(*populations, policies, CONDITIONS[0], config)
        self.assertIn(fitted["best_epoch"], (1, 2))
        self.assertFalse(fitted["submission_allowed"])
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        invalid = copy.deepcopy(populations[1])
        invalid["metadata"][0]["base_profile_id"] = populations[0]["metadata"][0]["base_profile_id"]
        with self.assertRaises(ValueError):
            train(populations[0], invalid, policies, CONDITIONS[0], config)


if __name__ == "__main__":
    unittest.main()
