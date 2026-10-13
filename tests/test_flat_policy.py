import unittest

import torch
from torch import nn

from src.flat_policy import METHODS, sam_step, train_flat
from src.paired_policy import make_paired_model, predict_paired
from tests import test_paired_policy as fixture_module


class FlatPolicyTests(unittest.TestCase):
    def quadratic(self, weight=1.0):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(weight)
        return model, torch.optim.SGD(model.parameters(), lr=.1)

    def test_sam_restores_before_optimizer_update_and_matches_scalar_formula(self):
        model, optimizer = self.quadratic(.5)
        record = sam_step(model, optimizer, lambda: model.weight.square().sum() / 2, .05)
        torch.testing.assert_close(model.weight, torch.tensor([[.445]]))
        self.assertAlmostEqual(record["base_loss"], .125, places=6)
        self.assertAlmostEqual(record["perturbed_loss"], .15125, places=6)
        self.assertAlmostEqual(record["base_gradient_norm"], .5, places=6)
        self.assertEqual(record["optimizer_updates"], 1)
        self.assertEqual(record["gradient_evaluations"], 2)

    def test_gradient_clipping_applies_to_second_gradient_after_restoration(self):
        model, optimizer = self.quadratic(2)
        record = sam_step(model, optimizer, lambda: model.weight.square().sum() / 2, .05)
        torch.testing.assert_close(model.weight, torch.tensor([[1.9]]))
        self.assertAlmostEqual(record["base_gradient_norm"], 2, places=6)
        self.assertAlmostEqual(record["perturbed_loss"], 2.10125, places=5)

    def test_second_closure_failure_restores_weights_without_optimizer_step(self):
        model, optimizer = self.quadratic()
        calls = []

        def closure():
            calls.append(float(model.weight.detach()))
            if len(calls) == 2:
                raise RuntimeError("probe failure")
            return model.weight.square().sum()

        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            sam_step(model, optimizer, closure, .05)
        torch.testing.assert_close(model.weight, torch.ones(1, 1), atol=0, rtol=0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertAlmostEqual(calls[1], 1.05, places=6)

    def test_dropout_realization_and_rng_progress_are_reused_once(self):
        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(3, 3, bias=False), nn.Dropout(.4))
        with torch.no_grad():
            model[0].weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        masks, states = [], []

        def closure():
            values = model(torch.ones(20, 3))
            masks.append(values == 0)
            states.append(torch.get_rng_state().clone())
            return values.square().mean()

        sam_step(model, optimizer, closure, .05)
        self.assertTrue(torch.equal(masks[0], masks[1]))
        self.assertTrue(torch.equal(states[0], states[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), states[0]))

    def test_zero_gradient_remains_finite(self):
        model, optimizer = self.quadratic(0)
        record = sam_step(model, optimizer, lambda: model.weight.square().sum(), .05)
        self.assertEqual(record["base_gradient_norm"], 0)
        self.assertEqual(record["loss_increase"], 0)
        torch.testing.assert_close(model.weight, torch.zeros(1, 1), atol=0, rtol=0)

    def test_invalid_radius_is_rejected_before_loss_evaluation(self):
        model, optimizer = self.quadratic()
        for radius in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sam_step(model, optimizer, lambda: self.fail("closure must not be evaluated"), radius)

    def test_nonfinite_second_loss_restores_parameters(self):
        model, optimizer = self.quadratic()
        calls = []

        def closure():
            calls.append(True)
            return model.weight.square().sum() if len(calls) == 1 else model.weight.sum() * float("nan")

        with self.assertRaisesRegex(ValueError, "perturbed closure"):
            sam_step(model, optimizer, closure, .05)
        torch.testing.assert_close(model.weight, torch.ones(1, 1), atol=0, rtol=0)

    def fixture(self, classes):
        source = fixture_module.PairedPolicyTests()
        fitting, selection = source.values("f", classes), source.values("s", classes)
        policies = {prefix + str(index): "policy" + str(index % 2)
                    for prefix in ("f", "s") for index in range(4)}
        regularization = {"contrast_weight": .1, "temperature": .2, "sam_radius": .05}
        return fitting, selection, policies, source.settings(), regularization

    def test_all_conditions_have_matched_capacity_and_no_prediction_metadata(self):
        for classes in (3, 4):
            fitting, selection, policies, config, regularization = self.fixture(classes)
            expected = sum(parameter.numel() for parameter in
                           make_paired_model(fitting, classes, config).parameters())
            for method in METHODS:
                model, record = train_flat(
                    fitting, selection, policies, method, classes, 42, config, regularization,
                )
                self.assertEqual(record["parameter_count"], expected)
                self.assertEqual(record["active_contrast_weight"] > 0, method == "flat_policy")
                self.assertEqual(record["sam_used"], method != "two_pass_erm")
                self.assertFalse(record["pair_margin_used"])
                self.assertFalse(record["cosine_head_used"])
                probabilities = predict_paired(
                    model, {key: value for key, value in selection.items() if key != "metadata"}, 4,
                )
                self.assertEqual(probabilities.shape, (4 * classes, classes))
                self.assertTrue(torch.isfinite(torch.tensor(probabilities)).all())
                factor = 1 if method == "two_pass_erm" else 2
                self.assertEqual(record["gradient_evaluations"], record["optimizer_updates"] * factor)

    def test_repeat_training_is_deterministic(self):
        fitting, selection, policies, config, regularization = self.fixture(4)
        values = []
        for _ in range(2):
            model, record = train_flat(
                fitting, selection, policies, "flat_policy", 4, 42, config, regularization,
            )
            values.append(predict_paired(model, selection, 4))
        torch.testing.assert_close(torch.tensor(values[0]), torch.tensor(values[1]), atol=0, rtol=0)

    def test_invalid_regularization_and_policy_scope_are_rejected(self):
        fitting, selection, policies, config, regularization = self.fixture(4)
        for name, value in (("sam_radius", 0), ("temperature", -1), ("contrast_weight", float("nan"))):
            with self.assertRaises(ValueError):
                train_flat(fitting, selection, policies, "flat_policy", 4, 42, config,
                           {**regularization, name: value})
        with self.assertRaisesRegex(ValueError, "fitting policies"):
            train_flat(fitting, selection, {name: "one" for name in policies},
                       "flat_policy", 4, 42, config, regularization)


if __name__ == "__main__":
    unittest.main()
