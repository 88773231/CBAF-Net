import unittest

import torch
from torch.nn import functional as F

from src.cosine_policy import (
    METHODS, CosinePolicyModel, predict_cosine, train_cosine,
)
from src.paired_policy import PairedPolicyModel
from tests import test_paired_policy as fixture_module


class CosinePolicyTests(unittest.TestCase):
    def inputs(self):
        return torch.randn(6, 3), torch.randn(6, 2, 3), torch.randn(6, 2, 4), torch.randn(6, 5)

    def model(self, classes=4):
        return CosinePolicyModel(22, classes, 8, 0, 16)

    def test_formula_and_bound_match_fixed_cosine_classifier(self):
        for classes in (3, 4):
            model, inputs = self.model(classes), self.inputs()
            output = model(*inputs)
            expected = 16 * F.linear(F.normalize(output["representation"], dim=1),
                                     F.normalize(model.prototype_weight, dim=1))
            torch.testing.assert_close(output["logits"], expected)
            self.assertTrue((output["logits"].abs() <= 16.00001).all())
            torch.testing.assert_close(output["logits"].softmax(1).sum(1), torch.ones(6))

    def test_positive_prototype_scale_does_not_change_readout(self):
        model, inputs = self.model(), self.inputs()
        before = model(*inputs)["logits"]
        with torch.no_grad():
            model.prototype_weight.mul_(torch.tensor([.5, 2., 5., 10.])[:, None])
        torch.testing.assert_close(before, model(*inputs)["logits"])

    def test_no_hidden_linear_head_or_trainable_scale(self):
        model = self.model()
        self.assertEqual(len(model.network), 4)
        self.assertNotIn("network.4.bias", model.state_dict())
        self.assertNotIn("network.4.weight", model.state_dict())
        self.assertFalse(model.logit_scale.requires_grad)
        self.assertEqual(set(dict(model.named_parameters())).intersection({"logit_scale"}), set())

    def test_encoder_and_prototype_seed_initialization_match_linear_control(self):
        torch.manual_seed(42)
        model = self.model()
        torch.manual_seed(42)
        control = PairedPolicyModel(22, 4, 8, 0)
        for name, value in model.network.state_dict().items():
            torch.testing.assert_close(value, control.network.state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(model.prototype_weight, control.network[4].weight, atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in control.parameters()) -
                         sum(p.numel() for p in model.parameters()), 4)

    def test_zero_prototype_and_representation_are_finite(self):
        model = self.model()
        with torch.no_grad():
            model.prototype_weight.zero_()
        result = model(*self.inputs())["logits"]
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_multiclass_gradient_reaches_encoder_and_prototypes(self):
        model = self.model()
        loss = F.cross_entropy(model(*self.inputs())["logits"], torch.tensor([0, 1, 2, 3, 0, 1]))
        loss.backward()
        for parameter in (model.network[0].weight, model.prototype_weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_invalid_dimensions_and_scales_rejected(self):
        for args in ((22, 2, 8, 0, 16), (22, 4, 8, 0, 0),
                     (22, 4, 8, 0, float("nan")), (22, 4, 8, 1, 16)):
            with self.assertRaises(ValueError):
                CosinePolicyModel(*args)

    def test_both_conditions_train_without_prediction_metadata(self):
        fixtures = fixture_module.PairedPolicyTests()
        for classes in (3, 4):
            fitting, selection = fixtures.values("f", classes), fixtures.values("s", classes)
            policies = {prefix + str(i): "policy" + str(i % 2)
                        for prefix in ("f", "s") for i in range(4)}
            regularization = {"contrast_weight": .1, "temperature": .2, "cosine_scale": 16}
            counts = []
            for method in METHODS:
                model, record = train_cosine(fitting, selection, policies, method, classes, 42,
                                              fixtures.settings(), regularization)
                counts.append(record["parameter_count"])
                self.assertEqual(record["active_contrast_weight"] > 0, method == "cosine_policy")
                self.assertFalse(record["pair_margin_used"])
                predictions = predict_cosine(model, {key: value for key, value in selection.items()
                                                     if key != "metadata"}, 4)
                self.assertEqual(predictions.shape, (4 * classes, classes))
                self.assertTrue(torch.isfinite(torch.tensor(predictions)).all())
            self.assertEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
