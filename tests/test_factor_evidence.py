import unittest

import torch
from torch.nn import functional as F

from src.factor_evidence import (
    FactorEvidenceModel, factor_logits, factor_loss, taxonomy,
)


class FactorEvidenceTests(unittest.TestCase):
    def inputs(self):
        return torch.randn(6, 8), torch.randn(6, 2, 8), torch.randn(6, 2, 15), torch.randn(6, 24)

    def model(self, method="factor_residual", classes=4):
        return FactorEvidenceModel(8, 8, 15, classes, steps=2, hidden_dim=8,
                                   dropout=0, method=method)

    def test_fixed_taxonomy_and_invalid_class_count(self):
        generated, operation = taxonomy(4)
        self.assertEqual(generated.tolist(), [0, 0, 1, 1])
        self.assertEqual(operation.tolist(), [0, 1, 1, 2])
        with self.assertRaises(ValueError):
            taxonomy(2)

    def test_factor_potential_equals_normalized_product(self):
        left, right = torch.randn(6, 2), torch.randn(6, 3)
        potential = factor_logits(left, right, 4)
        expected = left.softmax(1)[:, [0, 0, 1, 1]] * right.softmax(1)[:, [0, 1, 1, 2]]
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(potential.softmax(1), expected)

    def test_three_and_four_class_predictions_are_normalized(self):
        for classes in (3, 4):
            result = self.model(classes=classes)(*self.inputs())
            self.assertEqual(result["logits"].shape, (6, classes))
            torch.testing.assert_close(result["logits"].softmax(1).sum(1), torch.ones(6))

    def test_residual_and_both_taxonomy_heads_receive_gradients(self):
        model = self.model()
        loss, auxiliary = factor_loss(model(*self.inputs()), torch.tensor([0, 1, 2, 3, 0, 1]), 4, 0.1)
        loss.backward()
        for parameter in (model.generation_head.weight, model.operation_head.weight,
                          model.residual[0][0].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(auxiliary))

    def test_without_residual_physically_omits_module(self):
        model = self.model("without_interaction_residual")
        self.assertIsNone(model.residual)
        result = model(*self.inputs())
        torch.testing.assert_close(result["logits"], result["base_logits"])
        self.assertFalse(any(name.startswith("residual.") for name in model.state_dict()))

    def test_flat_removal_keeps_auxiliary_supervision(self):
        model = self.model("without_factor_fusion")
        self.assertIsNotNone(model.flat_head)
        inputs = self.inputs()
        result = model(*inputs)
        factor_loss(result, torch.tensor([0, 1, 2, 3, 0, 1]), 4, 0.1)[0].backward()
        self.assertGreater(float(model.flat_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.generation_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.operation_head.weight.grad.abs().sum()), 0)

    def test_common_parameters_have_identical_seeded_initialization(self):
        torch.manual_seed(42)
        full = self.model()
        for method in ("without_factor_fusion", "without_interaction_residual"):
            torch.manual_seed(42)
            control = self.model(method)
            for name, value in control.state_dict().items():
                if name in full.state_dict():
                    torch.testing.assert_close(value, full.state_dict()[name], atol=0, rtol=0)

    def test_shared_inputs_consumed_without_prediction_labels(self):
        model = self.model()
        inputs = [value.requires_grad_() for value in self.inputs()]
        F.cross_entropy(model(*inputs)["logits"], torch.tensor([0, 1, 2, 3, 0, 1])).backward()
        for value in inputs:
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_invalid_schema_and_head_dimensions_rejected(self):
        values = list(self.inputs())
        values[3] = values[3][:, :-1]
        with self.assertRaisesRegex(ValueError, "schema"):
            self.model()(*values)
        with self.assertRaisesRegex(ValueError, "shapes"):
            factor_logits(torch.randn(2, 3), torch.randn(2, 3), 4)

    def test_unknown_method_and_invalid_objective_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.model("best_outer_score")
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            factor_loss(self.model()(*self.inputs()), torch.tensor([0, 1, 2, 3, 0, 1]), 4, -1)


if __name__ == "__main__":
    unittest.main()
