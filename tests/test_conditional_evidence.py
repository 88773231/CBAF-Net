import unittest

import torch

from src.conditional_evidence import (
    METHODS, ConditionalEvidenceModel, conditional_log_probs, conditional_loss, operation_targets,
)


class ConditionalEvidenceTests(unittest.TestCase):
    def inputs(self):
        return torch.randn(6, 8), torch.randn(6, 2, 8), torch.randn(6, 2, 15), torch.randn(6, 24)

    def model(self, method="conditional_bridge", classes=4):
        return ConditionalEvidenceModel(8, 8, 15, classes, steps=2, hidden_dim=8,
                                        dropout=0, method=method)

    def test_conditional_probability_has_no_discarded_combinations(self):
        for classes in (3, 4):
            generation, operation = torch.randn(6, 2), torch.randn(6, classes - 1)
            logs = conditional_log_probs(generation, operation, classes)
            expected = torch.stack([
                operation.softmax(1)[:, 0],
                operation.softmax(1)[:, 1] * generation.softmax(1)[:, 0],
                operation.softmax(1)[:, 1] * generation.softmax(1)[:, 1],
                *([operation.softmax(1)[:, 2]] if classes == 4 else []),
            ], dim=1)
            torch.testing.assert_close(logs.exp(), expected)
            torch.testing.assert_close(logs.exp().sum(1), torch.ones(6))

    def test_human_and_agent_probabilities_do_not_depend_on_subtype_head(self):
        operation = torch.randn(6, 3)
        first = conditional_log_probs(torch.randn(6, 2), operation, 4)
        second = conditional_log_probs(torch.randn(6, 2) * 10, operation, 4)
        torch.testing.assert_close(first[:, [0, 3]], second[:, [0, 3]], atol=0, rtol=0)
        torch.testing.assert_close(
            first.exp()[:, 1:3].sum(1), second.exp()[:, 1:3].sum(1),
        )

    def test_direct_leaves_block_subtype_classification_gradient(self):
        generation, operation = torch.randn(2, 2, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        logs = conditional_log_probs(generation, operation, 4)
        (-logs[torch.arange(2), torch.tensor([0, 3])].mean()).backward()
        torch.testing.assert_close(generation.grad, torch.zeros_like(generation))
        self.assertGreater(float(operation.grad.abs().sum()), 0)

    def test_three_four_class_outputs_are_finite_and_normalized(self):
        for classes in (3, 4):
            for method in METHODS:
                outputs = self.model(method, classes)(*self.inputs())
                self.assertEqual(outputs["logits"].shape, (6, classes))
                self.assertTrue(torch.isfinite(outputs["logits"]).all())
                torch.testing.assert_close(outputs["logits"].softmax(1).sum(1), torch.ones(6))

    def test_content_bridge_is_physically_removed(self):
        for method in ("without_content_bridge", "without_both"):
            model = self.model(method)
            self.assertIsNone(model.content_bridge)
            self.assertFalse(any(name.startswith("content_bridge.") for name in model.state_dict()))

    def test_flat_control_keeps_operation_and_subtype_auxiliary_heads(self):
        model = self.model("without_conditional_routing")
        outputs = model(*self.inputs())
        conditional_loss(outputs, torch.tensor([0, 1, 2, 3, 1, 2]), 4, 0.1)[0].backward()
        for parameter in (model.flat_head.weight, model.operation_head.weight, model.generation_head.weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_bridge_receives_gradient_at_zero_initialization(self):
        model = self.model()
        torch.testing.assert_close(model.content_bridge.weight, torch.zeros_like(model.content_bridge.weight))
        conditional_loss(model(*self.inputs()), torch.tensor([0, 1, 2, 3, 1, 2]), 4, 0.1)[0].backward()
        self.assertGreater(float(model.content_bridge.weight.grad.abs().sum()), 0)

    def test_zero_bridge_starts_identically_to_bridge_removal(self):
        torch.manual_seed(42)
        full = self.model()
        torch.manual_seed(42)
        removed = self.model("without_content_bridge")
        inputs = self.inputs()
        torch.testing.assert_close(full(*inputs)["logits"], removed(*inputs)["logits"], atol=0, rtol=0)

    def test_common_parameters_share_exact_initialization(self):
        torch.manual_seed(42)
        full = self.model()
        for method in METHODS[1:]:
            torch.manual_seed(42)
            removed = self.model(method)
            for name, parameter in full.state_dict().items():
                if name in removed.state_dict():
                    torch.testing.assert_close(parameter, removed.state_dict()[name], atol=0, rtol=0)

    def test_generation_auxiliary_only_uses_traditional_and_llm_rows(self):
        model = self.model()
        outputs = model(*self.inputs())
        labels = torch.tensor([0, 1, 2, 3, 0, 3])
        _, auxiliary = conditional_loss(outputs, labels, 4, 0.1)
        auxiliary["generation_loss"].backward()
        self.assertTrue(torch.isfinite(auxiliary["generation_loss"]))
        altered = {**outputs, "generation_logits": outputs["generation_logits"].detach().clone()}
        altered["generation_logits"][[0, 3, 4, 5]] += torch.randn(4, 2) * 100
        _, second = conditional_loss(altered, labels, 4, 0.1)
        torch.testing.assert_close(auxiliary["generation_loss"], second["generation_loss"])

    def test_no_subtype_rows_has_finite_zero_auxiliary(self):
        outputs = self.model()(*self.inputs())
        loss, auxiliary = conditional_loss(outputs, torch.tensor([0, 3, 0, 3, 0, 3]), 4, 0.1)
        self.assertEqual(float(auxiliary["generation_loss"]), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_fixed_taxonomy_and_invalid_inputs_rejected(self):
        self.assertEqual(operation_targets(4).tolist(), [0, 1, 1, 2])
        with self.assertRaises(ValueError):
            operation_targets(2)
        for generation, operation in (
            (torch.randn(2, 3), torch.randn(2, 3)),
            (torch.randn(2, 2), torch.randn(3, 3)),
            (torch.full((2, 2), float("nan")), torch.randn(2, 3)),
        ):
            with self.assertRaises(ValueError):
                conditional_log_probs(generation, operation, 4)
        inputs = list(self.inputs())
        inputs[3] = inputs[3][:, :-1]
        with self.assertRaisesRegex(ValueError, "schema"):
            self.model()(*inputs)
        with self.assertRaises(ValueError):
            self.model("best_challenge_score")
        with self.assertRaises(ValueError):
            conditional_loss(self.model()(*self.inputs()), torch.tensor([0, 1, 2, 3, 1, 2]), 4, -1)


if __name__ == "__main__":
    unittest.main()
