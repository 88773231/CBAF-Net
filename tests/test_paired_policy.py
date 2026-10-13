import unittest

import torch

from src.evidence_model import MatchedFeatureMLP
from src.paired_policy import (
    METHODS, PairedPolicyModel, cross_policy_contrast, matched_boundary_margin,
    predict_paired, train_paired,
)
from src.policy_robust import train_policy_model


class PairedPolicyTests(unittest.TestCase):
    def test_identical_parameters_logits_and_dropout_to_mlp(self):
        for classes in (3, 4):
            inputs = (torch.randn(8, 3), torch.randn(8, 2, 3),
                      torch.randn(8, 2, 4), torch.randn(8, 5))
            torch.manual_seed(42)
            full = PairedPolicyModel(22, classes, 8, .2)
            torch.manual_seed(42)
            baseline = MatchedFeatureMLP(22, classes, 8, .2)
            self.assertEqual(set(full.state_dict()), set(baseline.state_dict()))
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, baseline.state_dict()[name], atol=0, rtol=0)
            torch.manual_seed(17)
            left = full(*inputs)["logits"]
            torch.manual_seed(17)
            right = baseline(*inputs)["logits"]
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_contrast_is_finite_and_alignment_improves_it(self):
        labels, policies = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
        aligned = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
        reversed_values = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
        loss = cross_policy_contrast(aligned, labels, policies, .2)
        self.assertLess(float(loss), float(cross_policy_contrast(reversed_values, labels, policies, .2)))
        loss.backward()
        self.assertTrue(torch.isfinite(aligned.grad).all())

    def test_same_policy_batch_has_differentiable_zero_contrast(self):
        value = torch.randn(6, 8, requires_grad=True)
        loss = cross_policy_contrast(value, torch.tensor([0, 1, 2, 0, 1, 2]),
                                     torch.zeros(6, dtype=torch.long), .2)
        self.assertEqual(float(loss), 0)
        loss.backward()
        torch.testing.assert_close(value.grad, torch.zeros_like(value))

    def test_contrast_handles_no_negatives_without_nan(self):
        value = torch.randn(4, 8, requires_grad=True)
        loss = cross_policy_contrast(value, torch.zeros(4, dtype=torch.long),
                                     torch.tensor([0, 0, 1, 1]), .2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_margin_targets_only_paired_human_traditional_boundary(self):
        labels, profiles = torch.tensor([0, 1, 2, 3]), ["profile"] * 4
        good = torch.tensor([[2., -2., 0., 0.], [-2., 2., 0., 0.],
                             [0., 0., 2., 0.], [0., 0., 0., 2.]], requires_grad=True)
        loss = matched_boundary_margin(good, labels, profiles, 1)
        reversed_values = good.detach().clone()
        reversed_values[:2] = reversed_values[:2].flip(0)
        self.assertLess(float(loss), float(matched_boundary_margin(reversed_values, labels, profiles, 1)))
        loss.backward()
        torch.testing.assert_close(good.grad[2:], torch.zeros_like(good.grad[2:]))
        torch.testing.assert_close(good.grad[:, 2:], torch.zeros_like(good.grad[:, 2:]))

    def test_margin_is_row_permutation_and_shared_logit_shift_invariant(self):
        labels, profiles = torch.tensor([0, 1, 2, 0, 1, 2]), ["a"] * 3 + ["b"] * 3
        values, order = torch.randn(6, 3), torch.tensor([5, 1, 3, 2, 4, 0])
        loss = matched_boundary_margin(values, labels, profiles, 1)
        torch.testing.assert_close(loss, matched_boundary_margin(
            values[order], labels[order], [profiles[i] for i in order], 1,
        ))
        # Adding the same Human/Traditional logit offsets to both matched rows cancels.
        offsets = torch.randn(2, 3).repeat_interleave(3, dim=0)
        torch.testing.assert_close(loss, matched_boundary_margin(values + offsets, labels, profiles, 1))

    def test_invalid_losses_rejected(self):
        with self.assertRaises(ValueError):
            cross_policy_contrast(torch.randn(3, 8), torch.tensor([0, 1, 2]),
                                  torch.tensor([0, 1, 0]), 0)
        with self.assertRaises(ValueError):
            matched_boundary_margin(torch.randn(3, 3), torch.tensor([0, 0, 1]), ["a"] * 3, 1)
        with self.assertRaises(ValueError):
            matched_boundary_margin(torch.randn(3, 3), torch.tensor([0, 1, 2]), ["a", "b", "b"], 1)

    def values(self, prefix, classes):
        rows = [{"base_profile_id": prefix + str(i), "label": label}
                for i in range(4) for label in range(classes)]
        return {
            "metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
            "description": torch.randn(len(rows), 3), "text": torch.randn(len(rows), 2, 3),
            "action": torch.randn(len(rows), 2, 4), "tabular": torch.randn(len(rows), 5),
        }

    def settings(self):
        return {"mlp_hidden_dim": 8, "dropout": .2, "learning_rate": .001,
                "weight_decay": .001, "max_epochs": 1, "patience": 1, "batch_size": 12}

    def test_zero_regularization_reproduces_erm_training_exactly(self):
        for classes in (3, 4):
            fitting, selection = self.values("f", classes), self.values("s", classes)
            policies = {prefix + str(i): "policy" + str(i % 2)
                        for prefix in ("f", "s") for i in range(4)}
            regularization = {"contrast_weight": 0., "pair_weight": 0., "temperature": .2, "margin": 1.}
            model, _ = train_paired(fitting, selection, policies, METHODS[0], classes,
                                    42, self.settings(), regularization)
            control, _ = train_policy_model(fitting, selection, policies, "erm_mlp",
                                            classes, 42, self.settings())
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, control.state_dict()[name], atol=0, rtol=0)

    def test_all_conditions_train_with_no_metadata_at_prediction(self):
        for classes in (3, 4):
            fitting, selection = self.values("f", classes), self.values("s", classes)
            policies = {prefix + str(i): "policy" + str(i % 2)
                        for prefix in ("f", "s") for i in range(4)}
            regularization = {"contrast_weight": .1, "pair_weight": .1, "temperature": .2, "margin": 1.}
            for method in METHODS:
                model, record = train_paired(fitting, selection, policies, method,
                                              classes, 42, self.settings(), regularization)
                self.assertEqual(record["active_contrast_weight"] > 0,
                                 method != "without_policy_contrast")
                self.assertEqual(record["active_pair_weight"] > 0, method != "without_pair_margin")
                self.assertFalse(record["policy_or_profile_metadata_used_as_prediction_input"])
                without_metadata = {key: value for key, value in selection.items() if key != "metadata"}
                probabilities = predict_paired(model, without_metadata, 4)
                self.assertEqual(probabilities.shape, (4 * classes, classes))
                self.assertTrue(torch.isfinite(torch.tensor(probabilities)).all())


if __name__ == "__main__":
    unittest.main()
