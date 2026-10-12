import unittest

import torch

from src.policy_robust import (
    GroupDRO, NEURAL_METHODS, fitting_group_ids, profile_batches,
    selection_key, symmetric_kl, train_policy_model,
)


class PolicyRobustTests(unittest.TestCase):
    def test_symmetric_kl_zero_symmetric_and_differentiable(self):
        first = torch.tensor([[1.0, -1.0, 0.0], [0.0, 2.0, -2.0]], requires_grad=True)
        second = torch.tensor([[-1.0, 0.0, 2.0], [1.0, -1.0, 0.0]], requires_grad=True)
        torch.testing.assert_close(symmetric_kl(first, first), torch.zeros(2))
        torch.testing.assert_close(symmetric_kl(first, second), symmetric_kl(second, first))
        self.assertTrue((symmetric_kl(first, second) >= 0).all())
        symmetric_kl(first, second).mean().backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())

    def test_symmetric_kl_rejects_shape_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            symmetric_kl(torch.zeros(2, 3), torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            symmetric_kl(torch.full((2, 3), float("nan")), torch.zeros(2, 3))

    def test_robust_weight_increases_for_harder_group_and_preserves_gradients(self):
        objective = GroupDRO(2, 0.1)
        losses = torch.tensor([0.2, 0.4, 1.0, 2.0], requires_grad=True)
        value = objective.loss(losses, torch.tensor([0, 0, 1, 1]))
        self.assertGreater(objective.log_weights.softmax(0)[1],
                           objective.log_weights.softmax(0)[0])
        value.backward()
        self.assertTrue((losses.grad > 0).all())
        self.assertTrue(torch.isfinite(losses.grad).all())

    def test_absent_batch_group_and_unknown_group(self):
        objective = GroupDRO(3, 0.1)
        value = objective.loss(torch.tensor([1.0, 2.0]), torch.tensor([0, 0]))
        torch.testing.assert_close(value, torch.tensor(1.5))
        with self.assertRaises(ValueError):
            objective.loss(torch.ones(2), torch.tensor([0, 3]))

    def test_profile_sampler_is_complete_deterministic_and_no_duplicate_rows(self):
        rows = [{"base_profile_id": f"p{index}", "label": label}
                for index in range(9) for label in range(3)]
        first = list(profile_batches(rows, 3, 10, torch.Generator().manual_seed(42)))
        second = list(profile_batches(rows, 3, 10, torch.Generator().manual_seed(42)))
        self.assertEqual([row.tolist() for row in first], [row.tolist() for row in second])
        self.assertEqual(sorted(torch.cat(first).tolist()), list(range(27)))
        self.assertTrue(all(len(row) <= 10 and len(row) % 3 == 0 for row in first))
        with self.assertRaises(ValueError):
            list(profile_batches(rows[:-1], 3, 10, torch.Generator()))

    def test_fitting_vocabulary_requires_all_policy_class_pairs(self):
        rows = [{"base_profile_id": f"p{index}", "label": label}
                for index in range(4) for label in range(3)]
        policies = {f"p{index}": "a" if index < 2 else "b" for index in range(4)}
        group_ids, vocabulary = fitting_group_ids(rows, policies, 3)
        self.assertEqual(len(vocabulary), 6)
        self.assertEqual(group_ids.shape, (12,))
        with self.assertRaises(ValueError):
            fitting_group_ids([row for row in rows if row["label"] != 2], policies, 3)

    def test_checkpoint_selection_prefers_worst_known_policy_not_overall_score(self):
        better_worst = {"worst_policy_macro_f1": 0.8,
                        "all_profiles": {"macro_f1": 0.82, "mcc": 0.7}}
        better_average = {"worst_policy_macro_f1": 0.7,
                          "all_profiles": {"macro_f1": 0.9, "mcc": 0.8}}
        self.assertGreater(selection_key(better_worst), selection_key(better_average))

    def test_four_objectives_train_same_architecture_without_policy_prediction_input(self):
        def values(prefix):
            rows = [{"base_profile_id": f"{prefix}{index}", "label": label}
                    for index in range(4) for label in range(3)]
            return {
                "metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
                "description": torch.randn(12, 3), "text": torch.randn(12, 5, 4),
                "action": torch.randn(12, 5, 9), "tabular": torch.randn(12, 24),
            }
        fitting, selection = values("f"), values("s")
        policies = {f"{prefix}{index}": "a" if index < 2 else "b"
                    for prefix in ("f", "s") for index in range(4)}
        config = {
            "mlp_hidden_dim": 8, "dropout": 0.2, "learning_rate": 0.001,
            "weight_decay": 0.001, "max_epochs": 1, "patience": 1,
            "batch_size": 6, "dro_step_size": 0.01, "consistency_weight": 0.5,
        }
        counts = []
        for method in NEURAL_METHODS:
            model, record = train_policy_model(
                fitting, selection, policies, method, 3, 42, config
            )
            result = model(fitting["description"], fitting["text"],
                           fitting["action"], fitting["tabular"])
            self.assertEqual(result["logits"].shape, (12, 3))
            self.assertTrue(torch.isfinite(result["logits"]).all())
            self.assertFalse(record["group_metadata_used_as_prediction_input"])
            self.assertEqual(record["best_epoch"], 1)
            counts.append(record["parameter_count"])
        self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
