import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from analysis.run_botdmm_timing import graph_inputs, validate_reference
from src.botdmm_development import (
    DevelopmentGraphView, make_backbone, predict_backbone, train_backbone,
)
from src.data_loader import build_full_graph_batch
from src.provenance import sha256_file


class BotDMMDevelopmentTests(unittest.TestCase):
    def values(self, prefix="p", count=12):
        torch.manual_seed(17)
        rows = [{"base_profile_id": f"{prefix}{index}", "label": label}
                for index in range(count) for label in range(3)]
        n = len(rows)
        return {
            "metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
            "description": torch.randn(n, 4), "text": torch.randn(n, 5, 4),
            "action": torch.randn(n, 5, 11), "tabular": torch.randn(n, 24),
            "numerical": torch.randn(n, 13),
        }, torch.randn(n, 8)

    def config(self):
        return {
            "embedding_dimension": 8, "feature_dim": 8, "dropout": 0.0,
            "learning_rate": 0.001, "weight_decay": 0.001, "max_epochs": 1,
            "patience": 1, "batch_size": 12, "knn_k": 2,
        }

    def test_timing_only_changes_declared_columns_not_labels_or_graph(self):
        values, graph = self.values()
        timing = DevelopmentGraphView(values, graph, "botdmm_timing", k=2)
        old = DevelopmentGraphView(values, graph, "botdmm_without_timing", k=2)
        torch.testing.assert_close(timing.labels, old.labels)
        torch.testing.assert_close(timing.edge_pairs, old.edge_pairs)
        torch.testing.assert_close(timing.llm_features, old.llm_features)
        torch.testing.assert_close(timing.num_prop[:, :-7], old.num_prop)
        torch.testing.assert_close(timing.amrs_list[0][:, :-7], old.amrs_list[0])
        for target, neighbor in timing.edge_pairs.T.tolist():
            self.assertNotEqual(timing.base_profile_ids[target], timing.base_profile_ids[neighbor])

    def test_fold_graph_is_rebuilt_and_ignores_labels(self):
        values, graph = self.values()
        reference = DevelopmentGraphView(values, graph, "botdmm_timing", k=2)
        changed = {**values, "labels": 2 - values["labels"]}
        result = DevelopmentGraphView(changed, graph, "botdmm_timing", k=2)
        torch.testing.assert_close(reference.edge_pairs, result.edge_pairs)
        self.assertLess(int(result.edge_pairs.max()), len(values["labels"]))
        self.assertFalse(result.graph_receipt["true_social_edges"])
        self.assertFalse(result.graph_receipt["uses_labels_or_policy"])

    def test_seed_inference_matches_full_graph_and_batch_size_is_invariant(self):
        values, graph = self.values()
        dataset = DevelopmentGraphView(values, graph, "botdmm_timing", k=2)
        model = make_backbone(dataset, 3, self.config()).eval()
        first = predict_backbone(model, dataset, 12, 2)
        second = predict_backbone(model, dataset, 18, 2)
        torch.testing.assert_close(torch.tensor(first), torch.tensor(second), atol=1e-6, rtol=0)
        batch = build_full_graph_batch(dataset, expected_k=2)
        with torch.no_grad():
            full = model(
                batch["des"], batch["tweets"], batch["amrs"], batch["num_prop"],
                batch["llm_features"], batch["edge_indices"], compute_auxiliary_losses=False,
            )["logits"].softmax(1)
        torch.testing.assert_close(torch.tensor(first), full, atol=1e-6, rtol=0)

    def test_one_epoch_finite_and_training_selection_profiles_disjoint(self):
        fitting_values, fit_graph = self.values("f")
        selection_values, select_graph = self.values("s")
        fitting = DevelopmentGraphView(fitting_values, fit_graph, "botdmm_timing", k=2)
        selection = DevelopmentGraphView(selection_values, select_graph, "botdmm_timing", k=2)
        policies = {f"{prefix}{index}": "a" if index < 6 else "b"
                    for prefix in ("f", "s") for index in range(12)}
        progress = []
        model, record = train_backbone(
            fitting, selection, selection_values, policies, 3, 42, self.config(),
            progress=progress.append,
        )
        self.assertEqual(record["best_epoch"], 1)
        self.assertEqual(len(progress), len(record["history"]))
        self.assertEqual(progress[0]["epoch"], 1)
        self.assertFalse(record["policy_metadata_used_as_prediction_input"])
        self.assertTrue(torch.isfinite(torch.tensor(predict_backbone(model, selection, 12, 2))).all())
        with self.assertRaises(ValueError):
            train_backbone(fitting, fitting, fitting_values, policies, 3, 42, self.config())

    def test_test_graph_path_rejected_before_loading(self):
        with patch("analysis.run_botdmm_timing.torch.load") as load:
            with self.assertRaises(ValueError):
                graph_inputs(Path("unused"), "test", 12)
            load.assert_not_called()

    def test_reference_requires_matching_recipe_and_intact_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "Twibot22"
            target.mkdir()
            protocol = {
                "seeds": [42, 43, 44], "test_paths_opened": False,
                "outer_challenge_used_for_selection": False, "methods": ["erm_mlp"],
                "configuration": {"dropout": 0.2},
            }
            path = target / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            (target / "summary.json").write_text(json.dumps({
                "protocol_sha256": sha256_file(path), "test_state": "NOT_OPENED",
            }), encoding="utf-8")
            frozen = {"mlp_reference": {"method": "erm_mlp", "same_recipe_keys": ["dropout"]}}
            self.assertEqual(
                validate_reference(root, "Twibot22", [42, 43, 44], {"dropout": 0.2}, frozen),
                protocol,
            )
            with self.assertRaises(ValueError):
                validate_reference(root, "Twibot22", [42, 43, 44], {"dropout": 0.3}, frozen)
            protocol["test_paths_opened"] = True
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_reference(root, "Twibot22", [42, 43, 44], {"dropout": 0.2}, frozen)


if __name__ == "__main__":
    unittest.main()
