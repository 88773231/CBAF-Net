import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from analysis.run_factor_training_screen import POLICIES, replay, run, summarize
from src.factor_evidence import train_factor_model


class FactorScreenRoundtripTests(unittest.TestCase):
    def fixture(self, root, classes):
        features, temporal, partitions = (root / name for name in ("features", "timing", "partitions"))
        for folder in ("descriptions", "numerical", "graph"):
            (features / folder).mkdir(parents=True)
        temporal.mkdir()
        rows = [
            {"sample_id": "{}_{}".format(profile, label), "base_profile_id": profile,
             "label": label}
            for policy in range(3) for index in range(6)
            for profile in ["p{}_{}".format(policy, index)] for label in range(classes)
        ]
        serialized = "".join(json.dumps(row) + "\n" for row in rows)
        (features / "train_metadata.jsonl").write_text(serialized)
        (temporal / "train_metadata.jsonl").write_text(serialized)
        torch.manual_seed(42)
        torch.save(torch.tensor([row["label"] for row in rows]), features / "train_labels.pt")
        torch.save(torch.randn(len(rows), 8), features / "descriptions/train_des_tensor.pt")
        torch.save(torch.randn(len(rows), 6), features / "numerical/train_num_properties_tensor.pt")
        torch.save(torch.randn(len(rows), 11), features / "numerical/train_llm_features.pt")
        torch.save(torch.randn(len(rows), 8), features / "graph/train_node_features.pt")
        torch.save(torch.randn(len(rows), 7), temporal / "train_global.pt")
        torch.save(torch.randn(len(rows), 5, 7), temporal / "train_steps.pt")
        for step in range(1, 6):
            torch.save(torch.randn(len(rows), 8), features / f"train_text_tensor{step}.pt")
            torch.save(torch.randn(len(rows), 8), features / f"train_amr_tensor{step}.pt")
        for index, policy in enumerate(POLICIES):
            kept = [position for position, row in enumerate(rows)
                    if not row["base_profile_id"].startswith("p{}_".format(index))]
            selection = [position for position in kept if rows[position]["base_profile_id"].endswith("_0")]
            (partitions / policy).mkdir(parents=True)
            (partitions / policy / "normalization.json").write_text(json.dumps({
                "fitting_indices": [position for position in kept if position not in selection],
                "selection_indices": selection,
            }))
        return features, temporal, partitions

    def test_train_replay_and_complete_summary_without_val_test_files(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root, evidence = Path(folder), Path(folder) / "evidence"
            settings = json.loads((Path(__file__).parents[1] / "configs/factor_training_screen.json").read_text())
            settings["configuration"].update({
                "max_epochs": 1, "patience": 1, "hidden_dim": 8,
                "mlp_hidden_dim": 8, "embedding_dimension": 8, "feature_dim": 8,
                "batch_size": 32, "knn_k": 2, "dropout": 0,
            })
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(settings))
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = self.fixture(root / dataset, classes)
                run(*inputs, dataset, protocol, evidence / dataset)
                replay(*inputs, dataset, evidence, evidence / (dataset + "_audit"))
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocol, evidence / dataset)
            summarize(evidence, evidence / "summary")
            summary = json.loads((evidence / "summary/development_summary.json").read_text())
            self.assertFalse(summary["submission_allowed"])
            self.assertEqual(summary["datasets"]["Twibot22"]["runs_verified"], 18)
            self.assertEqual(summary["datasets"]["Quadbot"]["runs_verified"], 18)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))

    def test_training_rejects_overlapping_profile_before_model_creation(self):
        values = {
            "labels": torch.tensor([0, 1, 2]),
            "metadata": [{"base_profile_id": "p", "label": label} for label in range(3)],
        }
        with self.assertRaisesRegex(ValueError, "crosses"):
            train_factor_model(values, values, {"p": "first"}, "factor_residual", 3, 42, {})


if __name__ == "__main__":
    unittest.main()
