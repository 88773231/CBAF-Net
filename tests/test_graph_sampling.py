import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset_pipeline.audit_release_contract import audit_features
from dataset_pipeline.export_quadbot_v3_features import build_edge_index
from src.botdmm import BotDMM
from src.data_loader import (
    StrictOneHopNeighborCollator,
    build_full_graph_batch,
    validate_knn_pairs,
)
from src.layers.attention import StructuralAttention, TemporalMeanPooling


def ring_knn_pairs(num_nodes, k):
    targets = []
    neighbors = []
    for target in range(num_nodes):
        for offset in range(1, k + 1):
            targets.append(target)
            neighbors.append((target + offset) % num_nodes)
    return torch.tensor([targets, neighbors], dtype=torch.long)


class SyntheticDataset:
    def __init__(self, num_nodes=12, k=10, num_steps=2):
        generator = torch.Generator().manual_seed(17)
        self.num_samples = num_nodes
        self.des_tensor = torch.randn(num_nodes, 4, generator=generator)
        self.tweets_list = [
            torch.randn(num_nodes, 4, generator=generator) for _ in range(num_steps)
        ]
        self.amrs_list = [
            torch.randn(num_nodes, 4, generator=generator) for _ in range(num_steps)
        ]
        self.num_prop = torch.randn(num_nodes, 3, generator=generator)
        self.llm_features = torch.randn(num_nodes, 2, generator=generator)
        self.labels = torch.arange(num_nodes, dtype=torch.long) % 3
        pairs = ring_knn_pairs(num_nodes, k)
        self.edge_indices = [pairs.clone() for _ in range(num_steps)]
        self.base_profile_ids = [f"profile-{index}" for index in range(num_nodes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "_index": index,
            "des": self.des_tensor[index],
            "tweets": [value[index] for value in self.tweets_list],
            "amrs": [value[index] for value in self.amrs_list],
            "num_prop": self.num_prop[index],
            "llm_features": self.llm_features[index],
            "edge_indices": self.edge_indices,
            "label": self.labels[index],
        }


def seed_probabilities(model, batch):
    with torch.no_grad():
        logits = model(
            batch["des"],
            batch["tweets"],
            batch["amrs"],
            batch["num_prop"],
            batch["llm_features"],
            batch["edge_indices"],
            compute_auxiliary_losses=False,
        )["logits"]
    return torch.softmax(logits[batch["seed_positions"]], dim=1)


class GraphSamplingTests(unittest.TestCase):
    def test_feature_export_excludes_same_profile_by_default(self):
        graph_matrix = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
            dtype=np.float32,
        )
        base_profile_ids = ["shared", "shared", "other"]

        pairs = build_edge_index(
            graph_matrix,
            k=1,
            base_profile_ids=base_profile_ids,
        )

        for target, neighbor in zip(pairs[0].tolist(), pairs[1].tolist()):
            self.assertNotEqual(
                base_profile_ids[target],
                base_profile_ids[neighbor],
            )
        with self.assertRaisesRegex(ValueError, "base_profile_ids are required"):
            build_edge_index(graph_matrix, k=1)

    def test_release_audit_treats_same_profile_edges_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph").mkdir()
            torch.save(
                torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                root / "graph" / "train_edge_index.pt",
            )
            (root / "train_metadata.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "row_index": index,
                            "sample_id": f"sample-{index}",
                            "base_profile_id": "shared",
                            "label": index,
                        }
                    )
                    + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            (root / "feature_export_audit.json").write_text(
                json.dumps(
                    {
                        "schema_version": "quadbot-v3-feature-export-1",
                        "dimensions": {},
                        "rules": {
                            "knn_k": 1,
                            "knn_exclude_same_base_profile": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = audit_features(root, None, expected_k=1)

        self.assertTrue(
            any("sharing base_profile_id" in error for error in report["errors"])
        )

    def test_temporal_mean_pooling_name_and_checkpoint_keys_are_stable(self):
        pooling = TemporalMeanPooling(embedding_dim=4, dropout=0.0).eval()
        sequence = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                [[2.0, 1.0, 0.0, -1.0], [4.0, 3.0, 2.0, 1.0]],
            ]
        )
        expected = pooling.layernorm(sequence.mean(dim=1))
        torch.testing.assert_close(pooling(sequence), expected)
        self.assertEqual(
            set(pooling.state_dict()),
            {"layernorm.weight", "layernorm.bias"},
        )

        model = BotDMM(
            des_size=4,
            tweet_size=4,
            amr_size=4,
            num_prop_size=3,
            llm_features_size=2,
            embedding_dimension=8,
            feature_dim=8,
            num_temporal_steps=2,
            num_classes=3,
            ablation_mode="base",
        )
        temporal_keys = {
            key for key in model.state_dict() if key.startswith("temporal_pooling.")
        }
        self.assertEqual(
            temporal_keys,
            {
                "temporal_pooling.layernorm.weight",
                "temporal_pooling.layernorm.bias",
            },
        )

    def test_vectorized_attention_matches_grouped_reference(self):
        torch.manual_seed(11)
        layer = StructuralAttention(4, 4, dropout=0.0).eval()
        features = torch.randn(6, 4)
        edge_index = torch.tensor(
            [[1, 2, 3, 0, 4, 5], [0, 0, 0, 4, 4, 4]],
            dtype=torch.long,
        )

        actual = layer(features, edge_index)
        qk = layer.qk(features)
        values = layer.value(features)
        scores = (qk[edge_index[1]] * qk[edge_index[0]]).sum(dim=1) / (4 ** 0.5)
        scores = torch.clamp(scores, -5.0, 5.0)
        reference_messages = torch.zeros_like(features)
        for destination in torch.unique(edge_index[1]):
            mask = edge_index[1] == destination
            weights = torch.softmax(scores[mask], dim=0)
            sources = edge_index[0, mask]
            reference_messages[destination] = (weights.unsqueeze(1) * values[sources]).sum(dim=0)
        expected = layer.layernorm(layer.output_proj(reference_messages) + features)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

        layer.zero_grad(set_to_none=True)
        actual.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in layer.parameters()
        ))

    def test_every_seed_receives_all_ten_neighbors_with_correct_orientation(self):
        dataset = SyntheticDataset(num_nodes=12, k=10)
        collator = StrictOneHopNeighborCollator(dataset, expected_k=10, require_strict=True)
        batch = collator([dataset[0], dataset[5]])

        local_to_global = batch["node_global_indices"]
        for local_edges in batch["edge_indices"]:
            incoming = torch.bincount(local_edges[1], minlength=len(local_to_global))
            self.assertEqual(incoming[0].item(), 10)
            self.assertEqual(incoming[1].item(), 10)

            observed = {
                (int(local_to_global[dst]), int(local_to_global[src]))
                for src, dst in zip(local_edges[0].tolist(), local_edges[1].tolist())
            }
            expected = {
                (int(target), int(neighbor))
                for target, neighbor in zip(
                    dataset.edge_indices[0][0].tolist(),
                    dataset.edge_indices[0][1].tolist(),
                )
                if target in {0, 5}
            }
            self.assertEqual(observed, expected)

    def test_validation_rejects_same_profile_and_nonlocal_edges(self):
        same_profile_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "same-profile"):
            validate_knn_pairs(
                same_profile_pairs,
                num_nodes=3,
                base_profile_ids=["shared", "shared", "other"],
                expected_k=None,
                require_strict=True,
            )

        nonlocal_pairs = torch.tensor([[0], [3]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "not split-local"):
            validate_knn_pairs(
                nonlocal_pairs,
                num_nodes=3,
                expected_k=None,
                require_strict=False,
            )

    def test_sampler_matches_full_graph_and_eval_is_batch_size_invariant(self):
        dataset = SyntheticDataset(num_nodes=8, k=2, num_steps=2)
        model = BotDMM(
            des_size=4,
            tweet_size=4,
            amr_size=4,
            num_prop_size=3,
            llm_features_size=2,
            embedding_dimension=8,
            feature_dim=8,
            num_temporal_steps=2,
            dropout=0.0,
            temperature=0.1,
            alpha=0.5,
            num_classes=3,
            ablation_mode="base",
        )
        model.eval()

        full_batch = build_full_graph_batch(dataset, expected_k=2, require_strict=True)
        full_probabilities = seed_probabilities(model, full_batch)
        with torch.no_grad():
            outputs = model(
                full_batch["des"],
                full_batch["tweets"],
                full_batch["amrs"],
                full_batch["num_prop"],
                full_batch["llm_features"],
                full_batch["edge_indices"],
                compute_auxiliary_losses=False,
            )
        self.assertNotIn("view_gate_weights", outputs)

        for seed_batch_size in (1, 3, 5):
            collator = StrictOneHopNeighborCollator(
                dataset,
                expected_k=2,
                require_strict=True,
            )
            reconstructed = torch.empty_like(full_probabilities)
            for start in range(0, len(dataset), seed_batch_size):
                indices = list(range(start, min(start + seed_batch_size, len(dataset))))
                batch = collator([dataset[index] for index in indices])
                reconstructed[batch["seed_global_indices"]] = seed_probabilities(model, batch)
            torch.testing.assert_close(
                reconstructed,
                full_probabilities,
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
