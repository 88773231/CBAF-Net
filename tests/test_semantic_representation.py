import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import analysis.run_semantic_representation as runner
from dataset_pipeline.frozen_semantic import (
    DIMENSION, DIMENSIONS, FrozenSentenceEncoder, masked_mean, public_text_groups,
    representation_features,
)
from src.encoder_objective_diagnostic import state_digest
from src.observable_evolution import INPUTS
from src.semantic_representation import CONDITIONS, make_model, predict, train
from tests.test_observable_evolution import public_row


class FakeEncoder:
    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        vectors = []
        for text in texts:
            vector = np.zeros(DIMENSION, dtype=np.float32)
            if text:
                vector[hashlib.sha256(text.encode()).digest()[0]] = 1
            vectors.append(vector)
        return np.stack(vectors)


class SemanticRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def populations(self):
        output, policies = [], {}
        for prefix in ("fitting", "selection"):
            metadata = []
            for policy in ("a", "b"):
                profile = prefix + policy
                policies[profile] = policy
                metadata.extend({
                    "base_profile_id": profile, "sample_id": profile + str(label), "label": label,
                } for label in range(4))
            output.append({
                "metadata": metadata, "labels": torch.tensor([row["label"] for row in metadata]),
                **{name: torch.randn(
                    (len(metadata), 5, DIMENSIONS[name]) if name not in ("description", "numeric")
                    else (len(metadata), DIMENSIONS[name]),
                ) for name in INPUTS},
            })
        return output, policies

    def test_pooling_ignores_padding_and_handles_all_empty_mask(self):
        hidden = torch.tensor([[[3., 4.], [100., -200.]], [[5., 8.], [7., 2.]]])
        result = masked_mean(hidden, torch.tensor([[1, 0], [0, 0]]))
        torch.testing.assert_close(result, torch.tensor([[.6, .8], [0., 0.]]))
        with self.assertRaises(ValueError):
            masked_mean(hidden, torch.ones(2, 3))
        with self.assertRaises(ValueError):
            masked_mean(hidden, torch.full((2, 2), .5))

    def test_length_bucketing_preserves_text_order_empty_policy_and_statistics(self):
        batches = []

        def tokenizer(texts, **kwargs):
            lengths = [int(text) for text in texts]
            if kwargs.get("return_length"):
                return {"length": lengths}
            batches.append(lengths)
            return {
                "input_ids": torch.tensor(lengths).reshape(-1, 1),
                "attention_mask": torch.ones(len(lengths), 1, dtype=torch.long),
            }

        def model(input_ids, attention_mask):
            states = torch.zeros(len(input_ids), 1, DIMENSION)
            for index, value in enumerate(input_ids[:, 0]):
                states[index, 0, int(value) % DIMENSION] = 1
            return SimpleNamespace(last_hidden_state=states)

        encoder = FrozenSentenceEncoder.__new__(FrozenSentenceEncoder)
        encoder.tokenizer, encoder.model, encoder.batch_size = tokenizer, model, 2
        encoder.statistics = {"encoded_texts": 0, "truncated_texts": 0, "tokens_before_truncation": 0}
        values = encoder.encode(["99", "", "3", "300", "5"])
        self.assertEqual(batches, [[3, 5], [99, 300]])
        self.assertEqual(values.argmax(1).tolist(), [99, 0, 3, 300, 5])
        self.assertEqual(float(values[1].sum()), 0)
        self.assertEqual(encoder.statistics, {
            "encoded_texts": 4, "truncated_texts": 1, "tokens_before_truncation": 407,
        })
        self.assertEqual(encoder.encode([]).shape, (0, DIMENSION))
        encoder.embedding_cache = {}
        first = encoder.encode(["3", "5"])
        batches.clear()
        repeated = encoder.encode(["5", "3"])
        np.testing.assert_array_equal(first[[1, 0]], repeated)
        self.assertEqual(batches, [])
        self.assertEqual(encoder.statistics["frozen_text_cache_hits"], 2)
        repeated[0] = 0
        np.testing.assert_array_equal(encoder.encode(["5"]), first[1:])

    def test_inputs_exclude_internal_fields_and_absolute_dates(self):
        original = public_row()
        changed = public_row(offset=86400 * 100)
        changed.update({"label": "SECRET_LABEL", "base_profile_id": "SECRET_PROFILE",
                        "actions": "SECRET_ACTION", "provenance": "SECRET_PROVENANCE"})
        for post in changed["posts"]:
            post.update({"time_step": -999, "post_id": "SECRET_POST", "environment_triggered": True})
            post["interaction_targets"] = ["SECRET_TARGET"] if post["interaction_targets"] else []
        self.assertEqual(public_text_groups(original), public_text_groups(changed))
        encoder = FakeEncoder()
        first, _ = representation_features([original], encoder)
        second, _ = representation_features([changed], encoder)
        self.assertNotIn("secret", " ".join(encoder.seen))
        for kind in ("hash", "semantic"):
            for name in INPUTS:
                np.testing.assert_array_equal(first[kind][0][name], second[kind][0][name])

    def test_text_normalization_and_nontext_inputs_match(self):
        encoder = FakeEncoder()
        row = public_row()
        row["posts"][0]["text"] = "HELLO @one https://example.com #TOPIC"
        features, statistics = representation_features([row], encoder)
        self.assertIn("hello <user> <url> topic", encoder.seen)
        self.assertEqual(features["hash"][0]["description"].shape, (384,))
        self.assertEqual(features["semantic"][0]["text"].shape, (5, 384))
        for name in ("numeric", "behavior", "style", "timing"):
            np.testing.assert_array_equal(features["hash"][0][name], features["semantic"][0][name])
        self.assertFalse(statistics["gpu_used"])
        row["profile"] = {}
        self.assertEqual(public_text_groups(row)[0], [""])

    def test_pair_controls_have_exact_same_initialization_and_parameter_counts(self):
        for architecture in ("mlp", "backbone"):
            torch.manual_seed(42)
            first = make_model(runner.CONFIGURATION, architecture)
            torch.manual_seed(42)
            second = make_model(runner.CONFIGURATION, architecture)
            self.assertEqual(state_digest(first), state_digest(second))
            values, _ = self.populations()
            self.assertEqual(first(*(values[0][name] for name in INPUTS)).shape, (8, 4))
            self.assertEqual(sum(p.numel() for p in first.parameters()),
                             sum(p.numel() for p in second.parameters()))
        with self.assertRaises(ValueError):
            make_model(runner.CONFIGURATION, "retired_routing")

    def test_train_and_checkpoint_replay(self):
        values, policies = self.populations()
        config = {**runner.CONFIGURATION, "max_epochs": 2, "patience": 1}
        for architecture in ("mlp", "backbone"):
            model, fit = train(*values, policies, architecture, config)
            probabilities = predict(model, values[1], 4)
            replay = make_model(config, architecture)
            replay.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            np.testing.assert_array_equal(probabilities, predict(replay, values[1], 4))
            self.assertFalse(fit["encoder_finetuned"])
            self.assertFalse(fit["new_module_or_novelty_claim"])
        values[1]["metadata"][0]["base_profile_id"] = values[0]["metadata"][0]["base_profile_id"]
        with self.assertRaises(ValueError):
            train(*values, policies, "mlp", config)

    def test_configuration_is_exact_and_no_claim_follows_positive_screen(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/semantic_representation.json"
        protocol = json.loads(config_path.read_text())
        runner.validate_protocol(protocol)
        protocol["encoder"]["max_length"] = 512
        with self.assertRaises(ValueError):
            runner.validate_protocol(protocol)
        records = [{
            "policy": policy, "condition": condition,
            "inner_selection": {"all_profiles": {"macro_f1": .98 if condition.startswith("hash") else .99}},
        } for policy in runner.POLICIES for condition in CONDITIONS]
        result = runner.decision(records)
        self.assertTrue(result["representation_continuation_gate_passed"])
        self.assertFalse(result["submission_allowed"])
        self.assertFalse(result["algorithmic_novelty_established"])
        records[1]["inner_selection"]["all_profiles"]["macro_f1"] = .5
        self.assertFalse(runner.decision(records)["representation_continuation_gate_passed"])
        with self.assertRaises(ValueError):
            runner.decision(records[:-1])

    def test_excluded_rows_never_enter_encoder_and_scaling_uses_fitting_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, metadata = root / "train.jsonl", root / "train_metadata.jsonl"
            rows, identities, receipts = [], [], {}
            for policy_index, policy in enumerate(runner.POLICIES):
                for profile_index in range(2):
                    profile = policy + str(profile_index)
                    for label in range(4):
                        identity = {"row_index": len(rows), "sample_id": profile + str(label),
                                    "base_profile_id": profile, "label": label}
                        identities.append(identity)
                        row = {**public_row(), **identity}
                        row["profile"]["public_metrics"]["followers_count"] = 10 if profile_index == 0 else 10 ** 9
                        if policy_index == 0:
                            row["posts"] = "excluded rows must not enter either encoder"
                        rows.append(row)
            for policy_index, policy in enumerate(runner.POLICIES):
                fitting, selection, excluded = [], [], []
                for index in range(len(rows)):
                    (excluded if index // 8 == policy_index else
                     fitting if index % 8 < 4 else selection).append(index)
                receipts[policy] = {"fitting_indices": fitting, "selection_indices": selection,
                                    "challenge_indices": excluded}
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metadata.write_text("".join(json.dumps(row) + "\n" for row in identities), encoding="utf-8")
            with patch.object(runner, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()), \
                    patch.object(runner, "METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest()):
                allowed, populations, policies = runner.selected_rows(
                    source, metadata, receipts, runner.POLICIES[0],
                )
            features, _ = representation_features(allowed, FakeEncoder())
            values, transforms = runner.prepare_populations(features, populations, policies)
            self.assertEqual(len(values["semantic"][0]["labels"]), 8)
            self.assertAlmostEqual(transforms["numeric"]["mean"][0], np.log1p(10), places=5)
            self.assertTrue(torch.all(values["semantic"][1]["numeric"][:, 0] == 0))


if __name__ == "__main__":
    unittest.main()
