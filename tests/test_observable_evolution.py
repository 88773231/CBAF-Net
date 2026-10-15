import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import analysis.run_observable_evolution as runner
from dataset_pipeline.observable_cues import FEATURE_DIMENSIONS, STEPS, observe
from src.observable_evolution import (
    CONDITIONS, INPUTS, CausalCueEvolution, ObservableEvolutionModel, cue_distillation,
    shared_digest, train,
)


def public_row(offset=0):
    origin = datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)
    return {
        "profile": {"description": "public description", "public_metrics": {"followers_count": 10}},
        "posts": [{
            "text": "A public example with #topic <URL>",
            "action_type": ("post", "reply", "retweet")[index % 3],
            "created_at": (origin + timedelta(seconds=60 * index)).isoformat(),
            "interaction_targets": ["opaque-target"] if index % 2 else [],
            "time_step": index // 5 + 1,
        } for index in range(25)],
    }


class ObservableEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, count=4):
        return [torch.randn(count, STEPS, dimension) if name not in ("description", "numeric")
                else torch.randn(count, dimension) for name, dimension in FEATURE_DIMENSIONS.items()]

    def test_internal_fields_ids_steps_and_absolute_date_are_not_features(self):
        original = public_row()
        expected = observe(original)
        changed = public_row(offset=86400 * 365)
        changed.update({"label": 3, "base_profile_id": "different", "sample_id": "different",
                        "provenance": {"model": "secret"}, "actions": "not read", "graph": "not read"})
        for post in changed["posts"]:
            post.update({"time_step": -100, "post_id": "different", "environment_triggered": True})
            post["interaction_targets"] = ["different"] if post["interaction_targets"] else []
        actual = observe(changed)
        for name in INPUTS:
            np.testing.assert_array_equal(expected[name], actual[name])

    def test_shared_text_hash_namespace_across_identical_windows(self):
        values = observe(public_row())
        for window in values["text"][1:]:
            np.testing.assert_array_equal(values["text"][0], window)
        self.assertEqual(values["behavior"].shape, (5, 8))
        self.assertEqual(values["timing"].shape, (5, 7))

    def test_invalid_timestamps_or_short_history_fail(self):
        row = public_row()
        row["posts"][0]["created_at"] = ""
        with self.assertRaises(ValueError):
            observe(row)
        row["posts"] = row["posts"][:4]
        with self.assertRaises(ValueError):
            observe(row)

    def test_causal_evolution_does_not_read_future_cue_states(self):
        module = CausalCueEvolution(8).eval()
        states = torch.randn(3, 5, 8)
        changed = states.clone()
        changed[:, 3:] += 100
        torch.testing.assert_close(module(states)[:, :3], module(changed)[:, :3], atol=0, rtol=0)

    def test_self_distillation_does_not_backpropagate_through_teacher(self):
        student, teacher = torch.randn(4, 4, requires_grad=True), torch.randn(4, 4, requires_grad=True)
        cue_distillation(student, teacher, 2).backward()
        self.assertGreater(float(student.grad.abs().sum()), 0)
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            cue_distillation(student, teacher, 0)

    def test_variants_share_active_initialization_and_all_shapes_work(self):
        digests = set()
        for condition in CONDITIONS:
            torch.manual_seed(42)
            model = ObservableEvolutionModel(condition=condition).eval()
            output = model(*self.inputs())
            self.assertEqual(output["logits"].shape, (4, 4))
            if condition != "same_input_mlp":
                digests.add(shared_digest(model))
        self.assertEqual(len(digests), 1)
        with self.assertRaises(ValueError):
            ObservableEvolutionModel(condition="unsupported")

    def test_fitting_updates_model_and_preserves_profile_boundaries(self):
        populations, policies = [], {}
        for prefix in ("fit", "selection"):
            rows = []
            for policy in ("known-a", "known-b"):
                profile = prefix + "-" + policy
                policies[profile] = policy
                rows.extend({"base_profile_id": profile, "sample_id": profile + str(label),
                             "label": label} for label in range(4))
            populations.append({"metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
                                **dict(zip(INPUTS, self.inputs(8)))})
        config = {**runner.CONFIGURATION, "max_epochs": 2, "patience": 1}
        model, fitted = train(*populations, policies, CONDITIONS[0], config)
        self.assertIn(fitted["best_epoch"], (1, 2))
        self.assertFalse(fitted["submission_allowed"])
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        overlapping = copy.deepcopy(populations[1])
        overlapping["metadata"][0]["base_profile_id"] = populations[0]["metadata"][0]["base_profile_id"]
        with self.assertRaises(ValueError):
            train(populations[0], overlapping, policies, CONDITIONS[0], config)

    def test_excluded_policy_is_not_extracted_and_scaling_uses_fitting_only(self):
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
                            row["posts"] = "excluded rows must not be transformed"
                        rows.append(row)
            for policy_index, policy in enumerate(runner.POLICIES):
                fitting, selection, excluded = [], [], []
                for index in range(len(rows)):
                    group = index // 8
                    (excluded if group == policy_index else fitting if index % 8 < 4 else selection).append(index)
                receipts[policy] = {"fitting_indices": fitting, "selection_indices": selection,
                                    "challenge_indices": excluded}
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metadata.write_text("".join(json.dumps(row) + "\n" for row in identities), encoding="utf-8")
            with patch.object(runner, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()), \
                    patch.object(runner, "METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest()):
                values, _, normalization = runner.prepare_fold(source, metadata, receipts, runner.POLICIES[0])
            self.assertEqual(len(values[0]["labels"]), 8)
            self.assertAlmostEqual(normalization["numeric"]["mean"][0], np.log1p(10), places=5)
            self.assertTrue(torch.all(values[1]["numeric"][:, 0] == 0))


if __name__ == "__main__":
    unittest.main()
