import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from analysis import replicate_factor_training_screen as replication
from analysis import run_factor_training_screen as original
from analysis import run_paired_policy_screen as paired
from analysis.run_flat_policy_screen import (
    cached, expected_conditions, replay, run, summarize, validate_settings, verified,
)
from tests import test_factor_screen_roundtrip as fixture_module


class FlatPolicyScreenTests(unittest.TestCase):
    def settings(self):
        code = Path(__file__).parents[1]
        return [json.loads((code / "configs" / name).read_text()) for name in (
            "factor_training_screen.json", "factor_training_replication.json",
            "paired_policy_screen.json", "flat_policy_screen.json",
        )]

    def test_fixed_recipe_compute_control_and_axis_probes_are_required(self):
        prior, _, _, settings = self.settings()
        validate_settings(settings, prior)
        for key, value in (
            ("regularization", {"contrast_weight": .1, "temperature": .2, "sam_radius": .1}),
            ("new_fits", 54), ("axis_pairs", {}), ("axis_probe_is_independent_confirmation", True),
            ("same_epoch_not_same_compute", False), ("compute_control", "no_extra_control"),
            ("report_all_conditions_classes_policies_seeds_and_axes", False),
            ("original_val_or_test_paths_opened", True), ("submission_allowed", True),
            ("novelty_established", True), ("no_hyperparameter_search", False),
        ):
            changed = copy.deepcopy(settings)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed, prior)

    def test_expected_conditions_separate_training_seed_and_partition(self):
        expected = expected_conditions()
        self.assertEqual(len(expected), 51)
        self.assertEqual(sum(row[0] == "main" for row in expected), 27)
        self.assertEqual(sum(row[0] == "fixed_partition" for row in expected), 12)
        self.assertEqual(sum(row[0] == "fixed_training_seed" for row in expected), 12)
        self.assertTrue(all(row[2] == 42 for row in expected if row[0] == "fixed_partition"))
        self.assertTrue(all(row[1] == 42 for row in expected if row[0] == "fixed_training_seed"))

    def test_all_conditions_replay_pool_and_preserve_cached_evidence(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            prior_root, extension_root, paired_root, flat_root = [
                root / name for name in ("prior", "replication", "paired", "flat")
            ]
            settings = self.settings()
            reduced = {**settings[0]["configuration"], "max_epochs": 1, "patience": 1,
                       "hidden_dim": 8, "mlp_hidden_dim": 8, "embedding_dimension": 8,
                       "feature_dim": 8, "batch_size": 32, "knn_k": 2, "dropout": 0}
            protocols = []
            for index, item in enumerate(settings):
                item["configuration"] = copy.deepcopy(reduced)
                path = root / f"protocol{index}.json"
                path.write_text(json.dumps(item))
                protocols.append(path)
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = fixture_module.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                original.run(*inputs, dataset, protocols[0], prior_root / dataset)
                original.replay(*inputs, dataset, prior_root, prior_root / (dataset + "_audit"))
                replication.run(*inputs, dataset, protocols[1], prior_root, extension_root / dataset)
                replication.replay_extension(*inputs, dataset, extension_root, extension_root / (dataset + "_audit"))
                paired.run(*inputs, dataset, protocols[2], prior_root, extension_root, paired_root / dataset)
                paired.replay(*inputs, dataset, paired_root, paired_root / (dataset + "_audit"))
                preserved = {str(path): path.read_bytes() for study in (prior_root, extension_root, paired_root)
                             for path in study.rglob("*") if path.is_file()}
                _, old_rows, _ = cached(prior_root, extension_root, paired_root, dataset)
                self.assertEqual(len(old_rows), 36)
                removals = [row for row in old_rows if row["method"] == "without_flatness"]
                self.assertEqual(len(removals), 9)
                self.assertTrue(all(row["original_method"] == "without_pair_margin" for row in removals))
                run(*inputs, dataset, protocols[3], prior_root, extension_root, paired_root, flat_root / dataset)
                replay(*inputs, dataset, flat_root, flat_root / (dataset + "_audit"))
                self.assertEqual(preserved, {str(path): path.read_bytes() for study in (prior_root, extension_root, paired_root)
                                              for path in study.rglob("*") if path.is_file()})
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocols[3], prior_root, extension_root, paired_root, flat_root / dataset)
            summarize(prior_root, extension_root, paired_root, flat_root, flat_root / "summary")
            report = json.loads((flat_root / "summary/development_summary.json").read_text())
            self.assertEqual(report["new_fits"], 102)
            self.assertEqual(report["cached_fits"], 72)
            for dataset in ("Twibot22", "Quadbot"):
                data = report["datasets"][dataset]
                self.assertEqual(len(data["primary_pooled"]), 7)
                self.assertEqual(data["new_runs_verified"], 51)
                self.assertEqual(data["cached_runs_verified"], 36)
                for axis in ("fixed_partition", "fixed_training_seed"):
                    pairs = data["axis_probes"][axis]["flat_policy"]["training_seed_partition_pairs"]
                    self.assertEqual(pairs[0], [42, 42])
                    self.assertEqual(len(pairs), 3)
                    self.assertEqual(len(data["axis_probes"][axis]["full_minus_erm"]["values"]), 3)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            checkpoint = next((flat_root / "Quadbot").rglob("selected_model.pt"))
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verified(flat_root, "Quadbot")


if __name__ == "__main__":
    unittest.main()
