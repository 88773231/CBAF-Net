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
from analysis.run_cosine_policy_screen import cached, replay, run, summarize, validate_settings, verified_candidates
from tests import test_factor_screen_roundtrip as fixture_module


class CosinePolicyScreenTests(unittest.TestCase):
    def settings(self):
        code = Path(__file__).parents[1]
        return [json.loads((code / "configs" / name).read_text()) for name in (
            "factor_training_screen.json", "factor_training_replication.json",
            "paired_policy_screen.json", "cosine_policy_screen.json",
        )]

    def test_fixed_scale_and_transparent_transfer_are_required(self):
        prior, _, _, settings = self.settings()
        validate_settings(settings, prior)
        for key, value in (
            ("regularization", {"contrast_weight": .1, "temperature": .2, "cosine_scale": 32}),
            ("new_methods", []), ("seeds", [42]), ("primary_task", "best_after_scoring"),
            ("report_transfer_and_all_class_policy_seed_results", False),
            ("original_val_or_test_paths_opened", True), ("submission_allowed", True),
            ("novelty_established", True), ("no_hyperparameter_search", False),
        ):
            changed = copy.deepcopy(settings)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed, prior)

    def test_train_replay_and_cache_removals_without_earlier_evidence_changes(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            prior_root, extension_root, paired_root, cosine_root = [
                root / name for name in ("prior", "replication", "paired", "cosine")
            ]
            settings = self.settings()
            reduced = {**settings[0]["configuration"], "max_epochs": 1, "patience": 1,
                       "hidden_dim": 8, "mlp_hidden_dim": 8, "embedding_dimension": 8, "feature_dim": 8,
                       "batch_size": 32, "knn_k": 2, "dropout": 0}
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
                _, cached_rows, _ = cached(prior_root, extension_root, paired_root, dataset)
                self.assertEqual(len(cached_rows), 36)
                removals = [row for row in cached_rows if row["method"] == "without_cosine_head"]
                self.assertEqual(len(removals), 9)
                self.assertTrue(all(row["original_method"] == "without_pair_margin" for row in removals))
                run(*inputs, dataset, protocols[3], prior_root, extension_root, paired_root, cosine_root / dataset)
                replay(*inputs, dataset, cosine_root, cosine_root / (dataset + "_audit"))
                self.assertEqual(preserved, {str(path): path.read_bytes() for study in (prior_root, extension_root, paired_root)
                                              for path in study.rglob("*") if path.is_file()})
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocols[3], prior_root, extension_root, paired_root, cosine_root / dataset)
            summarize(prior_root, extension_root, paired_root, cosine_root, cosine_root / "summary")
            report = json.loads((cosine_root / "summary/development_summary.json").read_text())
            self.assertEqual(report["new_fits"], 36)
            self.assertEqual(report["cached_fits"], 72)
            self.assertEqual(report["primary_task"], "Quadbot")
            self.assertEqual(report["transfer_task"], "Twibot22")
            for dataset in ("Twibot22", "Quadbot"):
                table = report["datasets"][dataset]
                self.assertEqual(len(table["primary_pooled"]), 6)
                self.assertEqual(table["new_runs_verified"], 18)
                self.assertEqual(table["cached_runs_verified"], 36)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            checkpoint = next((cosine_root / "Quadbot").rglob("selected_model.pt"))
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verified_candidates(cosine_root, "Quadbot")


if __name__ == "__main__":
    unittest.main()
