import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from analysis import replicate_factor_training_screen as replication
from analysis import run_factor_training_screen as screen
from analysis.run_paired_policy_screen import replay, run, summarize, validate_settings, verified_candidates
from tests import test_factor_screen_roundtrip as fixture_module


class PairedPolicyScreenTests(unittest.TestCase):
    def settings(self):
        code = Path(__file__).parents[1]
        return tuple(json.loads((code / "configs" / name).read_text()) for name in (
            "factor_training_screen.json", "factor_training_replication.json", "paired_policy_screen.json",
        ))

    def test_fixed_protocol_rejects_search_and_out_of_bound_inputs(self):
        prior, _, paired = self.settings()
        prior["source_hashes"] = screen.source_hashes()
        validate_settings(paired, prior)
        for key, value in (
            ("seeds", [42]), ("methods", []), ("configuration", {}),
            ("original_val_or_test_paths_opened", True), ("no_hyperparameter_search", False),
            ("both_objectives_removed_control", "best_removal"), ("novelty_established", True),
            ("primary_metric", "best_single_class"), ("every_class_or_policy_improvement_required", True),
            ("new_fits", 18), ("submission_allowed", True),
        ):
            changed = copy.deepcopy(paired)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed, prior)

    def test_three_conditions_and_exact_erm_control_roundtrip(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            prior_root, replication_root, candidate = (root / name for name in ("prior", "replication", "paired"))
            prior, extension, paired = self.settings()
            reduced = {**prior["configuration"], "max_epochs": 1, "patience": 1,
                       "hidden_dim": 8, "mlp_hidden_dim": 8, "embedding_dimension": 8, "feature_dim": 8,
                       "batch_size": 32, "knn_k": 2, "dropout": 0}
            protocols = []
            for name, settings in zip(("prior", "extension", "paired"), (prior, extension, paired)):
                settings["configuration"] = copy.deepcopy(reduced)
                path = root / (name + ".json")
                path.write_text(json.dumps(settings))
                protocols.append(path)
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = fixture_module.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                screen.run(*inputs, dataset, protocols[0], prior_root / dataset)
                screen.replay(*inputs, dataset, prior_root, prior_root / (dataset + "_audit"))
                replication.run(*inputs, dataset, protocols[1], prior_root, replication_root / dataset)
                replication.replay_extension(*inputs, dataset, replication_root, replication_root / (dataset + "_audit"))
                before = {str(path): path.read_bytes() for evidence in (prior_root, replication_root)
                          for path in evidence.rglob("*") if path.is_file()}
                run(*inputs, dataset, protocols[2], prior_root, replication_root, candidate / dataset)
                replay(*inputs, dataset, candidate, candidate / (dataset + "_audit"))
                self.assertEqual(before, {str(path): path.read_bytes() for evidence in (prior_root, replication_root)
                                         for path in evidence.rglob("*") if path.is_file()})
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocols[2], prior_root, replication_root, candidate / dataset)
            summarize(prior_root, replication_root, candidate, candidate / "summary")
            report = json.loads((candidate / "summary/development_summary.json").read_text())
            self.assertEqual(report["new_fits"], 54)
            self.assertEqual(report["both_removed_control"], "erm_mlp")
            self.assertFalse(report["submission_allowed"])
            for dataset in ("Twibot22", "Quadbot"):
                self.assertEqual(report["datasets"][dataset]["new_runs_verified"], 27)
                self.assertEqual(report["datasets"][dataset]["cached_controls_verified"], 27)
                primary = report["datasets"][dataset]["primary_pooled"]
                self.assertEqual(len(primary), 6)
                self.assertEqual(len(primary["paired_policy"]["macro_f1"]["values"]), 3)
                self.assertEqual(len(report["datasets"][dataset]["primary_pooled_full_minus_comparator"]), 5)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            checkpoint = next((candidate / "Quadbot").rglob("selected_model.pt"))
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verified_candidates(candidate, "Quadbot")


if __name__ == "__main__":
    unittest.main()
