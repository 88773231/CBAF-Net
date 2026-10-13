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
from analysis.run_conditional_screen import replay, run, summarize, validate_settings, verified_candidates
from tests import test_factor_screen_roundtrip as fixture_module


class ConditionalScreenTests(unittest.TestCase):
    def settings(self):
        code = Path(__file__).parents[1]
        return tuple(json.loads((code / "configs" / name).read_text()) for name in (
            "factor_training_screen.json", "factor_training_replication.json", "conditional_screen.json",
        ))

    def test_fixed_recipe_rejects_search_and_out_of_bound_inputs(self):
        prior, _, conditional = self.settings()
        prior["source_hashes"] = screen.source_hashes()
        validate_settings(conditional, prior)
        for key, value in (
            ("seeds", [42]), ("methods", conditional["methods"][:-1]),
            ("configuration", {}), ("original_val_or_test_paths_opened", True),
            ("no_hyperparameter_search", False), ("novelty_established", True),
            ("reused_controls", []), ("new_fits", 36), ("submission_allowed", True),
        ):
            changed = copy.deepcopy(conditional)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed, prior)

    def test_complete_2x2_roundtrip_without_val_test_or_prior_modification(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            prior_root, replication_root, candidate_root = (
                root / name for name in ("prior", "replication", "candidate")
            )
            prior, extension, conditional = self.settings()
            reduced = {
                **prior["configuration"], "max_epochs": 1, "patience": 1,
                "hidden_dim": 8, "mlp_hidden_dim": 8, "embedding_dimension": 8,
                "feature_dim": 8, "batch_size": 32, "knn_k": 2, "dropout": 0,
            }
            protocols = []
            for name, settings in zip(("prior", "extension", "conditional"), (prior, extension, conditional)):
                settings["configuration"] = copy.deepcopy(reduced)
                path = root / (name + ".json")
                path.write_text(json.dumps(settings))
                protocols.append(path)
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = fixture_module.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                screen.run(*inputs, dataset, protocols[0], prior_root / dataset)
                screen.replay(*inputs, dataset, prior_root, prior_root / (dataset + "_audit"))
                replication.run(*inputs, dataset, protocols[1], prior_root, replication_root / dataset)
                replication.replay_extension(
                    *inputs, dataset, replication_root, replication_root / (dataset + "_audit"),
                )
                preserved = {
                    str(path): path.read_bytes()
                    for evidence in (prior_root, replication_root)
                    for path in evidence.rglob("*") if path.is_file()
                }
                run(*inputs, dataset, protocols[2], prior_root, replication_root, candidate_root / dataset)
                replay(*inputs, dataset, candidate_root, candidate_root / (dataset + "_audit"))
                self.assertEqual(preserved, {
                    str(path): path.read_bytes()
                    for evidence in (prior_root, replication_root)
                    for path in evidence.rglob("*") if path.is_file()
                })
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocols[2], prior_root, replication_root, candidate_root / dataset)
                with self.assertRaises(FileExistsError):
                    replay(*inputs, dataset, candidate_root, candidate_root / (dataset + "_audit"))
            summarize(prior_root, replication_root, candidate_root, candidate_root / "summary")
            result = json.loads((candidate_root / "summary/development_summary.json").read_text())
            self.assertFalse(result["submission_allowed"])
            self.assertEqual(result["new_fits"], 72)
            self.assertEqual(result["reused_control_fits"], 54)
            for dataset in ("Twibot22", "Quadbot"):
                self.assertEqual(result["datasets"][dataset]["new_runs_verified"], 36)
                self.assertEqual(result["datasets"][dataset]["cached_controls_verified"], 27)
                for policy in screen.POLICIES:
                    self.assertEqual(len(result["datasets"][dataset]["tables"][policy]), 7)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            checkpoint = next((candidate_root / "Quadbot").rglob("selected_model.pt"))
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verified_candidates(candidate_root, "Quadbot")


if __name__ == "__main__":
    unittest.main()
