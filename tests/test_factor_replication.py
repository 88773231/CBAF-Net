import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from analysis.replicate_factor_training_screen import (
    describe, replay_extension, run, source_hashes, summarize, validate_recipe, verified_runs,
)
from analysis.run_factor_training_screen import replay as old_replay, run as old_run
from tests import test_factor_screen_roundtrip as screen_fixture


class FactorReplicationTests(unittest.TestCase):
    def settings(self):
        code = Path(__file__).parents[1]
        prior = json.loads((code / "configs/factor_training_screen.json").read_text())
        prior["source_hashes"] = source_hashes()
        extension = json.loads((code / "configs/factor_training_replication.json").read_text())
        return prior, extension

    def test_fixed_recipe_rejects_seeds_methods_sources_and_weights(self):
        prior, extension = self.settings()
        validate_recipe(extension, prior)
        for key, value in (
            ("seeds", [42, 43, 44]), ("reused_seeds", []),
            ("methods", extension["methods"][:-1]), ("configuration", {}),
            ("submission_allowed", True),
        ):
            changed = copy.deepcopy(extension)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_recipe(changed, prior)
        changed = copy.deepcopy(prior)
        changed["source_hashes"] = {}
        with self.assertRaisesRegex(ValueError, "implementation"):
            validate_recipe(extension, changed)

    def test_seed_sd_is_sample_sd_and_all_seeds_are_required(self):
        result = describe([0.7, 0.8, 0.9])
        self.assertAlmostEqual(result["mean"], 0.8)
        self.assertAlmostEqual(result["seed_sd"], 0.1)
        for values in ([0.7], [0.7, 0.8], [0.7, 0.8, float("nan")]):
            with self.assertRaises(ValueError):
                describe(values)

    def test_extension_roundtrip_preserves_prior_and_requires_complete_coverage(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            prior_root, extension_root = root / "prior", root / "extension"
            prior, extension = self.settings()
            del prior["source_hashes"]
            prior["configuration"].update({
                "max_epochs": 1, "patience": 1, "hidden_dim": 8,
                "mlp_hidden_dim": 8, "embedding_dimension": 8, "feature_dim": 8,
                "batch_size": 32, "knn_k": 2, "dropout": 0,
            })
            extension["configuration"] = copy.deepcopy(prior["configuration"])
            protocol, new_protocol = root / "prior.json", root / "extension.json"
            protocol.write_text(json.dumps(prior))
            new_protocol.write_text(json.dumps(extension))
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = screen_fixture.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                old_run(*inputs, dataset, protocol, prior_root / dataset)
                old_replay(*inputs, dataset, prior_root, prior_root / (dataset + "_audit"))
                before = {str(path): path.read_bytes() for path in prior_root.rglob("*") if path.is_file()}
                run(*inputs, dataset, new_protocol, prior_root, extension_root / dataset)
                replay_extension(*inputs, dataset, extension_root, extension_root / (dataset + "_audit"))
                self.assertEqual(before, {
                    str(path): path.read_bytes() for path in prior_root.rglob("*") if path.is_file()
                })
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, new_protocol, prior_root, extension_root / dataset)
            summarize(prior_root, extension_root, extension_root / "summary")
            report = json.loads((extension_root / "summary/development_summary.json").read_text())
            self.assertEqual(report["total_fits"], 108)
            self.assertFalse(report["submission_allowed"])
            for dataset in ("Twibot22", "Quadbot"):
                self.assertEqual(report["datasets"][dataset]["runs_verified"], 54)
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            with self.assertRaises(FileExistsError):
                summarize(prior_root, extension_root, extension_root / "summary")
            audit_path = extension_root / "Quadbot_audit/audit.json"
            audit = json.loads(audit_path.read_text())
            audit["checkpoint_replays"].pop()
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verified_runs(extension_root, "Quadbot", (43, 44))


if __name__ == "__main__":
    unittest.main()
