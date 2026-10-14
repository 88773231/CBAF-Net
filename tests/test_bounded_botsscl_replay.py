import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from threadpoolctl import threadpool_limits

from analysis.bounded_botsscl_replay import SingleFoldCache, replay as bounded_replay
from analysis.run_botsscl_policy_control import SEEDS, replay, run
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from analysis.run_flat_policy_screen import source_hashes as reference_sources
from tests import test_factor_screen_roundtrip as fixtures


class BoundedBotSSCLReplayTests(unittest.TestCase):
    def test_cache_retains_only_latest_fold(self):
        cache = SingleFoldCache()
        cache[(42, "burst")] = "first"
        cache[(42, "cyclic")] = "second"
        self.assertEqual(cache, {(42, "cyclic"): "second"})

    def test_bounded_and_original_replays_match_and_reject_tampering(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            settings = json.loads(
                (Path(run.__code__.co_filename).resolve().parents[1]
                 / "configs/botsscl_policy_control.json").read_text()
            )
            settings["configuration"].update({
                "max_epochs": 1, "patience": 1, "mlp_hidden_dim": 8,
                "knn_k": 2, "batch_size": 32,
            })
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(settings))
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = fixtures.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                material = prepare(*inputs, dataset)
                reference, control = root / "reference", root / "control"
                (reference / dataset).mkdir(parents=True)
                (reference / dataset / "protocol.json").write_text(json.dumps({
                    "configuration": settings["configuration"],
                    "source_hashes": reference_sources(),
                    "original_val_paths_opened": False, "test_paths_opened": False,
                    "challenge_used_for_checkpoint_selection": False,
                    "input_hashes": material["input_hashes"],
                    "prior_partition_hashes": material["prior_partition_hashes"],
                }))
                for seed in SEEDS:
                    for policy in POLICIES:
                        _, _, normalization = fold(material, seed, policy, settings["configuration"]["knn_k"])
                        path = reference / dataset / "main" / f"init{seed}_fold{seed}" / policy
                        path.mkdir(parents=True)
                        (path / "normalization.json").write_text(json.dumps(normalization))
                run(*inputs, dataset, protocol, reference, control / dataset)
                original_out, bounded_out = root / (dataset + "_original"), root / (dataset + "_bounded")
                replay(*inputs, dataset, control, original_out)
                receipt = bounded_replay(*inputs, dataset, control, bounded_out)
                self.assertEqual((original_out / "audit.json").read_bytes(),
                                 (bounded_out / "audit.json").read_bytes())
                self.assertEqual(receipt["cache_initializers_replaced"], 1)
                self.assertFalse(receipt["training_rerun"])
                checkpoint = next((control / dataset).rglob("selected_model.pt"))
                checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "changed"):
                    bounded_replay(*inputs, dataset, control, root / (dataset + "_tamper"))


if __name__ == "__main__":
    unittest.main()
