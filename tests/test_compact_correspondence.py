import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_botsscl_policy_control import source_hashes as control_sources
from analysis.run_compact_correspondence_stability import (
    CONDITIONS, CONTROL_METHODS, SEEDS, cached_control, decision, run, validate_settings,
)
from analysis.run_encoder_objective_diagnostic import load_known_fold
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from src.botsscl_policy_control import train_control
from src.compact_correspondence import (
    CONDITIONS as COMPACT, make_model, predict, shared_state_digest, train,
)
from src.evidence_repair import model_inputs, seed_everything
from src.provenance import sha256_file
from src.trajectory_correspondence import make_model as legacy_model
from tests import test_factor_screen_roundtrip as fold_fixture
from tests import test_trajectory_correspondence as trajectory_fixture


class CompactCorrespondenceTests(unittest.TestCase):
    def settings(self):
        return json.loads((Path(__file__).parents[1] /
                           "configs/compact_correspondence_stability.json").read_text())

    def populations(self):
        return trajectory_fixture.TrajectoryCorrespondenceTests().populations()

    def test_compaction_equivalence_and_unchanged_rng(self):
        fitting, _, _, config = self.populations()
        for seed in SEEDS:
            for compact, legacy in zip(COMPACT, ("without_transition", "without_both")):
                seed_everything(seed)
                original = legacy_model(fitting, config, legacy)
                old_rng = torch.get_rng_state().clone()
                seed_everything(seed)
                model = make_model(fitting, config, compact)
                self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                model.eval()
                original.eval()
                torch.testing.assert_close(
                    model(*model_inputs(fitting, "mlp_matched"))["logits"],
                    original(*model_inputs(fitting, "mlp_matched"))["logits"],
                    atol=1e-6, rtol=1e-6,
                )
                self.assertEqual(model.readout[0].in_features, config["block_dim"] * 7)
                self.assertEqual(model.block_encoder[0].in_features,
                                 config["block_dim"] * (4 if compact == COMPACT[0] else 2))
                self.assertIsNone(model.transition)
                old_count = sum(p.numel() for p in original.parameters())
                new_count = sum(p.numel() for p in model.parameters())
                pruned = config["block_dim"] * config["readout_dim"]
                if compact == COMPACT[1]:
                    pruned += 2 * config["block_dim"] ** 2
                self.assertEqual(old_count - new_count, pruned)

    def test_shared_active_initialization_and_all_parameters_in_gradient_path(self):
        fitting, _, _, config = self.populations()
        for seed in SEEDS:
            digests = []
            for condition in COMPACT:
                seed_everything(seed)
                model = make_model(fitting, config, condition)
                digests.append(shared_state_digest(model))
                torch.nn.functional.cross_entropy(
                    model(*model_inputs(fitting, "mlp_matched"))["logits"], fitting["labels"],
                ).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
                    if name in ("block_encoder.0.weight", "readout.0.weight"):
                        self.assertTrue((parameter.grad.abs().sum(0) > 0).all(), name)
            self.assertEqual(len(set(digests)), 1)

    def test_checkpoint_roundtrip_and_row_isolation(self):
        fitting, selection, policies, config = self.populations()
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2):
            for condition in COMPACT:
                model, record = train(fitting, selection, policies, condition, 43, config)
                self.assertEqual(record["inactive_serialized_weight_count"], 0)
                checkpoint = Path(folder) / (condition + ".pt")
                torch.save(model.state_dict(), checkpoint)
                replay = make_model(fitting, config, condition)
                replay.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                predicted = predict(model, selection, 12)
                np.testing.assert_array_equal(predicted, predict(replay, selection, 12))
                np.testing.assert_allclose(predicted, predict(model, selection, 1), atol=1e-6)
                copied = copy.deepcopy(selection)
                copied["metadata"] = [{**row, "base_profile_id": "ignored"} for row in copied["metadata"]]
                np.testing.assert_array_equal(predicted, predict(model, copied, 12))

    def test_invalid_settings_and_populations_rejected_before_creation(self):
        settings = self.settings()
        validate_settings(settings)
        for key, value in (
            ("partition_seed", 43), ("training_seeds", [42]), ("new_fits", 36),
            ("challenge_forward_or_scoring", True), ("submission_allowed", True),
        ):
            with self.assertRaises(ValueError):
                validate_settings({**settings, key: value})
        fitting, selection, policies, config = self.populations()
        with patch("src.compact_correspondence.CompactCorrespondenceModel") as constructor:
            for key, value in (("batch_size", 3), ("block_dim", 1.5),
                               ("learning_rate", float("nan")), ("dropout", 1)):
                with self.assertRaises(ValueError):
                    train(fitting, selection, policies, COMPACT[0], 42, {**config, key: value})
            with self.assertRaises(ValueError):
                train(fitting, fitting, policies, COMPACT[0], 42, config)
            with self.assertRaises(ValueError):
                train(fitting, selection, policies, "unknown", 42, config)
            constructor.assert_not_called()

    def test_seed_mean_gate_not_independent_fold_significance(self):
        records = [
            {"seed": seed, "policy": policy, "condition": condition,
             "inner_selection": {"all_profiles": {
                 "macro_f1": .91 if condition == COMPACT[0] else .90,
             }}}
            for seed in SEEDS for policy in POLICIES for condition in CONDITIONS
        ]
        result = decision(records)
        self.assertTrue(result["development_continuation_gate_passed"])
        self.assertFalse(result["submission_allowed"])
        self.assertFalse(result["independent_confirmation"])
        for row in records:
            if row["condition"] == "botsscl_mechanism":
                row["inner_selection"]["all_profiles"]["macro_f1"] = .92
        self.assertFalse(decision(records)["development_continuation_gate_passed"])
        with self.assertRaises(ValueError):
            decision(records[:-1])
        with self.assertRaises(ValueError):
            decision(records + [records[0]])

    def make_reference(self, root):
        inputs = fold_fixture.FactorScreenRoundtripTests().fixture(root / "inputs", 4)
        material = prepare(*inputs, "Quadbot")
        source_config = json.loads((Path(__file__).parents[1] /
                                    "configs/botsscl_policy_control.json").read_text())
        config = {**source_config["configuration"], "max_epochs": 1, "patience": 1,
                  "batch_size": 32, "knn_k": 2}
        pretraining = {**source_config["pretraining"], "epochs": 1}
        reference_root = root / "reference"
        folder = reference_root / "Quadbot"
        folder.mkdir(parents=True)
        protocol = folder / "protocol.json"
        protocol.write_text(json.dumps({
            "configuration": config, "pretraining": pretraining, "source_hashes": control_sources(),
            "input_hashes": material["input_hashes"], "original_val_paths_opened": False,
            "test_paths_opened": False,
        }))
        for policy in POLICIES:
            values, _, normalization = fold(material, 42, policy, config["knn_k"])
            for method in CONTROL_METHODS.values():
                destination = folder / "seed42" / policy / method
                destination.mkdir(parents=True)
                (destination.parent / "normalization.json").write_text(json.dumps(normalization))
                model, fitted = train_control(
                    *values[:2], material["policies"], method, 4, 42, config, pretraining,
                )
                identity = {"dataset": "Quadbot", "seed": 42, "heldout_policy": policy, "method": method}
                checkpoint = destination / "selected_model.pt"
                torch.save({
                    **identity, "state_dict": model.state_dict(), "configuration": config,
                    "pretraining": pretraining, "normalization": normalization,
                    "protocol_sha256": sha256_file(protocol), "input_dimension": fitted["input_dimension"],
                }, checkpoint)
                (destination / "metrics.json").write_text(json.dumps({
                    **identity, "fit": fitted, "checkpoint_sha256": sha256_file(checkpoint),
                }))
        settings = self.settings()
        settings["configuration"].update({
            "block_dim": 8, "readout_dim": 8, "max_epochs": 1, "patience": 1, "batch_size": 32,
        })
        settings["pretraining"] = pretraining
        return inputs, reference_root, settings

    def test_36_cell_roundtrip_with_fixed_partitions_and_reference_unchanged(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference, settings = self.make_reference(root)
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(settings))
            original = {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()}
            output = root / "screen"
            with patch("analysis.run_compact_correspondence_stability.validate_settings"):
                run(*inputs[:2], reference, protocol, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["new_fits"], 30)
            self.assertEqual(summary["reused_reference_checkpoints"], 6)
            self.assertEqual(summary["checkpoint_replays"], 36)
            self.assertEqual(len(summary["records"]), 36)
            self.assertFalse(summary["submission_allowed"])
            self.assertEqual(original, {str(path): path.read_bytes()
                                        for path in reference.rglob("*") if path.is_file()})
            for policy in POLICIES:
                values, _, _, _, _ = load_known_fold(*inputs[:2], reference, policy)
                allowed = [(row["sample_id"], row["base_profile_id"], row["label"])
                           for row in values[1]["metadata"]]
                for path in (output / policy).rglob("inner_selection_predictions.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual([(row["sample_id"], row["base_profile_id"], row["label"])
                                      for row in rows], allowed)
            self.assertFalse(any(path.name.startswith(("val_", "test_", "challenge_predictions"))
                                 for path in output.rglob("*")))
            with self.assertRaises(FileExistsError):
                run(*inputs[:2], reference, protocol, output)

    def test_cached_control_identity_tampering_rejected(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2):
            inputs, reference, settings = self.make_reference(Path(folder))
            values, _, normalization, frozen, _ = load_known_fold(*inputs[:2], reference, POLICIES[0])
            path = reference / "Quadbot/seed42" / POLICIES[0] / "botsscl_flattened/metrics.json"
            record = json.loads(path.read_text())
            record["seed"] = 43
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                cached_control(reference, POLICIES[0], "botsscl_mechanism", normalization,
                               frozen["configuration"], values[0], settings["pretraining"])


if __name__ == "__main__":
    unittest.main()
