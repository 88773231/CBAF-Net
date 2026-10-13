import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_botsscl_policy_control import (
    METHODS, SEEDS, expected_conditions, replay, run, summarize,
    validate_settings, verified,
)
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from analysis.run_flat_policy_screen import source_hashes as reference_sources
from src.botsscl_adapter import corrupt_features
from src.botsscl_policy_control import predict_control, train_control
from src.evidence_repair import flattened_inputs
from tests import test_factor_screen_roundtrip as fixture_module


class BotSSCLPolicyControlTests(unittest.TestCase):
    def values(self, prefix, classes=4):
        torch.manual_seed(1)
        rows = [{"sample_id": f"{prefix}{profile}_{label}",
                 "base_profile_id": f"{prefix}{profile}", "label": label}
                for profile in range(4) for label in range(classes)]
        count = len(rows)
        return {
            "metadata": rows, "labels": torch.tensor([row["label"] for row in rows]),
            "description": torch.randn(count, 8), "text": torch.randn(count, 5, 8),
            "action": torch.randn(count, 5, 15), "tabular": torch.randn(count, 24),
        }

    def settings(self):
        return json.loads((Path(__file__).parents[1] / "configs/botsscl_policy_control.json").read_text())

    def test_fitting_only_donors_and_training_only_projection(self):
        torch.set_num_threads(2)
        fitting, selection = self.values("f"), self.values("s")
        policies = {f"{prefix}{index}": f"policy{index % 2}"
                    for prefix in ("f", "s") for index in range(4)}
        settings = self.settings()
        config = {**settings["configuration"], "mlp_hidden_dim": 8, "max_epochs": 2,
                  "patience": 2, "batch_size": 12}
        candidates = flattened_inputs(fitting)
        seen = []

        def check_donors(anchor, donor, rate, generator):
            self.assertTrue(all(any(np.array_equal(row, known) for known in candidates)
                                for row in donor.numpy()))
            seen.append(len(anchor))
            return corrupt_features(anchor, donor, rate, generator)

        with patch("src.botsscl_policy_control.corrupt_features", side_effect=check_donors):
            model, fitted = train_control(fitting, selection, policies, METHODS[0], 4, 42,
                                         config, {**settings["pretraining"], "epochs": 2})
        self.assertTrue(seen)
        self.assertEqual(fitted["input_dimension"], 8 + 5 * 8 + 5 * 15 + 24)
        self.assertEqual(fitted["pretraining_optimizer_updates"], 4)
        self.assertEqual(fitted["pretraining_encoder_forward_evaluations"], 8)
        self.assertEqual(fitted["supervised_optimizer_updates"], 4)
        self.assertFalse(fitted["policy_or_profile_metadata_used_as_prediction_input"])
        self.assertGreater(fitted["training_parameter_count"], fitted["inference_parameter_count"])
        original = predict_control(model, selection, 8)
        with torch.no_grad():
            for parameter in model.projection.parameters():
                parameter.fill_(1000)
        np.testing.assert_array_equal(original, predict_control(model, selection, 8))

    def test_removal_has_no_pretraining_and_valid_probabilities(self):
        fitting, selection = self.values("f", 3), self.values("s", 3)
        policies = {f"{prefix}{index}": f"policy{index % 2}"
                    for prefix in ("f", "s") for index in range(4)}
        settings = self.settings()
        config = {**settings["configuration"], "mlp_hidden_dim": 8,
                  "max_epochs": 1, "patience": 1, "batch_size": 9}
        model, fitted = train_control(fitting, selection, policies, METHODS[1], 3, 42,
                                     config, settings["pretraining"])
        self.assertEqual(fitted["pretraining_history"], [])
        self.assertEqual(fitted["pretraining_optimizer_updates"], 0)
        self.assertIsNone(fitted["donor_population"])
        self.assertEqual(predict_control(model, selection, 8).shape, (12, 3))

    def test_overlap_and_missing_policy_rejected_before_model_creation(self):
        fitting, selection = self.values("f"), self.values("s")
        policies = {f"f{index}": f"policy{index % 2}" for index in range(4)}
        policies.update({f"s{index}": "unseen" for index in range(4)})
        with patch("src.botsscl_policy_control.CorruptionAdapter") as model:
            with self.assertRaisesRegex(ValueError, "crosses"):
                train_control(fitting, fitting, policies, METHODS[0], 4, 42, {}, {})
            with self.assertRaisesRegex(ValueError, "both fitting policies"):
                train_control(fitting, selection, policies, METHODS[0], 4, 42, {}, {})
            model.assert_not_called()

    def test_frozen_protocol_rejects_selection_or_reproduction_claim_changes(self):
        settings = self.settings()
        reference = {"configuration": settings["configuration"], "source_hashes": reference_sources(),
                     "original_val_paths_opened": False, "test_paths_opened": False,
                     "challenge_used_for_checkpoint_selection": False}
        validate_settings(settings, reference)
        self.assertEqual(len(expected_conditions()), 18)
        for key, value in (
            ("seeds", [42]), ("new_fits", 18), ("original_val_or_test_paths_opened", True),
            ("no_hyperparameter_search", False), ("independent_confirmation", True),
            ("adaptation_status", "faithful_reproduction"), ("submission_allowed", True),
            ("matched_input_not_matched_architecture_or_total_compute", False),
            ("pretraining", {"epochs": 60, "corruption_rate": .6, "temperature": 1.0}),
        ):
            changed = copy.deepcopy(settings)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed, reference)

    def test_complete_fit_replay_summary_and_tamper_checks_without_val_or_test(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            reference_root, control_root = root / "reference", root / "control"
            settings = self.settings()
            settings["configuration"].update({
                "max_epochs": 1, "patience": 1, "mlp_hidden_dim": 8,
                "knn_k": 2, "batch_size": 32,
            })
            protocol_path = root / "settings.json"
            protocol_path.write_text(json.dumps(settings))
            for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
                inputs = fixture_module.FactorScreenRoundtripTests().fixture(root / dataset, classes)
                material = prepare(*inputs, dataset)
                reference = {
                    "configuration": settings["configuration"], "source_hashes": reference_sources(),
                    "original_val_paths_opened": False, "test_paths_opened": False,
                    "challenge_used_for_checkpoint_selection": False,
                    "input_hashes": material["input_hashes"],
                    "prior_partition_hashes": material["prior_partition_hashes"],
                }
                (reference_root / dataset).mkdir(parents=True)
                (reference_root / dataset / "protocol.json").write_text(json.dumps(reference))
                for seed in SEEDS:
                    for policy in POLICIES:
                        _, _, normalization = fold(material, seed, policy, settings["configuration"]["knn_k"])
                        folder_path = reference_root / dataset / "main" / f"init{seed}_fold{seed}" / policy
                        folder_path.mkdir(parents=True)
                        (folder_path / "normalization.json").write_text(json.dumps(normalization))
                preserved = {str(path): path.read_bytes() for path in reference_root.rglob("*")
                             if path.is_file()}
                run(*inputs, dataset, protocol_path, reference_root, control_root / dataset)
                replay(*inputs, dataset, control_root, control_root / (dataset + "_audit"))
                self.assertEqual(preserved, {str(path): path.read_bytes() for path in reference_root.rglob("*")
                                             if path.is_file()})
                with self.assertRaises(FileExistsError):
                    run(*inputs, dataset, protocol_path, reference_root, control_root / dataset)
            summarize(control_root, control_root / "summary")
            report = json.loads((control_root / "summary/development_summary.json").read_text())
            self.assertFalse(report["submission_allowed"])
            self.assertFalse(report["independent_confirmation"])
            self.assertEqual(report["new_fits"], 36)
            for dataset in ("Twibot22", "Quadbot"):
                current = report["datasets"][dataset]
                self.assertEqual(current["runs_verified"], 18)
                self.assertEqual(len(current["complete_runs"]), 18)
                self.assertEqual(len(current["primary_pooled"]), 2)
                for score in current["primary_pooled"].values():
                    self.assertEqual(score["macro_f1"]["seeds"], [42, 43, 44])
            self.assertFalse(any(path.name.startswith(("val_", "test_")) for path in root.rglob("*")))
            checkpoint = next((control_root / "Quadbot").rglob("selected_model.pt"))
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verified(control_root, "Quadbot")


if __name__ == "__main__":
    unittest.main()
