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
from analysis.run_encoder_objective_diagnostic import load_known_fold, run, validate_settings
from analysis.run_factor_training_screen import POLICIES, fold, prepare
from src.botsscl_policy_control import predict_control, train_control
from src.encoder_objective_diagnostic import CONDITIONS, state_digest, train_objective
from src.provenance import sha256_file
from tests import test_botsscl_policy_control as control_fixture
from tests import test_factor_screen_roundtrip as fold_fixture


class EncoderObjectiveDiagnosticTests(unittest.TestCase):
    def settings(self):
        return json.loads((Path(__file__).parents[1] /
                           "configs/encoder_objective_diagnostic.json").read_text())

    def populations(self):
        fixture = control_fixture.BotSSCLPolicyControlTests()
        fitting, selection = fixture.values("f"), fixture.values("s")
        policies = {f"{prefix}{index}": f"policy{index % 2}"
                    for prefix in ("f", "s") for index in range(4)}
        config = {
            **fixture.settings()["configuration"], "mlp_hidden_dim": 8,
            "max_epochs": 2, "patience": 2, "batch_size": 12,
        }
        return fitting, selection, policies, config

    def test_fixed_access_and_run_count(self):
        settings = self.settings()
        validate_settings(settings)
        for key, value in (("dataset", "Twibot22"), ("seeds", [42, 43, 44]),
                           ("new_fits", 18), ("challenge_forward_or_scoring", True),
                           ("submission_allowed", True)):
            changed = copy.deepcopy(settings)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_settings(changed)

    def test_same_initialization_and_control_reference_equivalence(self):
        torch.set_num_threads(2)
        fitting, selection, policies, config = self.populations()
        regularization = self.settings()["regularization"]
        models, records = {}, {}
        with threadpool_limits(limits=2):
            for condition in CONDITIONS:
                models[condition], records[condition] = train_objective(
                    fitting, selection, policies, condition, 4, 42, config, regularization,
                )
            reference, _ = train_control(
                fitting, selection, policies, "botsscl_supervised_only", 4, 42, config,
                {"epochs": 30, "corruption_rate": .6, "temperature": 1.0},
            )
        self.assertEqual(len({row["initialization_digest"] for row in records.values()}), 1)
        np.testing.assert_array_equal(
            predict_control(reference, selection, 12),
            predict_control(models["supervised_reference"], selection, 12),
        )
        for condition, row in records.items():
            self.assertFalse(row["challenge_forward_or_scoring"])
            self.assertFalse(row["pretraining_used"])
            self.assertTrue(row["initial_probe"]["parameters_and_rng_unchanged"])
            self.assertTrue(row["selected_probe"]["parameters_and_rng_unchanged"])
            factor = 2 if condition.startswith("sam") else 1
            self.assertEqual(row["gradient_evaluations"], factor * row["optimizer_updates"])
            self.assertFalse(any(p.requires_grad for p in models[condition].projection.parameters()))
        self.assertEqual(state_digest(reference), state_digest(models["supervised_reference"]))

    def test_invalid_population_rejected_before_model_creation(self):
        fitting, selection, policies, config = self.populations()
        with patch("src.encoder_objective_diagnostic.CorruptionAdapter") as constructor:
            with self.assertRaises(ValueError):
                train_objective(fitting, fitting, policies, CONDITIONS[0], 4, 42,
                                config, self.settings()["regularization"])
            with self.assertRaises(ValueError):
                train_objective(fitting, selection, policies, CONDITIONS[0], 3, 42,
                                config, self.settings()["regularization"])
            constructor.assert_not_called()

    def make_reference(self, root):
        inputs = fold_fixture.FactorScreenRoundtripTests().fixture(root / "inputs", 4)
        material = prepare(*inputs, "Quadbot")
        config = {
            **control_fixture.BotSSCLPolicyControlTests().settings()["configuration"],
            "mlp_hidden_dim": 8, "max_epochs": 1, "patience": 1, "batch_size": 32, "knn_k": 2,
        }
        reference_root = root / "reference"
        folder = reference_root / "Quadbot"
        folder.mkdir(parents=True)
        protocol = {
            "configuration": config, "source_hashes": control_sources(),
            "input_hashes": material["input_hashes"], "original_val_paths_opened": False,
            "test_paths_opened": False,
        }
        protocol_path = folder / "protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        for policy in POLICIES:
            values, _, normalization = fold(material, 42, policy, config["knn_k"])
            destination = folder / "seed42" / policy / "botsscl_supervised_only"
            destination.mkdir(parents=True)
            (destination.parent / "normalization.json").write_text(json.dumps(normalization))
            model, fitted = train_control(
                values[0], values[1], material["policies"], "botsscl_supervised_only",
                4, 42, config, {"epochs": 30, "corruption_rate": .6, "temperature": 1.0},
            )
            identity = {"dataset": "Quadbot", "seed": 42, "heldout_policy": policy,
                        "method": "botsscl_supervised_only"}
            checkpoint = destination / "selected_model.pt"
            torch.save({
                **identity, "state_dict": model.state_dict(), "configuration": config,
                "normalization": normalization, "protocol_sha256": sha256_file(protocol_path),
                "input_dimension": fitted["input_dimension"],
            }, checkpoint)
            (destination / "metrics.json").write_text(json.dumps({
                **identity, "fit": fitted, "checkpoint_sha256": sha256_file(checkpoint),
            }))
        return inputs, reference_root

    def test_roundtrip_only_known_policy_rows_and_reuses_three_references(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference = self.make_reference(root)
            settings_path = root / "settings.json"
            settings_path.write_text(json.dumps(self.settings()))
            unchanged = {str(p): p.read_bytes() for p in reference.rglob("*") if p.is_file()}
            run(inputs[0], inputs[1], reference, settings_path, root / "diagnostic")
            report = json.loads((root / "diagnostic" / "summary.json").read_text())
            self.assertEqual(report["new_fits"], 9)
            self.assertEqual(len(report["records"]), 12)
            self.assertFalse(report["challenge_forward_or_scoring"])
            self.assertFalse(report["independent_confirmation"])
            self.assertEqual(unchanged, {str(p): p.read_bytes() for p in reference.rglob("*") if p.is_file()})
            for policy in POLICIES:
                populations, _, _, _, _ = load_known_fold(inputs[0], inputs[1], reference, policy)
                allowed = {row["sample_id"] for row in populations[1]["metadata"]}
                for path in (root / "diagnostic" / policy).rglob("inner_selection_predictions.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual({row["sample_id"] for row in rows}, allowed)
            self.assertFalse(any("challenge_predictions" in p.name
                                 for p in (root / "diagnostic").rglob("*")))
            with self.assertRaises(FileExistsError):
                run(inputs[0], inputs[1], reference, settings_path, root / "diagnostic")

    def test_tampered_training_inputs_rejected_before_fit(self):
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2):
            root = Path(folder)
            inputs, reference = self.make_reference(root)
            path = inputs[1] / "train_steps.pt"
            path.write_bytes(path.read_bytes() + b"changed")
            with patch("analysis.run_encoder_objective_diagnostic.train_objective") as fitting:
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    load_known_fold(inputs[0], inputs[1], reference, POLICIES[0])
                fitting.assert_not_called()


if __name__ == "__main__":
    unittest.main()
