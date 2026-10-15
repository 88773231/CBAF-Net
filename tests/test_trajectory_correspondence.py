import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_encoder_objective_diagnostic import load_known_fold
from analysis.run_factor_training_screen import POLICIES
from analysis.run_trajectory_correspondence_screen import (
    run, validate_settings, verified_checkpoint,
)
from src.evidence_repair import model_inputs, seed_everything
from src.trajectory_correspondence import (
    CONDITIONS, AdjacentTransition, HistoricalCorrespondence, make_model,
    predict, shared_state_digest, train,
)
from tests import test_encoder_objective_diagnostic as diagnostic_fixture


class TrajectoryCorrespondenceTests(unittest.TestCase):
    def settings(self):
        return json.loads((Path(__file__).parents[1] /
                           "configs/trajectory_correspondence_screen.json").read_text())

    def populations(self):
        fitting, selection, policies, _ = (
            diagnostic_fixture.EncoderObjectiveDiagnosticTests().populations()
        )
        config = {
            **self.settings()["configuration"], "block_dim": 8, "readout_dim": 8,
            "max_epochs": 2, "patience": 2, "batch_size": 12,
        }
        return fitting, selection, policies, config

    def test_historical_mask_and_row_isolation(self):
        seed_everything(42)
        module = HistoricalCorrespondence(8, 5).eval()
        text, action = torch.randn(3, 5, 8), torch.randn(3, 5, 8)
        original = module(text, action)
        for key in ("text_to_action", "action_to_text"):
            matrix = original[key]
            torch.testing.assert_close(matrix.sum(-1), torch.ones(3, 5))
            self.assertTrue(torch.isfinite(matrix).all())
            self.assertTrue(torch.equal(matrix.triu(1), torch.zeros_like(matrix)))
        changed_text, changed_action = text.clone(), action.clone()
        changed_text[:, 3:] += 10
        changed_action[:, 3:] -= 10
        changed = module(changed_text, changed_action)
        for key in original:
            torch.testing.assert_close(original[key][:, :3], changed[key][:, :3])
        permutation = [2, 0, 1]
        permuted = module(text[permutation], action[permutation])
        for key in original:
            torch.testing.assert_close(original[key][permutation], permuted[key])
            torch.testing.assert_close(original[key][:1], module(text[:1], action[:1])[key])

    def test_full_backward_and_prediction_row_isolation(self):
        fitting, _, _, config = self.populations()
        seed_everything(42)
        model = make_model(fitting, config, CONDITIONS[0])
        output = model(*model_inputs(fitting, "mlp_matched"))
        self.assertEqual(output["logits"].shape, (16, 4))
        self.assertEqual(output["blocks"].shape, (16, 5, 8))
        torch.nn.functional.cross_entropy(output["logits"], fitting["labels"]).backward()
        for module in (model.correspondence, model.transition):
            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for parameter in module.parameters()))
        all_rows = predict(model, fitting, 16)
        np.testing.assert_allclose(all_rows.sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(all_rows, predict(model, fitting, 1), atol=1e-6)
        copied = copy.deepcopy(fitting)
        copied["metadata"] = [{**row, "sample_id": "changed", "base_profile_id": "ignored"}
                              for row in copied["metadata"]]
        np.testing.assert_array_equal(all_rows, predict(model, copied, 16))

    def test_common_initialization_and_inactive_paths(self):
        fitting, _, _, config = self.populations()
        digests = []
        for condition in CONDITIONS:
            seed_everything(42)
            model = make_model(fitting, config, condition)
            digests.append(shared_state_digest(model))
            output = model(*model_inputs(fitting, "mlp_matched"))
            output["logits"].sum().backward()
            without_correspondence = condition in ("without_correspondence", "without_both")
            without_transition = condition in ("without_transition", "without_both")
            self.assertEqual(model.correspondence is None, without_correspondence)
            self.assertEqual(model.transition is None, without_transition)
            if without_correspondence:
                self.assertIsNone(output["correspondence"])
                self.assertEqual(model.block_encoder[0].weight.grad[:, 16:].abs().sum(), 0)
                self.assertFalse(any(name.startswith("correspondence.") for name in model.state_dict()))
            if without_transition:
                self.assertEqual(output["transition_representation"].abs().sum(), 0)
                self.assertEqual(model.readout[0].weight.grad[:, -8:].abs().sum(), 0)
                self.assertFalse(any(name.startswith("transition.") for name in model.state_dict()))
        self.assertEqual(len(set(digests)), 1)

    def test_training_counts_and_probabilities_for_all_conditions(self):
        torch.set_num_threads(2)
        fitting, selection, policies, config = self.populations()
        digests = []
        with threadpool_limits(limits=2):
            for condition in CONDITIONS:
                model, record = train(fitting, selection, policies, condition, 42, config)
                digests.append(record["shared_initialization_digest"])
                self.assertEqual(record["optimizer_updates"], 4)
                self.assertEqual(record["parameter_count"], sum(p.numel() for p in model.parameters()))
                self.assertFalse(record["sam_or_contrast_used"])
                self.assertFalse(record["submission_allowed"])
                self.assertEqual(predict(model, selection, 12).shape, (16, 4))
        self.assertEqual(len(set(digests)), 1)

    def test_invalid_inputs_and_budget_rejected_before_model_creation(self):
        fitting, selection, policies, config = self.populations()
        with patch("src.trajectory_correspondence.TrajectoryCorrespondenceModel") as constructor:
            for key, value in (
                ("batch_size", 3), ("max_epochs", 0), ("block_dim", 3.5),
                ("learning_rate", float("inf")), ("weight_decay", -1), ("dropout", 1),
            ):
                with self.assertRaises(ValueError):
                    train(fitting, selection, policies, CONDITIONS[0], 42, {**config, key: value})
            with self.assertRaises(ValueError):
                train(fitting, fitting, policies, CONDITIONS[0], 42, config)
            with self.assertRaises(ValueError):
                train(fitting, selection, policies, "unknown", 42, config)
            for key, value in (
                ("text", torch.zeros(16, 4, 8)), ("tabular", torch.zeros(16, 23)),
                ("description", torch.full((16, 8), float("nan"))),
                ("action", torch.zeros(16, 5, 16)),
            ):
                with self.assertRaises(ValueError):
                    train(fitting, {**selection, key: value}, policies, CONDITIONS[0], 42, config)
            constructor.assert_not_called()
        with self.assertRaises(ValueError):
            AdjacentTransition(8)(torch.zeros(2, 1, 8))
        with self.assertRaises(ValueError):
            AdjacentTransition(8)(torch.zeros(2, 5, 7))

    def test_frozen_settings_and_checkpoint_identity(self):
        settings = self.settings()
        validate_settings(settings)
        for key, value in (("dataset", "Twibot22"), ("new_fits", 24),
                           ("challenge_forward_or_scoring", True),
                           ("algorithmic_novelty_established", True)):
            with self.assertRaises(ValueError):
                validate_settings({**settings, key: value})
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "model.pt"
            identity = {"condition": CONDITIONS[0], "seed": 42}
            torch.save({"state_dict": {"weight": torch.zeros(1)}, **identity}, checkpoint)
            verified_checkpoint(checkpoint, identity)
            with self.assertRaisesRegex(ValueError, "envelope"):
                verified_checkpoint(checkpoint, {**identity, "seed": 43})
            torch.save({"state_dict": {}, **identity, "extra": True}, checkpoint)
            with self.assertRaisesRegex(ValueError, "envelope"):
                verified_checkpoint(checkpoint, identity)

    def test_twelve_cell_roundtrip_only_known_rows_and_reference_unchanged(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference = (
                diagnostic_fixture.EncoderObjectiveDiagnosticTests().make_reference(root)
            )
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(self.settings()))
            original = {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()}
            output = root / "screen"
            run(*inputs[:2], reference, protocol, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["new_fits"], 12)
            self.assertEqual(summary["checkpoint_replays"], 12)
            self.assertEqual(len(summary["records"]), 12)
            self.assertTrue(summary["reference_files_unchanged"])
            self.assertFalse(summary["challenge_forward_or_scoring"])
            self.assertFalse(summary["submission_allowed"])
            self.assertEqual(original, {str(path): path.read_bytes() for path in reference.rglob("*")
                                        if path.is_file()})
            for policy in POLICIES:
                populations, _, _, _, _ = load_known_fold(*inputs[:2], reference, policy)
                allowed = {row["sample_id"] for row in populations[1]["metadata"]}
                for path in (output / policy).rglob("inner_selection_predictions.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual({row["sample_id"] for row in rows}, allowed)
            self.assertFalse(any(path.name.startswith(("val_", "test_", "challenge_predictions"))
                                 for path in output.rglob("*")))
            with self.assertRaises(FileExistsError):
                run(*inputs[:2], reference, protocol, output)

    def test_tampered_reference_inputs_rejected_before_fit(self):
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference = (
                diagnostic_fixture.EncoderObjectiveDiagnosticTests().make_reference(root)
            )
            path = inputs[1] / "train_steps.pt"
            path.write_bytes(path.read_bytes() + b"changed")
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(self.settings()))
            with patch("analysis.run_trajectory_correspondence_screen.train") as fitting:
                with self.assertRaisesRegex(ValueError, "inputs differ"):
                    run(*inputs[:2], reference, protocol, root / "screen")
                fitting.assert_not_called()


if __name__ == "__main__":
    unittest.main()
