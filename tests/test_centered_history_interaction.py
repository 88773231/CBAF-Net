import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from threadpoolctl import threadpool_limits

from analysis.run_centered_history_interaction import CONDITIONS, decision, run, validate_settings
from analysis.audit_centered_history_interaction import audit
from analysis.run_compact_correspondence_stability import run as run_retained
from analysis.run_factor_training_screen import POLICIES
from src.centered_history_interaction import interaction_features, make_model, predict, train
from src.compact_correspondence import make_model as removal_model, shared_state_digest
from src.evidence_repair import model_inputs, seed_everything
from tests import test_trajectory_correspondence as fixture
from tests import test_compact_correspondence as compact_fixture


class CenteredHistoryInteractionTests(unittest.TestCase):
    def populations(self):
        return fixture.TrajectoryCorrespondenceTests().populations()

    def test_centered_relation_is_offset_invariant_and_not_future_dependent(self):
        generator = torch.Generator().manual_seed(42)
        text = torch.randn(4, 5, 8, generator=generator)
        action = torch.randn(4, 5, 8, generator=generator)
        first = interaction_features(text, action)
        shifted = interaction_features(text + 3, action - 2)
        for original, changed in zip(first, shifted):
            torch.testing.assert_close(original, changed, atol=1e-6, rtol=1e-5)
            self.assertTrue((original[:, 0] == 0).all())
        modified = action.clone()
        modified[:, 3:].add_(20)
        for original, changed in zip(first, interaction_features(text, modified)):
            torch.testing.assert_close(original[:, :3], changed[:, :3])
        constant = torch.ones(4, 5, 8)
        for block in interaction_features(constant, action):
            torch.testing.assert_close(block, torch.zeros_like(block), atol=1e-6, rtol=0)
        with self.assertRaises(ValueError):
            interaction_features(text[:, :4], action[:, :4])

    def test_shared_initialization_no_dead_attention_and_nonlinear_gradient(self):
        fitting, _, _, config = self.populations()
        digests = []
        for condition in CONDITIONS[:3]:
            seed_everything(42)
            model = (make_model(fitting, config, condition) if condition in CONDITIONS[:2]
                     else removal_model(fitting, config, condition))
            digests.append(shared_state_digest(model))
            self.assertFalse(any(name.startswith(("correspondence.", "transition."))
                                 for name in model.state_dict()))
            torch.nn.functional.cross_entropy(
                model(*model_inputs(fitting, "mlp_matched"))["logits"], fitting["labels"]).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
        self.assertEqual(len(set(digests)), 1)

    def test_training_roundtrip_and_metadata_not_prediction_input(self):
        fitting, selection, policies, config = self.populations()
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2):
            for condition in CONDITIONS[:2]:
                model, record = train(fitting, selection, policies, condition, 42, config)
                self.assertEqual(record["inactive_serialized_weight_count"], 0)
                checkpoint = Path(folder) / (condition + ".pt")
                torch.save(model.state_dict(), checkpoint)
                replay = make_model(fitting, config, condition)
                replay.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                torch.testing.assert_close(torch.from_numpy(predict(model, selection, 2)),
                                           torch.from_numpy(predict(replay, selection, 12)))
                copied = {**selection, "metadata": [{}] * len(selection["labels"])}
                torch.testing.assert_close(torch.from_numpy(predict(model, copied, 12)),
                                           torch.from_numpy(predict(model, selection, 12)))
        with patch("src.centered_history_interaction.CenteredHistoryInteractionModel") as constructor:
            with self.assertRaises(ValueError):
                train(fitting, fitting, policies, CONDITIONS[0], 42, config)
            with self.assertRaises(ValueError):
                train(fitting, selection, policies, CONDITIONS[0], 42, {**config, "batch_size": 3})
            constructor.assert_not_called()

    def test_fixed_gate_rejects_tiny_module_gain_and_incomplete_matrix(self):
        settings = json.loads((Path(__file__).parents[1] /
                               "configs/centered_history_interaction.json").read_text())
        validate_settings(settings)
        records = [
            {"policy": policy, "condition": condition,
             "inner_selection": {"all_profiles": {"macro_f1": .91 if condition == CONDITIONS[0] else .90}}}
            for policy in POLICIES for condition in CONDITIONS]
        self.assertTrue(decision(records)["development_continuation_gate_passed"])
        for row in records:
            if row["condition"] == "without_centering":
                row["inner_selection"]["all_profiles"]["macro_f1"] = .9099
        self.assertFalse(decision(records)["development_continuation_gate_passed"])
        with self.assertRaises(ValueError):
            decision(records[:-1])
        with self.assertRaises(ValueError):
            validate_settings({**settings, "original_validation_or_test_access": True})

    def test_fifteen_cell_reference_roundtrip_and_no_excluded_predictions(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference, retained_settings = compact_fixture.CompactCorrespondenceTests().make_reference(root)
            retained_protocol = root / "retained_settings.json"
            retained_protocol.write_text(json.dumps(retained_settings))
            retained = root / "retained"
            with patch("analysis.run_compact_correspondence_stability.validate_settings"):
                run_retained(*inputs[:2], reference, retained_protocol, retained)
            preserved = {str(path): path.read_bytes() for path in retained.rglob("*") if path.is_file()}
            settings = json.loads((Path(__file__).parents[1] /
                                   "configs/centered_history_interaction.json").read_text())
            settings["configuration"] = retained_settings["configuration"]
            protocol = root / "settings.json"
            protocol.write_text(json.dumps(settings))
            output = root / "screen"
            with patch("analysis.run_centered_history_interaction.validate_settings"):
                run(*inputs[:2], reference, retained, protocol, output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["new_fits"], 6)
            self.assertEqual(summary["reused_exact_partition_checkpoints"], 9)
            self.assertEqual(summary["checkpoint_replays"], 15)
            self.assertEqual(len(summary["records"]), 15)
            self.assertEqual(summary["decision"], decision(summary["records"]))
            self.assertEqual(preserved, {str(path): path.read_bytes()
                                        for path in retained.rglob("*") if path.is_file()})
            self.assertFalse(any("challenge_predictions" in str(path)
                                 or path.name.startswith(("val_", "test_"))
                                 for path in output.rglob("*")))
            for row in summary["records"]:
                folder = output / row["policy"] / row["condition"]
                predicted = [json.loads(line) for line in
                             (folder / "inner_selection_predictions.jsonl").read_text().splitlines()]
                original = [json.loads(line) for line in
                            (retained / row["policy"] / "seed42/without_correspondence" /
                             "inner_selection_predictions.jsonl").read_text().splitlines()]
                self.assertEqual([(r["sample_id"], r["base_profile_id"], r["label"]) for r in predicted],
                                 [(r["sample_id"], r["base_profile_id"], r["label"]) for r in original])
            archive = root / "evidence.zip"
            receipt = root / "receipt.json"

            def package():
                with ZipFile(archive, "w", compression=ZIP_DEFLATED) as stream:
                    for path in output.rglob("*"):
                        if path.is_file():
                            stream.write(path, path.relative_to(output))
                receipt.write_text(json.dumps({
                    "phase": "complete", "new_fits": 6, "reused_exact_partition_checkpoints": 9,
                    "checkpoint_replays": 15, "gpu_used": False, "challenge_forward_or_scoring": False,
                    "submission_allowed": False, "reference_checkpoints_unchanged": True,
                    "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                }))

            package()
            with patch("analysis.audit_centered_history_interaction.validate_settings"):
                audited = audit(archive, output / "summary.json", receipt, protocol)
                self.assertEqual(audited["verified_cells"], 15)
                summary["records"][0]["inner_selection"]["all_profiles"]["macro_f1"] = .123
                row = summary["records"][0]
                (output / row["policy"] / row["condition"] / "diagnostic.json").write_text(json.dumps(row))
                (output / "summary.json").write_text(json.dumps(summary))
                package()
                with self.assertRaisesRegex(ValueError, "scores differ"):
                    audit(archive, output / "summary.json", receipt, protocol)
            with self.assertRaises(FileExistsError):
                run(*inputs[:2], reference, retained, protocol, output)


if __name__ == "__main__":
    unittest.main()
