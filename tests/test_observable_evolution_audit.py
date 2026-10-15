import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.audit_observable_evolution import audit
from analysis.run_observable_evolution import CONDITIONS, POLICIES, decision, validate_settings
from src.evidence_repair import metrics
from src.provenance import sha256_file


class ObservableEvolutionAuditTests(unittest.TestCase):
    def test_configuration_is_frozen(self):
        config = json.loads((Path(__file__).resolve().parents[1] /
                             "configs/observable_evolution.json").read_text())
        validate_settings(config)
        changed = copy.deepcopy(config)
        changed["configuration"]["distillation_weight"] = .2
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_settings(changed)

    def test_negative_module_effect_fails_and_incomplete_matrix_is_rejected(self):
        records = [{"policy": policy, "condition": condition,
                    "inner_selection": {"all_profiles": {
                        "macro_f1": .95 if condition == "without_distillation" else
                        .9 if condition == CONDITIONS[0] else .85}}}
                   for policy in POLICIES for condition in CONDITIONS]
        result = decision(records)
        self.assertFalse(result["development_continuation_gate_passed"])
        self.assertLess(result["comparisons"]["without_distillation"]["mean_difference"], 0)
        self.assertEqual(result["comparisons"]["without_distillation"]["positive_folds"], 0)
        self.assertFalse(result["submission_allowed"])
        with self.assertRaises(ValueError):
            decision(records[:-1])

    def test_prediction_audit_recomputes_complete_matrix_and_detects_member_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            configuration = json.loads((Path(__file__).resolve().parents[1] /
                                        "configs/observable_evolution.json").read_text())
            protocol = study / "protocol.json"
            protocol.write_text(json.dumps(configuration), encoding="utf-8")
            records = []
            probabilities = np.eye(4) * .9 + .025
            scores = metrics(np.arange(4), probabilities, 4)
            for policy in POLICIES:
                for condition in CONDITIONS:
                    cell = study / policy / condition
                    cell.mkdir(parents=True)
                    checkpoint = cell / "selected_model.pt"
                    checkpoint.write_bytes(b"hash-only fixture; training replay is tested separately")
                    prediction = cell / "inner_selection_predictions.jsonl"
                    prediction.write_text("".join(json.dumps({
                        "sample_id": policy + str(label), "base_profile_id": policy,
                        "label": label, "probabilities": probabilities[label].tolist(),
                    }) + "\n" for label in range(4)), encoding="utf-8")
                    record = {
                        "policy": policy, "condition": condition,
                        "inner_selection": {"all_profiles": scores},
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "predictions_sha256": sha256_file(prediction),
                        "maximum_replay_probability_difference": 0,
                        "original_validation_test_or_excluded_policy_forwards": False,
                        "submission_allowed": False,
                    }
                    (cell / "diagnostic.json").write_text(json.dumps(record), encoding="utf-8")
                    records.append(record)
            summary = {
                "protocol_sha256": sha256_file(protocol), "records": records, **decision(records),
                "new_fits_completed": 15, "checkpoint_replays_completed": 15,
                "gpu_used": False, "new_account_generation": False,
                "original_validation_or_test_access": False,
            }
            (study / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            result = audit(study)
            self.assertEqual(result["audited_cells"], 15)
            self.assertEqual(result["member_hash_checks"], 30)
            self.assertFalse(result["development_continuation_gate_passed"])
            with prediction.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "member hash"):
                audit(study)


if __name__ == "__main__":
    unittest.main()
