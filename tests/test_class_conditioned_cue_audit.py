import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.audit_class_conditioned_cues import audit
from analysis.run_class_conditioned_cues import (
    CONDITIONS, NEW_CONDITIONS, POLICIES, decision, settings, validate_settings,
)
from src.evidence_repair import metrics
from src.provenance import sha256_file


class ClassConditionedCueAuditTests(unittest.TestCase):
    def test_configuration_matches_frozen_file_and_refuses_weight_search(self):
        config = json.loads((Path(__file__).resolve().parents[1] /
                             "configs/class_conditioned_cues.json").read_text())
        validate_settings(config)
        changed = copy.deepcopy(config)
        changed["configuration"]["pair_margin_weight"] = .2
        with self.assertRaisesRegex(ValueError, "settings"):
            validate_settings(changed)

    def test_complete_negative_removal_fails_continuation(self):
        records = [{"policy": policy, "condition": condition, "inner_selection": {
            "all_profiles": {"macro_f1": .96 if condition == "without_pair_margin" else
                            .95 if condition == CONDITIONS[0] else .94}}}
                   for policy in POLICIES for condition in CONDITIONS]
        result = decision(records)
        self.assertLess(result["comparisons"]["without_pair_margin"]["mean_difference"], 0)
        self.assertFalse(result["development_continuation_gate_passed"])
        self.assertFalse(result["submission_allowed"])
        with self.assertRaisesRegex(ValueError, "matrix"):
            decision(records[:-1])

    def test_audit_recomputes_predictions_and_detects_edited_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder)
            protocol = study / "protocol.json"
            protocol.write_text(json.dumps(settings()), encoding="utf-8")
            probabilities = np.eye(4) * .96 + .01
            scores = metrics(np.arange(4), probabilities, 4)
            records = []
            for policy in POLICIES:
                for condition in CONDITIONS:
                    cell = study / policy / condition
                    cell.mkdir(parents=True)
                    checkpoint = cell / "selected_model.pt"
                    checkpoint.write_bytes(b"hash-only fixture")
                    prediction = cell / "inner_selection_predictions.jsonl"
                    prediction.write_text("".join(json.dumps({
                        "sample_id": policy + str(label), "base_profile_id": policy,
                        "label": label, "probabilities": probabilities[label].tolist(),
                    }) + "\n" for label in range(4)), encoding="utf-8")
                    record = {
                        "policy": policy, "condition": condition,
                        "fit": {"new_fit": condition in NEW_CONDITIONS},
                        "inner_selection": {"all_profiles": scores},
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "predictions_sha256": sha256_file(prediction),
                        "maximum_replay_probability_difference": 0,
                        "original_validation_test_or_excluded_policy_forwards": False,
                        "submission_allowed": False,
                    }
                    (cell / "diagnostic.json").write_text(json.dumps(record), encoding="utf-8")
                    records.append(record)
            (study / "summary.json").write_text(json.dumps({
                "protocol_sha256": sha256_file(protocol), "records": records, **decision(records),
                "new_fits_completed": 12, "checkpoint_replays_completed": 18,
                "cached_controls_replayed": 6, "gpu_used": False,
                "new_account_generation": False, "original_validation_or_test_access": False,
            }), encoding="utf-8")
            result = audit(study)
            self.assertEqual(result["audited_cells"], 18)
            self.assertEqual(result["member_hash_checks"], 36)
            self.assertFalse(result["development_continuation_gate_passed"])
            checkpoint.write_bytes(b"edited")
            with self.assertRaisesRegex(ValueError, "member hash"):
                audit(study)


if __name__ == "__main__":
    unittest.main()
