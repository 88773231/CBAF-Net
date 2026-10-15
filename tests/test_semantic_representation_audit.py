import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.audit_semantic_representation import audit
from analysis.run_semantic_representation import CONDITIONS, POLICIES, decision
from src.evidence_repair import metrics
from src.provenance import sha256_file


class RepresentationAuditTests(unittest.TestCase):
    def make_study(self, folder):
        root = Path(folder)
        config = Path(__file__).resolve().parents[1] / "configs/semantic_representation.json"
        protocol = {
            **json.loads(config.read_text()),
            "prior_partition_hashes": {policy: "partition-" + policy for policy in POLICIES},
        }
        (root / "protocol.json").write_text(json.dumps(protocol))
        records = []
        for policy in POLICIES:
            parent = root / policy
            parent.mkdir()
            cache = parent / "known_observations.pt"
            cache.write_bytes(b"fixture-cache")
            (parent / "feature_receipt.json").write_text(json.dumps({
                "cache_sha256": sha256_file(cache),
                "partition_sha256": "partition-" + policy,
                "normalization_scope": "inner_fitting_only",
                "excluded_policy_encoded_transformed_or_forwarded": False,
            }))
            predictions = [{
                "sample_id": policy + str(label), "base_profile_id": policy,
                "label": label, "probabilities": (np.eye(4)[label] * .8 + .05).tolist(),
            } for label in range(4)]
            score = metrics(np.arange(4), np.asarray([row["probabilities"] for row in predictions]), 4)
            for condition in CONDITIONS:
                cell = parent / condition
                cell.mkdir()
                checkpoint = cell / "selected_model.pt"
                checkpoint.write_bytes(b"fixture-not-a-real-model")
                prediction = cell / "inner_selection_predictions.jsonl"
                prediction.write_text("".join(json.dumps(row) + "\n" for row in predictions))
                record = {
                    "policy": policy, "condition": condition,
                    "inner_selection": {"all_profiles": score},
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "predictions_sha256": sha256_file(prediction),
                    "maximum_replay_probability_difference": 0,
                    "original_validation_test_or_excluded_policy_forwards": False,
                    "submission_allowed": False,
                    "fit": {"initialization_digest": condition.split("_")[1],
                            "training_parameter_count": 100},
                }
                (cell / "diagnostic.json").write_text(json.dumps(record))
                records.append(record)
        summary = {
            "protocol_sha256": sha256_file(root / "protocol.json"), "records": records,
            **decision(records), "new_fits_completed": 12, "checkpoint_replays_completed": 12,
            "gpu_used": False, "new_account_generation": False,
            "original_validation_or_test_access": False,
        }
        (root / "summary.json").write_text(json.dumps(summary))
        return root

    def test_metrics_hashes_and_negative_result_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            study = self.make_study(folder)
            result = audit(study)
            self.assertEqual(result["audited_cells"], 12)
            self.assertEqual(result["member_hash_checks"], 27)
            self.assertFalse(result["representation_continuation_gate_passed"])

    def test_modified_cache_and_predictions_rejected(self):
        for filename in ("known_observations.pt", "hash_mlp/inner_selection_predictions.jsonl"):
            with tempfile.TemporaryDirectory() as folder:
                study = self.make_study(folder)
                (study / POLICIES[0] / filename).write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    audit(study)

    def test_changed_initialization_rejected_even_with_summary_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            study = self.make_study(folder)
            path = study / POLICIES[0] / "semantic_mlp/diagnostic.json"
            record = json.loads(path.read_text())
            record["fit"]["initialization_digest"] = "different"
            path.write_text(json.dumps(record))
            summary_path = study / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["records"][1] = record
            summary_path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                audit(study)


if __name__ == "__main__":
    unittest.main()
