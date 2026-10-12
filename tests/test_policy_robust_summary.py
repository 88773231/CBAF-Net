import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.summarize_policy_robust import verified_records
from src.evidence_repair import metrics
from src.provenance import sha256_file


class PolicyRobustSummaryTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "Twibot22"
        source.mkdir()
        frozen = {
            "seeds": [42, 43, 44], "methods": ["erm_mlp", "group_dro_rdrop_mlp"],
            "class_count": 3, "test_paths_opened": False,
            "outer_challenge_used_for_selection": False,
        }
        (source / "protocol.json").write_text(json.dumps(frozen))
        protocol_hash = sha256_file(source / "protocol.json")
        rows = []
        for seed in frozen["seeds"]:
            for policy in ("burst", "uniform", "cyclic"):
                for method in frozen["methods"]:
                    destination = source / f"seed{seed}" / policy / method
                    destination.mkdir(parents=True)
                    probabilities = np.eye(3)
                    row = {
                        "dataset": "Twibot22", "seed": seed, "heldout_policy": policy,
                        "method": method, "challenge_profiles": 1,
                        "test_state": "NOT_OPENED", "protocol_sha256": protocol_hash,
                        "outer_challenge_used_for_selection": False,
                        "challenge": metrics(np.arange(3), probabilities, 3),
                    }
                    (destination / "metrics.json").write_text(json.dumps(row))
                    predictions = [
                        {"base_profile_id": policy, "label": label,
                         "prediction": label, "probabilities": probabilities[label].tolist()}
                        for label in range(3)
                    ]
                    (destination / "challenge_predictions.jsonl").write_text(
                        "\n".join(json.dumps(item) for item in predictions)
                    )
                    rows.append(row)
        summary = {
            "test_state": "NOT_OPENED", "submission_allowed": False,
            "protocol_sha256": protocol_hash, "runs": rows,
        }
        (source / "summary.json").write_text(json.dumps(summary))
        return source, summary

    def test_complete_prediction_recomputed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            _, records = verified_records(root, "Twibot22")
            self.assertEqual(len(records), 18)

    def test_missing_seed_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, summary = self.fixture(root)
            summary["runs"].pop()
            (source / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verified_records(root, "Twibot22")

    def test_metrics_invented_in_both_summary_and_run_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, summary = self.fixture(root)
            row = summary["runs"][0]
            row["challenge"]["macro_f1"] = 0.9
            destination = source / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
            (destination / "metrics.json").write_text(json.dumps(row))
            (source / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verified_records(root, "Twibot22")

    def test_test_access_and_submission_relabelling_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, summary = self.fixture(root)
            summary["submission_allowed"] = True
            (source / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                verified_records(root, "Twibot22")


if __name__ == "__main__":
    unittest.main()
