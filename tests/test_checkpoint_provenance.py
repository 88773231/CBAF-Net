import json
import tempfile
import unittest
from pathlib import Path

from src.provenance import sha256_file
from src.run_cbaf import validate_checkpoint_protocol


class CheckpointProvenanceTests(unittest.TestCase):
    def _summary(self, digest=None):
        summary = {
            "dataset": "Twibot22",
            "configuration": {
                "edge_policy": "strict_split_local_knn",
                "batching": "one_hop_target_neighbor",
                "message_direction": "selected_neighbor_to_target",
                "knn_k": 10,
                "checkpoint_format": "cbaf-botdmm-checkpoint-v2",
                "tabular_preprocessing": {
                    "version": "train-column-zscore-v1",
                    "signature_sha256": "preprocessing-signature",
                },
            },
        }
        if digest is not None:
            summary["checkpoint_sha256"] = digest
        return summary

    def test_declared_checkpoint_hash_is_verified_and_tampering_is_rejected(self):
        preprocessing = {
            "version": "train-column-zscore-v1",
            "signature_sha256": "preprocessing-signature",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"release checkpoint bytes")
            summary_path = root / "training_summary.json"
            summary_path.write_text(
                json.dumps(self._summary(sha256_file(checkpoint))),
                encoding="utf-8",
            )

            self.assertEqual(
                validate_checkpoint_protocol(checkpoint, "twibot22", 10, preprocessing),
                summary_path,
            )

            checkpoint.write_bytes(b"tampered checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_checkpoint_protocol(checkpoint, "twibot22", 10, preprocessing)

    def test_summary_without_hash_is_rejected_by_default(self):
        preprocessing = {
            "version": "train-column-zscore-v1",
            "signature_sha256": "preprocessing-signature",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            checkpoint.write_bytes(b"legacy checkpoint bytes")
            summary_path = root / "training_summary.json"
            summary_path.write_text(json.dumps(self._summary()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing checkpoint_sha256"):
                validate_checkpoint_protocol(checkpoint, "twibot22", 10, preprocessing)


if __name__ == "__main__":
    unittest.main()
