import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.audit_observable_schema import audit, training_identities


class ObservableSchemaAuditTests(unittest.TestCase):
    def fixture(self, root):
        metadata, master = root / "train_metadata.jsonl", root / "master.jsonl"
        identities, rows = [], []
        for label in range(4):
            identities.append({"row_index": label, "sample_id": str(label),
                               "base_profile_id": "fitting-profile", "label": label})
            rows.append({**identities[-1], "posts": [{"text": "private text"}],
                         "actions": [{"environment_triggered": label == 3}]})
        rows.append({"sample_id": "excluded", "posts": "not examined", "actions": "not examined"})
        metadata.write_text("".join(json.dumps(row) + "\n" for row in identities), encoding="utf-8")
        master.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return master, metadata

    def run_audit(self, master, metadata):
        return audit(master, metadata, hashlib.sha256(master.read_bytes()).hexdigest(),
                     hashlib.sha256(metadata.read_bytes()).hexdigest())

    def test_only_selected_rows_are_examined_and_no_raw_text_is_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            master, metadata = self.fixture(Path(folder))
            result = self.run_audit(master, metadata)
            self.assertEqual(result["training_rows"], 4)
            self.assertEqual(result["classes"]["3"]["trigger_true_counts"],
                             [{"value": 1, "trajectories": 1}])
            self.assertNotIn("private text", json.dumps(result))
            self.assertNotIn("fitting-profile", json.dumps(result))
            self.assertFalse(result["proven_label_leakage"])

    def test_frozen_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            master, metadata = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "metadata hash"):
                audit(master, metadata, "0" * 64, "0" * 64)
            with self.assertRaisesRegex(ValueError, "source hash"):
                audit(master, metadata, "0" * 64, hashlib.sha256(metadata.read_bytes()).hexdigest())

    def test_nontraining_metadata_and_incomplete_profiles_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            master, metadata = self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, "training metadata"):
                training_identities(Path(folder) / "test_metadata.jsonl")
            records = metadata.read_text().splitlines()
            metadata.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly four"):
                self.run_audit(master, metadata)

    def test_duplicate_selected_master_row_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            master, metadata = self.fixture(Path(folder))
            with master.open("a", encoding="utf-8") as stream:
                stream.write(master.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate selected"):
                self.run_audit(master, metadata)


if __name__ == "__main__":
    unittest.main()
