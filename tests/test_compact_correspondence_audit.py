import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from threadpoolctl import threadpool_limits

from analysis.audit_compact_correspondence_stability import audit
from analysis.run_compact_correspondence_stability import run
from src.provenance import sha256_file
from tests import test_compact_correspondence as compact_fixture


class CompactCorrespondenceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        with threadpool_limits(limits=2), patch("builtins.print"):
            inputs, reference, settings = compact_fixture.CompactCorrespondenceTests().make_reference(cls.root)
            cls.settings_path = cls.root / "settings.json"
            cls.settings_path.write_text(json.dumps(settings))
            cls.output = cls.root / "study"
            with patch("analysis.run_compact_correspondence_stability.validate_settings"):
                run(*inputs[:2], reference, cls.settings_path, cls.output)
        cls.contents = {
            path.relative_to(cls.output).as_posix(): path.read_bytes()
            for path in cls.output.rglob("*") if path.is_file()
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def material(self, name, changed=None):
        folder = self.root / name
        folder.mkdir()
        contents = {**self.contents, **(changed or {})}
        archive_path = folder / "evidence.zip"
        with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
            for member, content in contents.items():
                archive.writestr(member, content)
        summary_path = folder / "summary.json"
        summary_path.write_bytes(contents["summary.json"])
        receipt_path = folder / "receipt.json"
        receipt_path.write_text(json.dumps({
            "phase": "complete", "completed_new_fits": 30, "reused_reference_checkpoints": 6,
            "checkpoint_replays": 36, "gpu_used": False, "challenge_forward_or_scoring": False,
            "submission_allowed": False, "reference_files_unchanged": True,
            "summary_sha256": sha256_file(summary_path),
            "evidence_archive_sha256": sha256_file(archive_path),
        }))
        return archive_path, summary_path, receipt_path

    def run_audit(self, paths):
        with patch("analysis.audit_compact_correspondence_stability.validate_settings"):
            return audit(*paths, self.settings_path)

    def test_receipt_recomputes_every_cell_and_preserves_boundary(self):
        result = self.run_audit(self.material("valid"))
        self.assertEqual(result["verified_cells"], 36)
        self.assertEqual(result["new_fits_verified"], 30)
        self.assertEqual(result["reused_reference_checkpoints_verified"], 6)
        self.assertFalse(result["submission_allowed"])
        self.assertFalse(result["research_tensors_opened"])
        self.assertFalse(result["challenge_predictions_opened"])
        self.assertFalse(result["independent_confirmation"])

    def test_unsafe_or_challenge_archive_members_rejected(self):
        for name, member in (("unsafe", "../outside"), ("challenge", "challenge_predictions.jsonl")):
            with self.assertRaisesRegex(ValueError, "evidence member"):
                self.run_audit(self.material(name, {member: b"{}"}))

    def test_tampered_probabilities_detected_after_rehashed_receipt(self):
        member = next(name for name in self.contents if name.endswith("inner_selection_predictions.jsonl"))
        rows = [json.loads(line) for line in self.contents[member].decode().splitlines()]
        original_prediction = max(range(4), key=lambda index: rows[0]["probabilities"][index])
        rows[0]["probabilities"] = [int(index == (original_prediction + 1) % 4)
                                    for index in range(4)]
        changed = {member: ("\n".join(json.dumps(row) for row in rows) + "\n").encode()}
        diagnostic = member.replace("inner_selection_predictions.jsonl", "diagnostic.json")
        record = json.loads(self.contents[diagnostic])
        record["predictions_sha256"] = hashlib.sha256(changed[member]).hexdigest()
        changed[diagnostic] = json.dumps(record).encode()
        summary = json.loads(self.contents["summary.json"])
        for index, candidate in enumerate(summary["records"]):
            if (candidate["seed"], candidate["policy"], candidate["condition"]) == (
                record["seed"], record["policy"], record["condition"],
            ):
                summary["records"][index] = record
        changed["summary.json"] = json.dumps(summary).encode()
        with self.assertRaisesRegex(ValueError, "metrics differ"):
            self.run_audit(self.material("probability_tamper", changed))

    def test_false_submission_flag_rejected(self):
        summary = json.loads(self.contents["summary.json"])
        summary["submission_allowed"] = True
        with self.assertRaisesRegex(ValueError, "scope differs"):
            self.run_audit(self.material("false_claim", {"summary.json": json.dumps(summary).encode()}))


if __name__ == "__main__":
    unittest.main()
