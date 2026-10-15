import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import torch
from threadpoolctl import threadpool_limits

from analysis.audit_trajectory_correspondence_screen import audit, decision
from analysis.run_factor_training_screen import POLICIES
from analysis.run_trajectory_correspondence_screen import run
from src.provenance import sha256_file
from src.trajectory_correspondence import CONDITIONS
from tests import test_encoder_objective_diagnostic as fixture_module


class TrajectoryCorrespondenceAuditTests(unittest.TestCase):
    def records(self):
        return [{
            "policy": policy, "condition": condition,
            "inner_selection": {"all_profiles": {"macro_f1": .99 if index == 0 else .97}},
        } for policy in POLICIES for index, condition in enumerate(CONDITIONS)]

    def test_positive_gate_requires_each_removal_and_reference(self):
        report = decision(self.records(), dict.fromkeys(POLICIES, .98))
        self.assertTrue(report["feasibility_gate_passed"])
        self.assertFalse(report["independent_confirmation"])
        self.assertFalse(report["submission_allowed"])
        for row in report["removal_comparisons"].values():
            self.assertEqual(row["strict_positive_folds"], 3)
        self.assertFalse(decision(self.records(), dict.fromkeys(POLICIES, .995))[
            "feasibility_gate_passed"])

    def test_one_favorable_fold_or_tied_mean_cannot_rescue_removal(self):
        records = self.records()
        for row in records:
            if row["condition"] == "without_correspondence":
                row["inner_selection"]["all_profiles"]["macro_f1"] = (
                    .94 if row["policy"] == POLICIES[0] else .995
                )
        report = decision(records, dict.fromkeys(POLICIES, .98))
        self.assertGreater(report["removal_comparisons"]["without_correspondence"]["mean_difference"], 0)
        self.assertFalse(report["feasibility_gate_passed"])
        for row in records:
            if row["condition"] == "without_correspondence":
                row["inner_selection"]["all_profiles"]["macro_f1"] = .99
        self.assertFalse(decision(records, dict.fromkeys(POLICIES, .98))["feasibility_gate_passed"])

    def test_missing_duplicate_and_nonfinite_records_fail(self):
        records = self.records()
        for changed in (records[:-1], records[:-1] + [copy.deepcopy(records[0])]):
            with self.assertRaises(ValueError):
                decision(changed, dict.fromkeys(POLICIES, .98))
        for value in (float("nan"), float("inf"), -1, 1.1):
            changed = copy.deepcopy(records)
            changed[0]["inner_selection"]["all_profiles"]["macro_f1"] = value
            with self.assertRaises(ValueError):
                decision(changed, dict.fromkeys(POLICIES, .98))
        with self.assertRaises(ValueError):
            decision(records, {POLICIES[0]: .98})

    def test_complete_read_only_receipt_and_invented_metric_rejected(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=2), patch("builtins.print"):
            root = Path(folder)
            inputs, reference = fixture_module.EncoderObjectiveDiagnosticTests().make_reference(root)
            settings = Path(__file__).parents[1] / "configs/trajectory_correspondence_screen.json"
            evidence = root / "screen"
            run(*inputs[:2], reference, settings, evidence)
            selections = root / "selections.json"
            reference_records = []
            for policy in POLICIES:
                result = json.loads((reference / "Quadbot/seed42" / policy /
                                     "botsscl_supervised_only/metrics.json").read_text())
                selected = next(row for row in result["fit"]["history"]
                                if row["epoch"] == result["fit"]["best_epoch"])
                reference_records.append({
                    "policy": policy, "condition": "supervised_reference", "seed": 42,
                    "inner_selection": selected["inner_selection"],
                })
            selections.write_text(json.dumps({
                "dataset": "Quadbot", "records": reference_records,
                "challenge_forward_or_scoring": False,
                "independent_confirmation": False, "submission_allowed": False,
            }))
            archive = root / "evidence.zip"
            with ZipFile(archive, "w") as stream:
                for path in evidence.rglob("*"):
                    if path.is_file():
                        stream.write(path, path.relative_to(evidence))
            receipt_path = root / "receipt.json"
            receipt = {
                "phase": "complete", "completed_new_fits": 12, "checkpoint_replays": 12,
                "gpu_used": False, "challenge_forward_or_scoring": False,
                "submission_allowed": False, "reference_files_unchanged": True,
                "summary_sha256": sha256_file(evidence / "summary.json"),
                "evidence_archive_sha256": sha256_file(archive),
            }
            receipt_path.write_text(json.dumps(receipt))
            original = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            report = audit(archive, evidence / "summary.json", receipt_path,
                           reference / "Quadbot/protocol.json", selections)
            self.assertEqual(report["verified_cells"], 12)
            self.assertEqual(report["research_fits_added"], 0)
            self.assertFalse(report["research_tensors_opened"])
            self.assertFalse(report["submission_allowed"])
            self.assertEqual(original, {str(path): path.read_bytes() for path in root.rglob("*")
                                        if path.is_file()})
            with ZipFile(archive) as stream:
                members = {name: stream.read(name) for name in stream.namelist()}
            summary = json.loads(members["summary.json"])
            changed = summary["records"][0]
            old_score = changed["inner_selection"]["all_profiles"]["macro_f1"]
            changed["inner_selection"]["all_profiles"]["macro_f1"] = (
                old_score + .01 if old_score <= .99 else old_score - .01
            )
            member = changed["policy"] + "/" + changed["condition"] + "/diagnostic.json"
            members[member] = json.dumps(changed).encode()
            members["summary.json"] = json.dumps(summary).encode()
            modified = root / "invented.zip"
            with ZipFile(modified, "w") as stream:
                for name, content in members.items():
                    stream.writestr(name, content)
            modified_summary = root / "invented_summary.json"
            modified_summary.write_bytes(members["summary.json"])
            receipt.update({
                "summary_sha256": hashlib.sha256(members["summary.json"]).hexdigest(),
                "evidence_archive_sha256": sha256_file(modified),
            })
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "metrics differ"):
                audit(modified, modified_summary, receipt_path,
                      reference / "Quadbot/protocol.json", selections)


if __name__ == "__main__":
    unittest.main()
