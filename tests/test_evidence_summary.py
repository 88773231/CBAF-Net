import unittest

from analysis.summarize_evidence_repair import audit_summary


class EvidenceSummaryTests(unittest.TestCase):
    def summary(self):
        runs = [
            {"method": "candidate", "seed": seed, "test_state": "NOT_OPENED",
             "validation": {"macro_f1": 0.5, "accuracy": 0.6, "mcc": 0.4}}
            for seed in (42, 43, 44)
        ]
        return {
            "test_state": "NOT_OPENED", "runs": runs,
            "aggregates": {"candidate": {
                metric: {"mean": value, "seed_sd": 0.0}
                for metric, value in (("macro_f1", 0.5), ("accuracy", 0.6), ("mcc", 0.4))
            }},
        }

    def test_missing_seed_and_duplicate_runs_rejected(self):
        summary = self.summary()
        summary["runs"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            audit_summary(summary, ["candidate"])
        summary = self.summary()
        summary["runs"].append(summary["runs"][0])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            audit_summary(summary, ["candidate"])

    def test_invented_aggregate_is_rejected(self):
        summary = self.summary()
        summary["aggregates"]["candidate"]["macro_f1"]["mean"] = 0.9
        with self.assertRaisesRegex(ValueError, "aggregate"):
            audit_summary(summary, ["candidate"])

    def test_test_access_cannot_be_relabelled_validation(self):
        summary = self.summary()
        summary["runs"][0]["test_state"] = "OPENED"
        with self.assertRaisesRegex(ValueError, "test use"):
            audit_summary(summary, ["candidate"])

    def test_valid_complete_aggregate_recomputed(self):
        rows = audit_summary(self.summary(), ["candidate"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["macro_f1"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
