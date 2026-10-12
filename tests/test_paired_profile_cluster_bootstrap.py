import json
import tempfile
import unittest
from pathlib import Path

from analysis import paired_profile_cluster_bootstrap as paired


class PairedProfileClusterBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, name, predictions, pred_key="pred"):
        path = self.root / name
        rows = []
        labels = [0, 1, 0, 1, 0, 1]
        for index, (label, prediction) in enumerate(zip(labels, predictions)):
            rows.append(
                {
                    "base_profile_id": f"profile-{index // 2}",
                    "label": label,
                    pred_key: prediction,
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_paired_aggregate_detects_consistent_improvement(self):
        seeds = {}
        for seed in (42, 43, 44):
            proposed = self.write_rows(f"proposed-{seed}.jsonl", [0, 1, 0, 1, 0, 1])
            reference = self.write_rows(f"reference-{seed}.jsonl", [0, 0, 0, 0, 0, 0])
            seeds[str(seed)] = {
                "Proposed": {"path": proposed.name},
                "Reference": {"path": reference.name},
            }
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "reference_methods": ["Reference"],
                    "datasets": {"Toy": {"seeds": seeds}},
                }
            ),
            encoding="utf-8",
        )

        result = paired.run(manifest, replicates=200, seed=7)
        comparison = result["datasets"]["Toy"]["aggregate_comparisons"][
            "Proposed_minus_Reference"
        ]

        self.assertGreater(comparison["macro_f1"]["estimate"], 0.0)
        self.assertGreater(comparison["macro_f1"]["lower_95"], 0.0)
        self.assertEqual(
            comparison["macro_f1"]["bootstrap_probability_positive"], 1.0
        )

    def test_alignment_mismatch_is_rejected(self):
        first = self.write_rows("first.jsonl", [0, 1, 0, 1, 0, 1])
        second = self.write_rows("second.jsonl", [0, 1, 0, 1, 0, 1])
        rows = second.read_text(encoding="utf-8").splitlines()
        payload = json.loads(rows[-1])
        payload["base_profile_id"] = "different-profile"
        rows[-1] = json.dumps(payload)
        second.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "reference_methods": ["B"],
                    "datasets": {
                        "Toy": {
                            "seeds": {
                                "42": {
                                    "A": {"path": first.name},
                                    "B": {"path": second.name},
                                }
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "row alignment differs"):
            paired.run(manifest, replicates=100, seed=7)


if __name__ == "__main__":
    unittest.main()
