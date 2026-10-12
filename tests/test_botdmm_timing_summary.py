import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.summarize_botdmm_timing import summarize
from src.provenance import sha256_file


class BotDMMTimingSummaryTests(unittest.TestCase):
    def fixture(self, temporary):
        root, reference = Path(temporary) / "round", Path(temporary) / "reference"
        frozen_by_key, records_by_key = {}, {}
        seeds, policies = [42, 43, 44], ["burst", "periodic", "replay"]
        for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
            for source in (root, reference):
                (source / dataset).mkdir(parents=True)
                (source / dataset / "protocol.json").write_text("{}", encoding="utf-8")
            common = {
                "seeds": seeds, "input_hashes": {"train": "same"},
                "timing": {"version": "same"}, "master_sha256": "same",
                "configuration": {"dropout": 0.2}, "class_count": classes,
            }
            frozen_by_key[(root, dataset)] = {
                **common, "methods": ["botdmm_timing", "botdmm_without_timing"],
                "mlp_reference_protocol_sha256": sha256_file(reference / dataset / "protocol.json"),
                "mlp_reference": {"same_recipe_keys": ["dropout"]},
            }
            frozen_by_key[(reference, dataset)] = {**common, "methods": ["erm_mlp"]}
            for source in (root, reference):
                records = {}
                methods = frozen_by_key[(source, dataset)]["methods"]
                for seed in seeds:
                    for policy in policies:
                        for method in methods:
                            folder = source / dataset / f"seed{seed}" / policy / method
                            folder.mkdir(parents=True)
                            rows = [{
                                "base_profile_id": profile, "label": label,
                                "prediction": 0 if method == "botdmm_timing" else label,
                            } for profile in ("p1", "p2") for label in range(classes)]
                            (folder / "challenge_predictions.jsonl").write_text(
                                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
                            )
                            records[(seed, policy, method)] = {
                                "fitting_profiles": 10, "selection_profiles": 2,
                                "challenge_profiles": 2,
                                "challenge": {"macro_f1": 0.1 if method == "botdmm_timing" else 1.0},
                                "fit": {"parameter_count": 100 if method == "botdmm_timing" else 90},
                            }
                records_by_key[(source, dataset)] = records
        return root, reference, frozen_by_key, records_by_key

    def test_complete_comparison_retains_negative_timing_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, reference, frozen, records = self.fixture(temporary)
            with patch("analysis.summarize_botdmm_timing.verified_records",
                       side_effect=lambda source, dataset: (
                           frozen[(source, dataset)], records[(source, dataset)]
                       )):
                report = summarize(root, reference, Path(temporary) / "summary", 100)
            self.assertFalse(report["submission_allowed"])
            for dataset in ("Twibot22", "Quadbot"):
                self.assertEqual(report["datasets"][dataset]["runs_verified"], 18)
                comparisons = report["datasets"][dataset]["paired_intervals"]["burst"]
                self.assertLess(comparisons["aggregate_comparisons"][
                    "botdmm_timing_minus_botdmm_without_timing"
                ]["macro_f1"]["estimate"], 0)

    def test_different_reference_input_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, reference, frozen, records = self.fixture(temporary)
            frozen[(reference, "Twibot22")]["input_hashes"] = {"train": "different"}
            with patch("analysis.summarize_botdmm_timing.verified_records",
                       side_effect=lambda source, dataset: (
                           frozen[(source, dataset)], records[(source, dataset)]
                       )):
                with self.assertRaises(ValueError):
                    summarize(root, reference, Path(temporary) / "summary", 100)


if __name__ == "__main__":
    unittest.main()
