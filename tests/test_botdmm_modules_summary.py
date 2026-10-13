import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.summarize_botdmm_modules import summarize
from src.provenance import sha256_file


class BotDMMModulesSummaryTests(unittest.TestCase):
    def fixture(self, temporary):
        root, baseline = Path(temporary) / "round", Path(temporary) / "baseline"
        protocols, records = {}, {}
        for dataset, classes in (("Twibot22", 3), ("Quadbot", 4)):
            for source in (root, baseline):
                (source / dataset).mkdir(parents=True)
                (source / dataset / "protocol.json").write_text("{}", encoding="utf-8")
                methods = (["botdmm_timing"] if source == baseline
                           else ["botdmm_without_structure", "botdmm_without_orthogonal"])
                protocols[(source, dataset)] = {
                    "configuration": {"dropout": 0.2}, "input_hashes": {"train": "same"},
                    "graph_input_hashes": {"train": "same"}, "timing": {"version": "same"},
                    "seeds": [42, 43, 44], "master_sha256": "same", "methods": methods,
                    "class_count": classes,
                    "baseline_protocol_sha256": sha256_file(baseline / dataset / "protocol.json")
                    if (baseline / dataset / "protocol.json").is_file() else "",
                }
                rows_by_key = {}
                for seed in (42, 43, 44):
                    for policy in ("burst", "cyclic", "uniform"):
                        for method in methods:
                            folder = source / dataset / f"seed{seed}" / policy / method
                            folder.mkdir(parents=True)
                            rows = [{
                                "base_profile_id": profile, "label": label,
                                "prediction": 0 if method == "botdmm_without_structure" else label,
                            } for profile in ("p1", "p2") for label in range(classes)]
                            (folder / "challenge_predictions.jsonl").write_text(
                                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
                            )
                            rows_by_key[(seed, policy, method)] = {
                                "fitting_profiles": 10, "selection_profiles": 2,
                                "challenge_profiles": 2, "graph_challenge": {"hash": "same"},
                                "fit": {"parameter_count": 100, "graph_fit": {"hash": "same"},
                                        "graph_inner_selection": {"hash": "same"}},
                                "challenge": {
                                    "macro_f1": 0.1 if method == "botdmm_without_structure" else 1.0,
                                    "per_class_f1": [0.1] * classes,
                                },
                            }
                records[(source, dataset)] = rows_by_key
            protocols[(root, dataset)]["baseline_protocol_sha256"] = sha256_file(
                baseline / dataset / "protocol.json"
            )
        return root, baseline, protocols, records

    def test_complete_summary_preserves_negative_ablation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, baseline, protocols, records = self.fixture(temporary)
            with patch("analysis.summarize_botdmm_modules.verified_records",
                       side_effect=lambda source, dataset: (
                           protocols[(source, dataset)], records[(source, dataset)]
                       )):
                report = summarize(root, baseline, Path(temporary) / "summary", 100)
            self.assertFalse(report["submission_allowed"])
            for dataset in ("Twibot22", "Quadbot"):
                self.assertEqual(report["datasets"][dataset]["runs_verified"], 18)
                delta = report["datasets"][dataset]["paired_intervals"]["burst"][
                    "aggregate_comparisons"
                ]["botdmm_without_structure_minus_botdmm_timing"]["macro_f1"]["estimate"]
                self.assertLess(delta, 0)

    def test_changed_graph_context_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, baseline, protocols, records = self.fixture(temporary)
            records[(root, "Twibot22")][(42, "burst", "botdmm_without_structure")][
                "graph_challenge"
            ] = {"hash": "different"}
            with patch("analysis.summarize_botdmm_modules.verified_records",
                       side_effect=lambda source, dataset: (
                           protocols[(source, dataset)], records[(source, dataset)]
                       )):
                with self.assertRaises(ValueError):
                    summarize(root, baseline, Path(temporary) / "summary", 100)


if __name__ == "__main__":
    unittest.main()
