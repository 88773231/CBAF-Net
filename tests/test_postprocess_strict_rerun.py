import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from analysis import postprocess_strict_rerun as post
from src.bse_config import estimator_configuration, release_bse_parameters


def source_metric_block(block):
    return {
        "accuracy": block["accuracy"],
        "precision": block["macro_precision"],
        "recall": block["macro_recall"],
        "macro_f1": block["macro_f1"],
        "mcc": block["mcc"],
        "confusion_matrix": block["confusion_matrix"],
    }


class StrictPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results_root = self.root / "results"
        self.training_root = self.root / "training"
        self.ablation_root = self.root / "ablations"
        self.output_dir = self.root / "postprocessed"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _probability(class_count, preferred, confidence):
        remainder = (1.0 - confidence) / (class_count - 1)
        values = [remainder] * class_count
        values[preferred] = confidence
        return values

    def _prediction_rows(self, dataset, seed, split, mode="both"):
        class_count = post.DATASETS[dataset]
        rows = []
        for index in range(class_count * 2):
            label = index % class_count
            baseline_preferred = label
            if seed == 42 and index == 0:
                baseline_preferred = (label + 1) % class_count
            baseline = self._probability(class_count, baseline_preferred, 0.62)
            if mode == "num":
                bse_preferred = label if index != 1 else (label + 1) % class_count
                bse = self._probability(class_count, bse_preferred, 0.72)
            elif mode == "style":
                bse_preferred = label if index != 2 else (label + 1) % class_count
                bse = self._probability(class_count, bse_preferred, 0.74)
            else:
                bse = self._probability(class_count, label, 0.76)
            rows.append(
                {
                    "row_index": index,
                    "sample_id": f"{dataset}-{split}-{index}",
                    "base_profile_id": f"profile-{index // class_count}",
                    "label": label,
                    "baseline_prob": baseline,
                    "bse_prob": bse,
                }
            )
        weight = post.select_fusion_weight(
            [row["label"] for row in rows],
            [row["baseline_prob"] for row in rows],
            [row["bse_prob"] for row in rows],
            class_count,
        )
        for row in rows:
            baseline = row["baseline_prob"]
            bse = row["bse_prob"]
            fused = [
                (1.0 - weight) * base_value + weight * bse_value
                for base_value, bse_value in zip(baseline, bse)
            ]
            row["baseline_pred"] = post._argmax(baseline)
            row["bse_pred"] = post._argmax(bse)
            row["pred"] = post._argmax(fused)
            row["fused_prob"] = fused
        return rows, weight

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_result(self, directory, dataset, seed, mode="both"):
        directory.mkdir(parents=True, exist_ok=True)
        estimator = HistGradientBoostingClassifier(**release_bse_parameters(seed))
        joblib.dump(estimator, directory / "behavioral_statistics_expert.joblib")
        bse_configuration = estimator_configuration(estimator, sklearn.__version__)
        split_rows = {}
        split_metrics = {}
        weight = None
        for split in ("val", "test"):
            rows, weight = self._prediction_rows(dataset, seed, split, mode)
            split_rows[split] = rows
            self._write_jsonl(directory / f"{split}_predictions.jsonl", rows)
            labels = [row["label"] for row in rows]
            fused = [row["fused_prob"] for row in rows]
            split_metrics[split] = post.metrics_from_probabilities(
                labels, fused, post.DATASETS[dataset]
            )
        labels = [row["label"] for row in split_rows["test"]]
        bse = [row["bse_prob"] for row in split_rows["test"]]
        bse_metrics = post.metrics_from_probabilities(
            labels, bse, post.DATASETS[dataset]
        )
        checkpoint = self.training_root / f"{dataset}_seed{seed}" / "best_model.pt"
        preprocessing = self._preprocessing(dataset)
        metrics = {
            "dataset": dataset,
            "seed": seed,
            "method": "CBAF-Net",
            "feature_dim": {"both": 17, "num": 6, "style": 11}[mode],
            "feature_mode": mode,
            "bse_configuration": bse_configuration,
            "fusion_weight_bse": weight,
            "baseline_checkpoint": str(checkpoint),
            "checkpoint_sha256": post.sha256_file(checkpoint),
            "checkpoint_sha256_verified": True,
            "checkpoint_protocol_verified": True,
            "checkpoint_protocol_record": str(checkpoint.parent / "training_summary.json"),
            "checkpoint_envelope": {
                "format_version": post.EXPECTED_CHECKPOINT_FORMAT,
                "verified": True,
                "tabular_preprocessing_signature": preprocessing["signature_sha256"],
            },
            "backbone_tabular_preprocessing": preprocessing,
            "graph_execution": dict(post.EXPECTED_GRAPH_PROTOCOL),
            "val": source_metric_block(split_metrics["val"]),
            "test": source_metric_block(split_metrics["test"]),
            "bse_only_test": source_metric_block(bse_metrics),
            "prediction_artifacts": {
                "val": "val_predictions.jsonl",
                "test": "test_predictions.jsonl",
            },
        }
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _preprocessing(dataset):
        signature = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
        return {
            "version": post.EXPECTED_PREPROCESSING_VERSION,
            "fit_split": "train",
            "transform": "per_column_population_zscore",
            "zero_variance_policy": "map_to_zero",
            "clamp": None,
            "signature_sha256": signature,
            "num_prop": {
                "dimension": 6,
                "train_mean": [0.0] * 6,
                "train_population_std": [1.0] * 6,
                "zero_variance_indices": [],
            },
            "llm_features": {
                "dimension": 11,
                "train_mean": [0.0] * 11,
                "train_population_std": [1.0] * 11,
                "zero_variance_indices": [],
            },
        }

    def _write_training(self, dataset, seed):
        directory = self.training_root / f"{dataset}_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "best_model.pt"
        checkpoint.write_bytes(f"checkpoint:{dataset}:{seed}".encode("ascii"))
        checkpoint_hash = post.sha256_file(checkpoint)
        edge_report = {
            "edge_count": 100,
            "min_degree": 10,
            "max_degree": 10,
            "same_profile_edge_count": 0,
        }
        summary = {
            "dataset": dataset,
            "seed": seed,
            "device": "cuda",
            "best_epoch": 3,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "validation": {},
            "test": {},
            "configuration": {
                "optimizer": "AdamW",
                "objective": "seed_node_cross_entropy",
                **post.EXPECTED_GRAPH_PROTOCOL,
                "checkpoint_format": post.EXPECTED_CHECKPOINT_FORMAT,
                "tabular_preprocessing": self._preprocessing(dataset),
            },
            "edge_validation": {
                "train": dict(edge_report),
                "val": dict(edge_report),
                "test": dict(edge_report),
            },
            "history": [],
        }
        (directory / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def _build_fixture(self, include_ablations=False):
        for dataset in post.DATASETS:
            for seed in post.SEEDS:
                self._write_training(dataset, seed)
                self._write_result(
                    self.results_root / f"{dataset}_seed{seed}",
                    dataset,
                    seed,
                )
                if include_ablations:
                    for mode in ("num", "style"):
                        self._write_result(
                            self.ablation_root / f"{dataset}_seed{seed}_{mode}",
                            dataset,
                            seed,
                            mode,
                        )

    def _args(self, include_ablations=False):
        values = [
            "--results-root",
            str(self.results_root),
            "--training-root",
            str(self.training_root),
            "--output-dir",
            str(self.output_dir),
            "--skip-figures",
            "--require-declared-checkpoint-hash",
            "--bootstrap-replicates",
            "200",
            "--bootstrap-seed",
            "20260930",
        ]
        if include_ablations:
            values.extend(["--ablation-root", str(self.ablation_root)])
        return post.parse_args(values)

    def test_complete_fixture_emits_population_statistics_and_ablations(self):
        self._build_fixture(include_ablations=True)

        summary = post.process(self._args(include_ablations=True))

        self.assertEqual(summary["verification"]["status"], "passed")
        self.assertEqual(summary["verification"]["main_run_count"], 6)
        self.assertEqual(
            summary["runs"]["Twibot22_seed42"]["bse_configuration"]["parameters"]
            ["learning_rate"],
            0.1,
        )
        self.assertEqual(
            summary["runs"]["Twibot22_seed42"]["bse_configuration"]["parameters"]
            ["min_samples_leaf"],
            20,
        )
        aggregate = summary["aggregates"]["Twibot22"]["CBAF-Net"]
        self.assertEqual(aggregate["n_seeds"], 3)
        weights = aggregate["fusion_weight_bse"]["values_by_seed_order"]
        self.assertAlmostEqual(
            aggregate["fusion_weight_bse"]["mean"], sum(weights) / len(weights)
        )
        self.assertAlmostEqual(
            aggregate["fusion_weight_bse"]["population_sd"],
            math.sqrt(
                sum(
                    (weight - aggregate["fusion_weight_bse"]["mean"]) ** 2
                    for weight in weights
                )
                / len(weights)
            ),
        )
        self.assertEqual(
            summary["aggregates"]["Quadbot"]["BSE-numerical-only fusion"]["status"],
            "available",
        )
        self.assertTrue((self.output_dir / "strict_rerun_summary.json").is_file())
        self.assertTrue((self.output_dir / "strict_rerun_table.csv").is_file())
        self.assertTrue((self.output_dir / "confusion_input.json").is_file())
        self.assertTrue((self.output_dir / "reliability_input.json").is_file())
        self.assertTrue((self.output_dir / "profile_cluster_bootstrap.json").is_file())
        self.assertTrue(
            (self.output_dir / "predictions" / "Quadbot_seed44" / "test_predictions.jsonl").is_file()
        )
        public_path = (
            self.output_dir / "predictions" / "Twibot22_seed42" / "test_predictions.jsonl"
        )
        public_rows = [
            json.loads(line)
            for line in public_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        public_metrics = json.loads(
            (
                self.output_dir / "predictions" / "Twibot22_seed42" / "metrics.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            public_metrics["baseline_checkpoint"],
            "training/Twibot22_seed42/best_model.pt",
        )
        self.assertEqual(
            public_metrics["checkpoint_protocol_record"],
            "training/Twibot22_seed42/training_summary.json",
        )
        self.assertTrue(all(row["sample_id"].startswith("sample_") for row in public_rows))
        self.assertTrue(
            all(row["base_profile_id"].startswith("profile_") for row in public_rows)
        )
        self.assertNotIn("Twibot22-test-0", public_path.read_text(encoding="utf-8"))
        original_rows = [
            json.loads(line)
            for line in (
                self.results_root / "Twibot22_seed42" / "test_predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(
            post.equality_partition([row["base_profile_id"] for row in original_rows]),
            post.equality_partition([row["base_profile_id"] for row in public_rows]),
        )
        seed43_public = [
            json.loads(line)
            for line in (
                self.output_dir
                / "predictions"
                / "Twibot22_seed43"
                / "test_predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(
            [row["sample_id"] for row in public_rows],
            [row["sample_id"] for row in seed43_public],
        )
        self.assertEqual(
            [row["base_profile_id"] for row in public_rows],
            [row["base_profile_id"] for row in seed43_public],
        )
        summary_text = (self.output_dir / "strict_rerun_summary.json").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(str(self.root), summary_text)
        self.assertEqual(
            summary["contract"]["ordered_split_identity_sha256_by_dataset"]["Twibot22"]
            ["test"]["verified_equal_across_seeds"],
            [42, 43, 44],
        )
        aggregate_bootstrap = summary["profile_cluster_bootstrap_aggregate"]["Twibot22"]
        for key in post.METRIC_KEYS:
            self.assertAlmostEqual(
                aggregate_bootstrap["CBAF-Net_minus_BotDMM"][key]["estimate"],
                aggregate_bootstrap["CBAF-Net"][key]["estimate"]
                - aggregate_bootstrap["BotDMM"][key]["estimate"],
            )
        bootstrap = json.loads(
            (self.output_dir / "profile_cluster_bootstrap.json").read_text(encoding="utf-8")
        )
        for dataset in post.DATASETS:
            for seed in post.SEEDS:
                seed_block = bootstrap["datasets"][dataset]["per_seed"][str(seed)]
                self.assertEqual(
                    set(seed_block),
                    {"BotDMM", "BSE-only", "CBAF-Net", "CBAF-Net_minus_BotDMM"},
                )
            for method in ("BotDMM", "BSE-only", "CBAF-Net", "CBAF-Net_minus_BotDMM"):
                for key in post.METRIC_KEYS:
                    interval = bootstrap["datasets"][dataset]["aggregate"][method][key]
                    self.assertLessEqual(interval["lower_95"], interval["upper_95"])
        public_bootstrap_data = {}
        for dataset, class_count in post.DATASETS.items():
            for seed in post.SEEDS:
                directory = self.output_dir / "predictions" / f"{dataset}_seed{seed}"
                metrics = json.loads(
                    (directory / "metrics.json").read_text(encoding="utf-8")
                )
                path = directory / "test_predictions.jsonl"
                data = post.validate_prediction_rows(
                    post.load_jsonl(path),
                    path,
                    class_count,
                    float(metrics["fusion_weight_bse"]),
                )
                public_bootstrap_data[(dataset, seed)] = {"test": data}
        self.assertEqual(
            bootstrap,
            post.build_profile_cluster_bootstrap(
                public_bootstrap_data,
                200,
                20260930,
            ),
        )

    def test_multiclass_metrics_match_known_values(self):
        metrics = post.metrics_from_predictions(
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 1, 2, 0],
            3,
        )

        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["macro_precision"], (0.5 + 2 / 3 + 1.0) / 3)
        self.assertAlmostEqual(metrics["macro_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertAlmostEqual(metrics["mcc"], 12 / math.sqrt(22 * 24))

    def test_missing_ablations_are_disclosed_without_numeric_substitution(self):
        self._build_fixture(include_ablations=False)

        summary = post.process(self._args(include_ablations=False))

        missing = summary["aggregates"]["Twibot22"]["BSE-style-only fusion"]
        self.assertEqual(missing["status"], "not_available")
        self.assertNotIn("statistics", missing)
        per_run = summary["runs"]["Twibot22_seed42"]["ablations"][
            "BSE-style-only fusion"
        ]
        self.assertEqual(per_run["status"], "not_available")

    def test_checkpoint_hash_mismatch_is_rejected(self):
        self._build_fixture(include_ablations=False)
        checkpoint = self.training_root / "Twibot22_seed42" / "best_model.pt"
        checkpoint.write_bytes(b"tampered-after-summary")

        with self.assertRaisesRegex(post.AuditError, "Checkpoint SHA-256 mismatch"):
            post.process(self._args(include_ablations=False))

    def test_reported_metric_mismatch_is_rejected(self):
        self._build_fixture(include_ablations=False)
        metrics_path = self.results_root / "Quadbot_seed44" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["test"]["macro_f1"] += 0.01
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(post.AuditError, "Metric mismatch"):
            post.process(self._args(include_ablations=False))

    def test_bse_configuration_mismatch_is_rejected(self):
        self._build_fixture(include_ablations=False)
        metrics_path = self.results_root / "Twibot22_seed42" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["bse_configuration"]["parameters"]["learning_rate"] = 0.2
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with self.assertRaisesRegex(post.AuditError, "Recorded BSE configuration differs"):
            post.process(self._args(include_ablations=False))

    def test_legacy_metrics_configuration_is_recovered_from_serialized_bse(self):
        self._build_fixture(include_ablations=False)
        for metrics_path in self.results_root.rglob("metrics.json"):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload.pop("bse_configuration")
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        summary = post.process(self._args(include_ablations=False))

        verification = summary["runs"]["Quadbot_seed44"]["verification"]
        self.assertFalse(
            verification["bse_configuration"]["configuration_record_present_in_metrics"]
        )
        public_metrics = json.loads(
            (
                self.output_dir
                / "predictions"
                / "Quadbot_seed44"
                / "metrics.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            public_metrics["bse_configuration"]["parameters"]["learning_rate"],
            0.1,
        )

    def test_cross_seed_ordered_identity_mismatch_is_rejected(self):
        self._build_fixture(include_ablations=False)
        path = self.results_root / "Twibot22_seed43" / "test_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["sample_id"] = "different-private-sample-id"
        self._write_jsonl(path, rows)

        with self.assertRaisesRegex(post.AuditError, "Cross-seed ordered split identity mismatch"):
            post.process(self._args(include_ablations=False))

    def test_bootstrap_output_is_deterministic(self):
        self._build_fixture(include_ablations=False)
        post.process(self._args(include_ablations=False))
        first = (self.output_dir / "profile_cluster_bootstrap.json").read_bytes()

        second_output = self.root / "postprocessed-second"
        args = self._args(include_ablations=False)
        args.output_dir = second_output
        post.process(args)
        second = (second_output / "profile_cluster_bootstrap.json").read_bytes()

        self.assertEqual(first, second)

    def test_public_path_guard_rejects_windows_and_unix_absolute_paths(self):
        with self.assertRaisesRegex(post.AuditError, "absolute path"):
            post.assert_no_absolute_paths({"checkpoint": r"C:\\private\\best_model.pt"}, "x")
        with self.assertRaisesRegex(post.AuditError, "absolute path"):
            post.assert_no_absolute_paths({"checkpoint": "/root/private/best_model.pt"}, "x")


if __name__ == "__main__":
    unittest.main()
