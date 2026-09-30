#!/usr/bin/env python3
"""Audit and summarize the six corrected strict CBAF-Net reruns.

The tool deliberately treats prediction JSONL files as the numeric source of
truth.  It recomputes every reported metric, verifies the training/evaluation
protocol records, hashes the referenced checkpoints, and emits deterministic
machine-readable inputs for the manuscript tables and figures.

Typical usage from the repository root::

    python analysis/postprocess_strict_rerun.py \
        --results-root experiments/strict_cbaf \
        --training-root experiments/strict_backbones \
        --ablation-root experiments/strict_cbaf_ablations \
        --output-dir results/strict_postprocessed

``--result-dir`` and ``--training-summary`` may be repeated instead of using
the discovery roots.  Exactly the two datasets times three seeds are required.
Ablations are optional and are reported as unavailable when their prediction
artifacts do not exist; the tool never invents replacement values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.bse_config import configuration_mismatches, estimator_configuration


DATASETS: Mapping[str, int] = {"Twibot22": 3, "Quadbot": 4}
SEEDS: Tuple[int, ...] = (42, 43, 44)
METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mcc",
)
SOURCE_METRIC_KEYS: Mapping[str, str] = {
    "accuracy": "accuracy",
    "macro_precision": "precision",
    "macro_recall": "recall",
    "macro_f1": "macro_f1",
    "mcc": "mcc",
}
PROBABILITY_FIELDS: Mapping[str, str] = {
    "BotDMM": "baseline_prob",
    "BSE-only": "bse_prob",
    "CBAF-Net": "fused_prob",
}
EXPECTED_GRAPH_PROTOCOL: Mapping[str, object] = {
    "edge_policy": "strict_split_local_knn",
    "knn_k": 10,
    "batching": "one_hop_target_neighbor",
    "message_direction": "selected_neighbor_to_target",
}
EXPECTED_CHECKPOINT_FORMAT = "cbaf-botdmm-checkpoint-v2"
EXPECTED_PREPROCESSING_VERSION = "train-column-zscore-v1"
SCHEMA_VERSION = "cbaf-strict-postprocess-v1"
BOOTSTRAP_SCHEMA_VERSION = "cbaf-profile-cluster-bootstrap-v1"


class AuditError(ValueError):
    """Raised when an input artifact violates the strict rerun contract."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def ordered_identity_hashes(data: Mapping[str, Sequence[object]]) -> dict:
    labels = [int(value) for value in data["labels"]]
    sample_ids = [str(value) for value in data["sample_ids"]]
    profile_ids = [str(value) for value in data["base_profile_ids"]]
    if not (len(labels) == len(sample_ids) == len(profile_ids)):
        raise AuditError("Split identity fields have different lengths")
    joint = [
        [label, sample_id, profile_id]
        for label, sample_id, profile_id in zip(labels, sample_ids, profile_ids)
    ]
    return {
        "row_count": len(labels),
        "ordered_labels_sha256": sha256_json(labels),
        "ordered_sample_ids_sha256": sha256_json(sample_ids),
        "ordered_base_profile_ids_sha256": sha256_json(profile_ids),
        "ordered_joint_identity_sha256": sha256_json(joint),
    }


def looks_like_absolute_path(value: str) -> bool:
    value = value.strip()
    if value.startswith(("/", "\\\\")):
        return True
    return len(value) >= 3 and value[1] == ":" and value[2] in ("/", "\\")


def assert_no_absolute_paths(value: object, context: str) -> None:
    if isinstance(value, str):
        if looks_like_absolute_path(value):
            raise AuditError(f"Public artifact contains an absolute path in {context}: {value!r}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_absolute_paths(child, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_absolute_paths(child, f"{context}[{index}]")


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AuditError(f"Missing JSON artifact: {path}") from None
    except json.JSONDecodeError as exc:
        raise AuditError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"Expected a JSON object in {path}")
    return payload


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise AuditError(f"Missing prediction artifact: {path}") from None
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise AuditError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise AuditError(f"No prediction rows found in {path}")
    return rows


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def validate_probability_vector(
    value: object,
    class_count: int,
    path: Path,
    row_number: int,
    field: str,
) -> List[float]:
    if not isinstance(value, list) or len(value) != class_count:
        raise AuditError(
            f"{path}:{row_number} has invalid {field}; expected {class_count} values"
        )
    probabilities = [float(item) for item in value]
    if any(not math.isfinite(item) for item in probabilities):
        raise AuditError(f"{path}:{row_number} has non-finite {field}")
    if any(item < -1e-8 or item > 1.0 + 1e-8 for item in probabilities):
        raise AuditError(f"{path}:{row_number} has out-of-range {field}")
    if not _close(sum(probabilities), 1.0, tolerance=2e-6):
        raise AuditError(f"{path}:{row_number} has non-normalized {field}")
    return probabilities


def confusion_matrix(
    labels: Sequence[int], predictions: Sequence[int], class_count: int
) -> List[List[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for label, prediction in zip(labels, predictions):
        if label < 0 or label >= class_count:
            raise AuditError(f"Ground-truth label {label} is outside 0..{class_count - 1}")
        if prediction < 0 or prediction >= class_count:
            raise AuditError(f"Prediction {prediction} is outside 0..{class_count - 1}")
        matrix[label][prediction] += 1
    return matrix


def metrics_from_predictions(
    labels: Sequence[int], predictions: Sequence[int], class_count: int
) -> dict:
    if not labels or len(labels) != len(predictions):
        raise AuditError("Labels and predictions must be non-empty and equally sized")
    matrix = confusion_matrix(labels, predictions, class_count)
    total = len(labels)
    true_positive = [matrix[index][index] for index in range(class_count)]
    true_support = [sum(row) for row in matrix]
    predicted_support = [
        sum(matrix[row][column] for row in range(class_count))
        for column in range(class_count)
    ]
    precision = [
        true_positive[index] / predicted_support[index]
        if predicted_support[index]
        else 0.0
        for index in range(class_count)
    ]
    recall = [
        true_positive[index] / true_support[index] if true_support[index] else 0.0
        for index in range(class_count)
    ]
    f1 = [
        2.0 * precision[index] * recall[index] / (precision[index] + recall[index])
        if precision[index] + recall[index]
        else 0.0
        for index in range(class_count)
    ]
    correct = sum(true_positive)
    numerator = correct * total - sum(
        predicted_support[index] * true_support[index]
        for index in range(class_count)
    )
    denominator_left = total * total - sum(value * value for value in predicted_support)
    denominator_right = total * total - sum(value * value for value in true_support)
    denominator = math.sqrt(max(0, denominator_left) * max(0, denominator_right))
    mcc = numerator / denominator if denominator else 0.0
    return {
        "accuracy": correct / total,
        "macro_precision": fmean(precision),
        "macro_recall": fmean(recall),
        "macro_f1": fmean(f1),
        "mcc": mcc,
        "confusion_matrix": matrix,
        "support": total,
        "class_support": true_support,
    }


def metrics_from_probabilities(
    labels: Sequence[int], probabilities: Sequence[Sequence[float]], class_count: int
) -> dict:
    return metrics_from_predictions(
        labels,
        [_argmax(row) for row in probabilities],
        class_count,
    )


def select_fusion_weight(
    labels: Sequence[int],
    baseline_probabilities: Sequence[Sequence[float]],
    bse_probabilities: Sequence[Sequence[float]],
    class_count: int,
) -> float:
    """Reproduce the validation Macro-F1 grid search and smaller-weight tie break."""
    best_candidate = None
    best_weight = None
    for step in range(21):
        weight = step / 20.0
        fused = [
            [
                (1.0 - weight) * base_value + weight * bse_value
                for base_value, bse_value in zip(base_row, bse_row)
            ]
            for base_row, bse_row in zip(baseline_probabilities, bse_probabilities)
        ]
        score = metrics_from_probabilities(labels, fused, class_count)["macro_f1"]
        candidate = (float(score), -weight, weight)
        if best_candidate is None or candidate > best_candidate:
            best_candidate = candidate
            best_weight = weight
    if best_weight is None:  # pragma: no cover - guarded by the non-empty metric contract
        raise AuditError("Fusion-weight selection received no validation examples")
    return best_weight


def assert_metrics_match(recomputed: Mapping[str, object], recorded: object, context: str) -> None:
    if not isinstance(recorded, dict):
        raise AuditError(f"Missing recorded metric block for {context}")
    mismatches = {}
    for output_key, source_key in SOURCE_METRIC_KEYS.items():
        if source_key not in recorded:
            mismatches[output_key] = {"actual": None, "expected": recomputed[output_key]}
            continue
        if not _close(float(recorded[source_key]), float(recomputed[output_key]), 2e-9):
            mismatches[output_key] = {
                "actual": recorded[source_key],
                "expected": recomputed[output_key],
            }
    if recorded.get("confusion_matrix") != recomputed["confusion_matrix"]:
        mismatches["confusion_matrix"] = {
            "actual": recorded.get("confusion_matrix"),
            "expected": recomputed["confusion_matrix"],
        }
    if mismatches:
        raise AuditError(f"Metric mismatch for {context}: {mismatches}")


def validate_prediction_rows(
    rows: Sequence[dict],
    path: Path,
    class_count: int,
    fusion_weight: float,
) -> dict:
    labels: List[int] = []
    sample_ids: List[str] = []
    base_profile_ids: List[str] = []
    probabilities: MutableMapping[str, List[List[float]]] = {
        key: [] for key in PROBABILITY_FIELDS.values()
    }
    seen_sample_ids = set()
    for offset, row in enumerate(rows):
        row_number = offset + 1
        if int(row.get("row_index", -1)) != offset:
            raise AuditError(f"{path}:{row_number} has a non-contiguous row_index")
        sample_id = row.get("sample_id")
        if sample_id is None or sample_id in seen_sample_ids:
            raise AuditError(f"{path}:{row_number} has a missing or duplicate sample_id")
        seen_sample_ids.add(sample_id)
        base_profile_id = row.get("base_profile_id")
        if base_profile_id is None:
            raise AuditError(f"{path}:{row_number} has a missing base_profile_id")
        sample_ids.append(str(sample_id))
        base_profile_ids.append(str(base_profile_id))
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError):
            raise AuditError(f"{path}:{row_number} has an invalid label") from None
        if label < 0 or label >= class_count:
            raise AuditError(f"{path}:{row_number} has label outside 0..{class_count - 1}")
        labels.append(label)

        for field in PROBABILITY_FIELDS.values():
            probabilities[field].append(
                validate_probability_vector(row.get(field), class_count, path, row_number, field)
            )
        baseline = probabilities["baseline_prob"][-1]
        bse = probabilities["bse_prob"][-1]
        fused = probabilities["fused_prob"][-1]
        expected_fused = [
            (1.0 - fusion_weight) * base_value + fusion_weight * bse_value
            for base_value, bse_value in zip(baseline, bse)
        ]
        if any(not _close(actual, expected, 2e-6) for actual, expected in zip(fused, expected_fused)):
            raise AuditError(
                f"{path}:{row_number} fused_prob does not match fusion_weight_bse={fusion_weight}"
            )
        expected_predictions = {
            "baseline_pred": _argmax(baseline),
            "bse_pred": _argmax(bse),
            "pred": _argmax(fused),
        }
        for field, expected in expected_predictions.items():
            if int(row.get(field, -1)) != expected:
                raise AuditError(f"{path}:{row_number} has inconsistent {field}")

    present_labels = sorted(set(labels))
    expected_labels = list(range(class_count))
    if present_labels != expected_labels:
        raise AuditError(f"{path} labels are {present_labels}; expected {expected_labels}")
    return {
        "labels": labels,
        "sample_ids": sample_ids,
        "base_profile_ids": base_profile_ids,
        **probabilities,
    }


def extract_identity(payload: Mapping[str, object], path: Path) -> Tuple[str, int]:
    dataset = payload.get("dataset")
    try:
        seed = int(payload.get("seed"))
    except (TypeError, ValueError):
        raise AuditError(f"Missing/invalid seed in {path}") from None
    if dataset not in DATASETS:
        raise AuditError(f"Unsupported dataset {dataset!r} in {path}")
    if seed not in SEEDS:
        raise AuditError(f"Unexpected seed {seed} in {path}; expected {SEEDS}")
    return str(dataset), seed


def discover_result_dirs(
    explicit: Sequence[Path], root: Optional[Path], feature_modes: Sequence[str]
) -> Dict[Tuple[str, int, str], Path]:
    candidates: List[Path] = [path.resolve() for path in explicit]
    if root is not None:
        candidates.extend(path.parent.resolve() for path in root.resolve().rglob("metrics.json"))
    discovered: Dict[Tuple[str, int, str], Path] = {}
    for directory in sorted(set(candidates)):
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            raise AuditError(f"Result directory lacks metrics.json: {directory}")
        payload = load_json(metrics_path)
        dataset, seed = extract_identity(payload, metrics_path)
        mode = str(payload.get("feature_mode", ""))
        if mode not in feature_modes:
            continue
        key = (dataset, seed, mode)
        if key in discovered and discovered[key] != directory:
            raise AuditError(f"Duplicate result directories for {key}: {discovered[key]} and {directory}")
        discovered[key] = directory
    return discovered


def discover_training_summaries(
    explicit: Sequence[Path], root: Optional[Path]
) -> Dict[Tuple[str, int], Path]:
    candidates: List[Path] = [path.resolve() for path in explicit]
    if root is not None:
        candidates.extend(path.resolve() for path in root.resolve().rglob("training_summary.json"))
    discovered: Dict[Tuple[str, int], Path] = {}
    for path in sorted(set(candidates)):
        payload = load_json(path)
        key = extract_identity(payload, path)
        if key in discovered and discovered[key] != path:
            raise AuditError(f"Duplicate training summaries for {key}: {discovered[key]} and {path}")
        discovered[key] = path
    return discovered


def require_complete_runs(mapping: Mapping[Tuple[str, int, str], Path], mode: str) -> None:
    expected = {(dataset, seed, mode) for dataset in DATASETS for seed in SEEDS}
    actual = {key for key in mapping if key[2] == mode}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AuditError(f"Incomplete {mode!r} result set; missing={missing}, extra={extra}")


def require_complete_training(mapping: Mapping[Tuple[str, int], Path]) -> None:
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AuditError(f"Incomplete training-summary set; missing={missing}, extra={extra}")


def _normalized_recorded_path(value: object) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def resolve_checkpoint(summary_path: Path, summary: Mapping[str, object]) -> Path:
    recorded = Path(str(summary.get("checkpoint", "best_model.pt")))
    candidates = []
    if recorded.is_absolute():
        candidates.append(recorded)
    candidates.extend(
        [
            summary_path.parent / recorded.name,
            Path.cwd() / recorded,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise AuditError(
        f"Cannot resolve checkpoint for {summary_path}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _declared_checkpoint_hashes(summary: Mapping[str, object], metrics: Mapping[str, object]) -> dict:
    declared = {}
    direct_candidates = {
        "training_summary.checkpoint_sha256": summary.get("checkpoint_sha256"),
        "metrics.checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "metrics.baseline_checkpoint_sha256": metrics.get("baseline_checkpoint_sha256"),
    }
    artifacts = summary.get("artifacts")
    if isinstance(artifacts, dict):
        direct_candidates["training_summary.artifacts.checkpoint_sha256"] = artifacts.get(
            "checkpoint_sha256"
        )
    for source, value in direct_candidates.items():
        if value is not None:
            normalized = str(value).strip().lower()
            if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
                raise AuditError(f"Invalid declared checkpoint SHA-256 at {source}: {value!r}")
            declared[source] = normalized
    return declared


def audit_bse_configuration(
    result_dir: Path,
    recorded: Mapping[str, object],
    seed: int,
) -> Tuple[dict, dict]:
    """Verify the serialized BSE and return its complete release configuration."""
    artifact = result_dir / "behavioral_statistics_expert.joblib"
    if not artifact.is_file():
        raise AuditError(f"Missing serialized BSE artifact: {artifact}")
    try:
        import joblib
        import sklearn

        estimator = joblib.load(artifact)
        actual = estimator_configuration(estimator, sklearn.__version__)
    except Exception as exc:
        raise AuditError(f"Cannot load and inspect serialized BSE {artifact}: {exc}") from exc

    mismatches = configuration_mismatches(actual, seed)
    if mismatches:
        raise AuditError(f"BSE release configuration mismatch in {artifact}: {mismatches}")

    declared = recorded.get("bse_configuration")
    if declared is not None and declared != actual:
        raise AuditError(
            f"Recorded BSE configuration differs from the serialized estimator in "
            f"{result_dir / 'metrics.json'}"
        )
    return actual, {
        "status": "verified",
        "source": "behavioral_statistics_expert.joblib",
        "configuration_record_present_in_metrics": declared is not None,
        "artifact_sha256": sha256_file(artifact),
    }


def validate_protocol_and_checkpoint(
    metrics_path: Path,
    metrics: Mapping[str, object],
    summary_path: Path,
    summary: Mapping[str, object],
    require_declared_checkpoint_hash: bool,
) -> dict:
    dataset, seed = extract_identity(metrics, metrics_path)
    if extract_identity(summary, summary_path) != (dataset, seed):
        raise AuditError(f"Training/evaluation identity mismatch for {dataset} seed {seed}")
    if metrics.get("method") != "CBAF-Net" or metrics.get("feature_mode") != "both":
        raise AuditError(f"{metrics_path} is not a main CBAF-Net/both result")
    if int(metrics.get("feature_dim", -1)) != 17:
        raise AuditError(f"Expected 17 BSE features in {metrics_path}")
    if metrics.get("checkpoint_protocol_verified") is not True:
        raise AuditError(f"{metrics_path} does not declare checkpoint_protocol_verified=true")

    graph_execution = metrics.get("graph_execution")
    if graph_execution != EXPECTED_GRAPH_PROTOCOL:
        raise AuditError(
            f"Graph protocol mismatch in {metrics_path}: {graph_execution!r}"
        )
    configuration = summary.get("configuration")
    if not isinstance(configuration, dict):
        raise AuditError(f"Missing configuration in {summary_path}")
    summary_graph = {key: configuration.get(key) for key in EXPECTED_GRAPH_PROTOCOL}
    if summary_graph != EXPECTED_GRAPH_PROTOCOL:
        raise AuditError(f"Graph protocol mismatch in {summary_path}: {summary_graph!r}")
    if configuration.get("checkpoint_format") != EXPECTED_CHECKPOINT_FORMAT:
        raise AuditError(f"Checkpoint format mismatch in {summary_path}")
    if configuration.get("objective") != "seed_node_cross_entropy":
        raise AuditError(f"Training objective mismatch in {summary_path}")

    preprocessing = configuration.get("tabular_preprocessing")
    if not isinstance(preprocessing, dict):
        raise AuditError(f"Missing tabular preprocessing record in {summary_path}")
    if preprocessing.get("version") != EXPECTED_PREPROCESSING_VERSION:
        raise AuditError(f"Tabular preprocessing version mismatch in {summary_path}")
    signature = str(preprocessing.get("signature_sha256", ""))
    if len(signature) != 64 or any(character not in "0123456789abcdef" for character in signature.lower()):
        raise AuditError(f"Invalid preprocessing signature in {summary_path}")
    result_preprocessing = metrics.get("backbone_tabular_preprocessing")
    if result_preprocessing != preprocessing:
        raise AuditError(f"Training/evaluation preprocessing records differ for {dataset} seed {seed}")
    envelope = metrics.get("checkpoint_envelope")
    if not isinstance(envelope, dict) or envelope.get("verified") is not True:
        raise AuditError(f"Unverified checkpoint envelope in {metrics_path}")
    if envelope.get("format_version") != EXPECTED_CHECKPOINT_FORMAT:
        raise AuditError(f"Checkpoint-envelope format mismatch in {metrics_path}")
    if envelope.get("tabular_preprocessing_signature") != signature:
        raise AuditError(f"Checkpoint-envelope preprocessing signature mismatch in {metrics_path}")

    edge_validation = summary.get("edge_validation")
    if not isinstance(edge_validation, dict):
        raise AuditError(f"Missing edge_validation in {summary_path}")
    for split in ("train", "val", "test"):
        report = edge_validation.get(split)
        if not isinstance(report, dict):
            raise AuditError(f"Missing {split} edge-validation report in {summary_path}")
        expected_k = int(EXPECTED_GRAPH_PROTOCOL["knn_k"])
        if int(report.get("min_degree", -1)) != expected_k:
            raise AuditError(f"{split} min_degree is not {expected_k} in {summary_path}")
        if int(report.get("max_degree", -1)) != expected_k:
            raise AuditError(f"{split} max_degree is not {expected_k} in {summary_path}")
        if int(report.get("same_profile_edge_count", -1)) != 0:
            raise AuditError(f"{split} contains same-profile edges in {summary_path}")

    checkpoint = resolve_checkpoint(summary_path, summary)
    summary_checkpoint = _normalized_recorded_path(summary.get("checkpoint"))
    metrics_checkpoint = _normalized_recorded_path(metrics.get("baseline_checkpoint"))
    actual_hash = sha256_file(checkpoint)
    declared_hashes = _declared_checkpoint_hashes(summary, metrics)
    if "checkpoint_sha256" in metrics and metrics.get("checkpoint_sha256_verified") is not True:
        raise AuditError(
            f"{metrics_path} declares checkpoint_sha256 without checkpoint_sha256_verified=true"
        )
    mismatched_hashes = {
        source: value for source, value in declared_hashes.items() if value != actual_hash
    }
    if mismatched_hashes:
        raise AuditError(
            f"Checkpoint SHA-256 mismatch for {dataset} seed {seed}: "
            f"actual={actual_hash}, declared={mismatched_hashes}"
        )
    if require_declared_checkpoint_hash:
        required_hash_sources = {
            "training_summary.checkpoint_sha256",
            "metrics.checkpoint_sha256",
        }
        missing_hash_sources = sorted(required_hash_sources - set(declared_hashes))
        if missing_hash_sources:
            raise AuditError(
                f"Missing required declared checkpoint hashes for {dataset} seed {seed}: "
                f"{missing_hash_sources}"
            )
    if summary_checkpoint != metrics_checkpoint and not declared_hashes:
        raise AuditError(
            f"Recorded checkpoint paths differ and no hash binds them for {dataset} seed {seed}: "
            f"{summary_checkpoint!r} vs {metrics_checkpoint!r}"
        )

    protocol_record = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint_format": EXPECTED_CHECKPOINT_FORMAT,
        "objective": configuration["objective"],
        "graph_protocol": dict(EXPECTED_GRAPH_PROTOCOL),
        "tabular_preprocessing_signature_sha256": signature,
    }
    return {
        "status": "verified",
        "protocol_sha256": sha256_json(protocol_record),
        "protocol": protocol_record,
        "checkpoint_artifact": f"training/{dataset}_seed{seed}/best_model.pt",
        "recorded_checkpoint_path_consistency": (
            "exact_match" if summary_checkpoint == metrics_checkpoint else "bound_by_sha256"
        ),
        "checkpoint_sha256": actual_hash,
        "checkpoint_hash_verification": (
            "verified_against_declared_hash" if declared_hashes else "computed_only_not_declared"
        ),
        "declared_checkpoint_hashes": declared_hashes,
        "training_summary_sha256": sha256_file(summary_path),
        "metrics_sha256": sha256_file(metrics_path),
    }


def _public_metric_block(metrics: Mapping[str, object]) -> dict:
    return {
        key: float(metrics[key]) for key in METRIC_KEYS
    } | {
        "confusion_matrix": metrics["confusion_matrix"],
        "support": int(metrics["support"]),
        "class_support": [int(value) for value in metrics["class_support"]],
    }


def audit_main_run(
    result_dir: Path,
    summary_path: Path,
    require_declared_checkpoint_hash: bool,
) -> Tuple[dict, dict]:
    metrics_path = result_dir / "metrics.json"
    recorded = load_json(metrics_path)
    summary = load_json(summary_path)
    dataset, seed = extract_identity(recorded, metrics_path)
    class_count = DATASETS[dataset]
    try:
        fusion_weight = float(recorded["fusion_weight_bse"])
    except (KeyError, TypeError, ValueError):
        raise AuditError(f"Invalid fusion_weight_bse in {metrics_path}") from None
    if not 0.0 <= fusion_weight <= 1.0:
        raise AuditError(f"fusion_weight_bse is outside [0, 1] in {metrics_path}")

    bse_configuration, bse_verification = audit_bse_configuration(
        result_dir,
        recorded,
        seed,
    )

    verification = validate_protocol_and_checkpoint(
        metrics_path,
        recorded,
        summary_path,
        summary,
        require_declared_checkpoint_hash,
    )
    verification["bse_configuration"] = bse_verification
    split_data = {}
    split_metrics = {}
    for split in ("val", "test"):
        prediction_path = result_dir / f"{split}_predictions.jsonl"
        rows = load_jsonl(prediction_path)
        data = validate_prediction_rows(
            rows,
            prediction_path,
            class_count,
            fusion_weight,
        )
        computed = {
            method: metrics_from_probabilities(
                data["labels"], data[field], class_count
            )
            for method, field in PROBABILITY_FIELDS.items()
        }
        assert_metrics_match(computed["CBAF-Net"], recorded.get(split), f"{dataset}/seed{seed}/{split}")
        if split == "test":
            assert_metrics_match(
                computed["BSE-only"],
                recorded.get("bse_only_test"),
                f"{dataset}/seed{seed}/bse_only_test",
            )
        split_data[split] = {"rows": rows, **data}
        split_metrics[split] = {
            method: _public_metric_block(value) for method, value in computed.items()
        }
        verification[f"{split}_predictions_sha256"] = sha256_file(prediction_path)
        verification[f"{split}_ordered_identity"] = ordered_identity_hashes(data)

    selected_weight = select_fusion_weight(
        split_data["val"]["labels"],
        split_data["val"]["baseline_prob"],
        split_data["val"]["bse_prob"],
        class_count,
    )
    if not _close(fusion_weight, selected_weight, 2e-9):
        raise AuditError(
            f"Validation-selected fusion weight mismatch for {dataset} seed {seed}: "
            f"recorded={fusion_weight}, recomputed={selected_weight}"
        )
    verification["validation_selected_fusion_weight_verified"] = True

    fixed_metrics = {}
    for split in ("val", "test"):
        data = split_data[split]
        fixed_probabilities = [
            [0.5 * base + 0.5 * bse for base, bse in zip(base_row, bse_row)]
            for base_row, bse_row in zip(data["baseline_prob"], data["bse_prob"])
        ]
        fixed_metrics[split] = _public_metric_block(
            metrics_from_probabilities(data["labels"], fixed_probabilities, class_count)
        )

    return (
        {
            "dataset": dataset,
            "seed": seed,
            "fusion_weight_bse": fusion_weight,
            "bse_configuration": bse_configuration,
            "verification": verification,
            "metrics": split_metrics,
            "fixed_weight_0_5": fixed_metrics,
        },
        split_data,
    )


def audit_ablation(
    mode: str,
    result_dir: Path,
    main_split_data: Mapping[str, Mapping[str, object]],
    dataset: str,
    seed: int,
    expected_checkpoint_sha256: str,
) -> dict:
    metrics_path = result_dir / "metrics.json"
    recorded = load_json(metrics_path)
    identity = extract_identity(recorded, metrics_path)
    if identity != (dataset, seed) or recorded.get("feature_mode") != mode:
        raise AuditError(f"Ablation identity/mode mismatch in {metrics_path}")
    expected_dimension = 6 if mode == "num" else 11
    if int(recorded.get("feature_dim", -1)) != expected_dimension:
        raise AuditError(
            f"Expected feature_dim={expected_dimension} for mode={mode} in {metrics_path}"
        )
    if recorded.get("method") != "CBAF-Net":
        raise AuditError(f"Ablation method mismatch in {metrics_path}")
    if recorded.get("checkpoint_protocol_verified") is not True:
        raise AuditError(f"Unverified checkpoint protocol in ablation {metrics_path}")
    if recorded.get("graph_execution") != EXPECTED_GRAPH_PROTOCOL:
        raise AuditError(f"Graph protocol mismatch in ablation {metrics_path}")
    bse_configuration, bse_verification = audit_bse_configuration(
        result_dir,
        recorded,
        seed,
    )
    declared_checkpoint_sha256 = recorded.get("checkpoint_sha256")
    if declared_checkpoint_sha256 is not None:
        if recorded.get("checkpoint_sha256_verified") is not True:
            raise AuditError(
                f"Ablation {metrics_path} declares a checkpoint hash without verified=true"
            )
        if str(declared_checkpoint_sha256).lower() != expected_checkpoint_sha256:
            raise AuditError(f"Ablation checkpoint hash differs from the main run in {metrics_path}")
    weight = float(recorded["fusion_weight_bse"])
    if not 0.0 <= weight <= 1.0:
        raise AuditError(f"Invalid ablation fusion weight in {metrics_path}")
    class_count = DATASETS[dataset]
    metrics_by_split = {}
    ablation_split_data = {}
    hashes = {"metrics_sha256": sha256_file(metrics_path)}
    for split in ("val", "test"):
        path = result_dir / f"{split}_predictions.jsonl"
        data = validate_prediction_rows(load_jsonl(path), path, class_count, weight)
        main = main_split_data[split]
        if data["labels"] != main["labels"]:
            raise AuditError(f"Ablation labels differ from the main run in {path}")
        if data["sample_ids"] != main["sample_ids"]:
            raise AuditError(f"Ablation sample order differs from the main run in {path}")
        if data["base_profile_ids"] != main["base_profile_ids"]:
            raise AuditError(f"Ablation profile order differs from the main run in {path}")
        for row_index, (actual, expected) in enumerate(
            zip(data["baseline_prob"], main["baseline_prob"])
        ):
            if any(not _close(a, b, 2e-6) for a, b in zip(actual, expected)):
                raise AuditError(
                    f"Ablation baseline probabilities differ at {path}:{row_index + 1}"
                )
        fused_metrics = metrics_from_probabilities(
            data["labels"], data["fused_prob"], class_count
        )
        assert_metrics_match(
            fused_metrics,
            recorded.get(split),
            f"{dataset}/seed{seed}/{mode}/{split}",
        )
        if split == "test":
            bse_only_metrics = metrics_from_probabilities(
                data["labels"], data["bse_prob"], class_count
            )
            assert_metrics_match(
                bse_only_metrics,
                recorded.get("bse_only_test"),
                f"{dataset}/seed{seed}/{mode}/bse_only_test",
            )
        metrics_by_split[split] = _public_metric_block(fused_metrics)
        ablation_split_data[split] = data
        hashes[f"{split}_predictions_sha256"] = sha256_file(path)
    selected_weight = select_fusion_weight(
        ablation_split_data["val"]["labels"],
        ablation_split_data["val"]["baseline_prob"],
        ablation_split_data["val"]["bse_prob"],
        class_count,
    )
    if not _close(weight, selected_weight, 2e-9):
        raise AuditError(
            f"Validation-selected ablation weight mismatch for {dataset} seed {seed} "
            f"mode={mode}: recorded={weight}, recomputed={selected_weight}"
        )
    return {
        "status": "available",
        "feature_mode": mode,
        "fusion_weight_bse": weight,
        "bse_configuration": bse_configuration,
        "bse_configuration_verification": bse_verification,
        "validation_selected_fusion_weight_verified": True,
        "metrics": metrics_by_split,
        "artifact_hashes": hashes,
        "result_artifact": f"ablations/{dataset}_seed{seed}_{mode}",
    }


def aggregate_metric_blocks(blocks: Sequence[Mapping[str, object]]) -> dict:
    if not blocks:
        raise AuditError("Cannot aggregate an empty metric sequence")
    result = {"n_seeds": len(blocks), "ddof": 0, "statistics": {}}
    for key in METRIC_KEYS:
        values = [float(block[key]) for block in blocks]
        result["statistics"][key] = {
            "mean": fmean(values),
            "population_sd": pstdev(values),
            "values_by_seed_order": values,
        }
    return result


def reliability_metrics(
    labels: Sequence[int], probabilities: Sequence[Sequence[float]], n_bins: int
) -> dict:
    counts = [0] * n_bins
    correct = [0] * n_bins
    confidence_sum = [0.0] * n_bins
    total_correct = 0
    for label, row in zip(labels, probabilities):
        prediction = _argmax(row)
        confidence = float(row[prediction])
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        counts[bin_index] += 1
        is_correct = int(prediction == label)
        correct[bin_index] += is_correct
        confidence_sum[bin_index] += confidence
        total_correct += is_correct
    bin_accuracy = [
        correct[index] / counts[index] if counts[index] else None
        for index in range(n_bins)
    ]
    bin_confidence = [
        confidence_sum[index] / counts[index] if counts[index] else None
        for index in range(n_bins)
    ]
    total = len(labels)
    gaps = [
        abs(float(bin_accuracy[index]) - float(bin_confidence[index]))
        if counts[index]
        else 0.0
        for index in range(n_bins)
    ]
    ece = sum(counts[index] / total * gaps[index] for index in range(n_bins))
    return {
        "accuracy": total_correct / total,
        "ece": ece,
        "mce": max(gaps),
        "mean_confidence": sum(confidence_sum) / total,
        "bin_edges": [index / n_bins for index in range(n_bins + 1)],
        "bin_counts": counts,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
    }


def build_confusion_input(all_split_data: Mapping[Tuple[str, int], Mapping[str, object]]) -> dict:
    payload = {
        "description": "Seed-wise and three-seed mean row-normalized test confusion matrices.",
        "normalization": "row_by_true_class_then_mean_across_seeds",
        "datasets": {},
    }
    for dataset, class_count in DATASETS.items():
        dataset_payload = {}
        for method, probability_field in PROBABILITY_FIELDS.items():
            by_seed = {}
            normalized_matrices = []
            for seed in SEEDS:
                data = all_split_data[(dataset, seed)]["test"]
                predictions = [_argmax(row) for row in data[probability_field]]
                counts = confusion_matrix(data["labels"], predictions, class_count)
                normalized = []
                for row in counts:
                    support = sum(row)
                    if not support:
                        raise AuditError(f"Zero class support for {dataset} seed {seed}")
                    normalized.append([value / support for value in row])
                normalized_matrices.append(normalized)
                by_seed[str(seed)] = {"counts": counts, "row_normalized": normalized}
            mean_matrix = [
                [
                    fmean(matrix[row][column] for matrix in normalized_matrices)
                    for column in range(class_count)
                ]
                for row in range(class_count)
            ]
            dataset_payload[method] = {
                "by_seed": by_seed,
                "mean_row_normalized": mean_matrix,
            }
        payload["datasets"][dataset] = dataset_payload
    return payload


def build_reliability_input(
    all_split_data: Mapping[Tuple[str, int], Mapping[str, object]], n_bins: int
) -> dict:
    payload = {
        "description": "Test-set confidence calibration from frozen per-seed predictions.",
        "n_bins": n_bins,
        "aggregation_sd": "population (ddof=0)",
        "datasets": {},
    }
    for dataset in DATASETS:
        dataset_payload = {}
        for method, probability_field in PROBABILITY_FIELDS.items():
            by_seed = {}
            for seed in SEEDS:
                data = all_split_data[(dataset, seed)]["test"]
                by_seed[str(seed)] = reliability_metrics(
                    data["labels"], data[probability_field], n_bins
                )
            aggregate = {}
            for key in ("accuracy", "ece", "mce", "mean_confidence"):
                values = [float(by_seed[str(seed)][key]) for seed in SEEDS]
                aggregate[key] = {
                    "mean": fmean(values),
                    "population_sd": pstdev(values),
                }
            dataset_payload[method] = {"by_seed": by_seed, "aggregate": aggregate}
        payload["datasets"][dataset] = dataset_payload
    return payload


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise AuditError("Cannot compute a percentile from no bootstrap values")
    if not 0.0 <= quantile <= 1.0:
        raise AuditError(f"Invalid quantile: {quantile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_interval(estimate: float, values: Sequence[float]) -> dict:
    return {
        "estimate": float(estimate),
        "lower_95": percentile(values, 0.025),
        "upper_95": percentile(values, 0.975),
    }


def metric_values(block: Mapping[str, object]) -> dict:
    return {key: float(block[key]) for key in METRIC_KEYS}


def profile_groups(data: Mapping[str, Sequence[object]]) -> Tuple[List[str], Dict[str, List[int]], dict]:
    groups: Dict[str, List[int]] = {}
    seen_pairs = set()
    for index, (profile_id, label) in enumerate(
        zip(data["base_profile_ids"], data["labels"])
    ):
        profile_id = str(profile_id)
        pair = (profile_id, int(label))
        if pair in seen_pairs:
            raise AuditError(f"Duplicate base_profile_id/label pair in bootstrap input: {pair}")
        seen_pairs.add(pair)
        groups.setdefault(profile_id, []).append(index)
    # First-occurrence order survives the public sequential-ID rewrite, so the
    # same RNG seed reproduces identical resamples from the sanitized files.
    names = list(groups)
    if not names:
        raise AuditError("Profile-cluster bootstrap input contains no profiles")
    size_counts: Dict[str, int] = {}
    for indices in groups.values():
        key = str(len(indices))
        size_counts[key] = size_counts.get(key, 0) + 1
    return names, groups, dict(sorted(size_counts.items(), key=lambda item: int(item[0])))


def derived_bootstrap_seed(base_seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_profile_cluster_bootstrap(
    all_split_data: Mapping[Tuple[str, int], Mapping[str, object]],
    replicates: int,
    base_seed: int,
) -> dict:
    """Build synchronized profile-cluster percentile intervals.

    For a dataset, one sampled profile sequence is applied to all methods and
    all three training seeds in each replicate. Aggregate intervals summarize
    the mean of the three seed-wise metrics for every synchronized replicate;
    the three fixed seeds are not themselves resampled.
    """
    if replicates < 100:
        raise AuditError("Profile-cluster bootstrap requires at least 100 replicates")
    output = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "split": "test",
        "confidence_interval": {
            "method": "percentile",
            "confidence_level": 0.95,
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
        },
        "sampling": {
            "unit": "base_profile_id",
            "draws_per_replicate": "number of distinct profiles in the test split",
            "with_replacement": True,
            "method_pairing": "shared sampled profile sequence across BotDMM, BSE-only, and CBAF-Net",
            "seed_pairing": "shared sampled profile sequence across seeds 42, 43, and 44",
            "aggregate_estimand": "mean of the three fixed-seed metrics per synchronized profile replicate",
            "training_seeds_resampled": False,
        },
        "bootstrap_replicates": int(replicates),
        "base_random_seed": int(base_seed),
        "datasets": {},
    }
    delta_name = "CBAF-Net_minus_BotDMM"
    for dataset, class_count in DATASETS.items():
        reference = all_split_data[(dataset, SEEDS[0])]["test"]
        group_names, groups, size_counts = profile_groups(reference)
        predictions: Dict[int, Dict[str, List[int]]] = {}
        observed: Dict[int, Dict[str, dict]] = {}
        for seed in SEEDS:
            data = all_split_data[(dataset, seed)]["test"]
            predictions[seed] = {
                method: [_argmax(row) for row in data[field]]
                for method, field in PROBABILITY_FIELDS.items()
            }
            observed[seed] = {
                method: metric_values(
                    metrics_from_predictions(
                        data["labels"], predictions[seed][method], class_count
                    )
                )
                for method in PROBABILITY_FIELDS
            }
            observed[seed][delta_name] = {
                key: observed[seed]["CBAF-Net"][key] - observed[seed]["BotDMM"][key]
                for key in METRIC_KEYS
            }

        distribution: Dict[int, Dict[str, Dict[str, List[float]]]] = {
            seed: {
                method: {key: [] for key in METRIC_KEYS}
                for method in (*PROBABILITY_FIELDS.keys(), delta_name)
            }
            for seed in SEEDS
        }
        aggregate_distribution: Dict[str, Dict[str, List[float]]] = {
            method: {key: [] for key in METRIC_KEYS}
            for method in (*PROBABILITY_FIELDS.keys(), delta_name)
        }
        rng_seed = derived_bootstrap_seed(base_seed, dataset)
        rng = random.Random(rng_seed)
        for _ in range(replicates):
            selected_names = [
                group_names[rng.randrange(len(group_names))]
                for _ in range(len(group_names))
            ]
            indices = [index for name in selected_names for index in groups[name]]
            current: Dict[int, Dict[str, dict]] = {}
            for seed in SEEDS:
                data = all_split_data[(dataset, seed)]["test"]
                sampled_labels = [int(data["labels"][index]) for index in indices]
                current[seed] = {}
                for method in PROBABILITY_FIELDS:
                    sampled_predictions = [predictions[seed][method][index] for index in indices]
                    values = metric_values(
                        metrics_from_predictions(
                            sampled_labels, sampled_predictions, class_count
                        )
                    )
                    current[seed][method] = values
                    for key in METRIC_KEYS:
                        distribution[seed][method][key].append(values[key])
                delta = {
                    key: current[seed]["CBAF-Net"][key] - current[seed]["BotDMM"][key]
                    for key in METRIC_KEYS
                }
                current[seed][delta_name] = delta
                for key in METRIC_KEYS:
                    distribution[seed][delta_name][key].append(delta[key])

            for method in (*PROBABILITY_FIELDS.keys(), delta_name):
                for key in METRIC_KEYS:
                    aggregate_distribution[method][key].append(
                        fmean(current[seed][method][key] for seed in SEEDS)
                    )

        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {
                method: {
                    key: bootstrap_interval(
                        observed[seed][method][key], distribution[seed][method][key]
                    )
                    for key in METRIC_KEYS
                }
                for method in (*PROBABILITY_FIELDS.keys(), delta_name)
            }
        aggregate_estimate = {
            method: {
                key: fmean(observed[seed][method][key] for seed in SEEDS)
                for key in METRIC_KEYS
            }
            for method in (*PROBABILITY_FIELDS.keys(), delta_name)
        }
        aggregate = {
            method: {
                key: bootstrap_interval(
                    aggregate_estimate[method][key],
                    aggregate_distribution[method][key],
                )
                for key in METRIC_KEYS
            }
            for method in (*PROBABILITY_FIELDS.keys(), delta_name)
        }
        output["datasets"][dataset] = {
            "trajectory_count": len(reference["labels"]),
            "profile_count": len(group_names),
            "records_per_profile": size_counts,
            "random_seed": rng_seed,
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
    return output


def equality_partition(values: Sequence[str]) -> List[int]:
    mapping: Dict[str, int] = {}
    result = []
    for value in values:
        value = str(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return result


def copy_prediction_inputs(
    main_results: Mapping[Tuple[str, int, str], Path],
    all_split_data: Mapping[Tuple[str, int], Mapping[str, object]],
    runs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> Tuple[Path, dict]:
    """Write public-safe prediction inputs with deterministic local IDs."""
    root = output_dir / "predictions"
    privacy_report = {
        "scheme": "dataset-local deterministic first-occurrence reindexing",
        "mapping_published": False,
        "cluster_membership_and_row_order_preserved": True,
        "datasets": {},
    }
    for dataset in DATASETS:
        sample_mapping: Dict[str, str] = {}
        profile_mapping: Dict[str, str] = {}
        for split in ("val", "test"):
            reference = all_split_data[(dataset, SEEDS[0])][split]
            for sample_id in reference["sample_ids"]:
                sample_id = str(sample_id)
                sample_mapping.setdefault(sample_id, f"sample_{len(sample_mapping):06d}")
            for profile_id in reference["base_profile_ids"]:
                profile_id = str(profile_id)
                profile_mapping.setdefault(profile_id, f"profile_{len(profile_mapping):06d}")

        split_reports = {}
        for split in ("val", "test"):
            reference = all_split_data[(dataset, SEEDS[0])][split]
            source_partition = equality_partition(reference["base_profile_ids"])
            public_profiles = [
                profile_mapping[str(value)] for value in reference["base_profile_ids"]
            ]
            public_partition = equality_partition(public_profiles)
            if public_partition != source_partition:
                raise AuditError(f"Public profile reindexing changed cluster membership for {dataset}/{split}")
            split_reports[split] = {
                "row_count": len(reference["labels"]),
                "profile_count": len(set(public_profiles)),
                "ordered_public_sample_ids_sha256": sha256_json(
                    [sample_mapping[str(value)] for value in reference["sample_ids"]]
                ),
                "ordered_public_profile_ids_sha256": sha256_json(public_profiles),
                "cluster_membership_order_sha256": sha256_json(public_partition),
            }

        for seed in SEEDS:
            source_dir = main_results[(dataset, seed, "both")]
            target_dir = root / f"{dataset}_seed{seed}"
            target_dir.mkdir(parents=True, exist_ok=True)
            metrics = load_json(source_dir / "metrics.json")
            metrics["baseline_checkpoint"] = f"training/{dataset}_seed{seed}/best_model.pt"
            metrics["checkpoint_protocol_record"] = (
                f"training/{dataset}_seed{seed}/training_summary.json"
            )
            metrics["bse_configuration"] = runs[f"{dataset}_seed{seed}"][
                "bse_configuration"
            ]
            metrics["public_identity_reindexing"] = {
                "sample_id": "dataset-local sequential identifier",
                "base_profile_id": "dataset-local sequential profile cluster identifier",
                "mapping_published": False,
            }
            assert_no_absolute_paths(metrics, f"public metrics {dataset}/seed{seed}")
            (target_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            for split in ("val", "test"):
                rows = all_split_data[(dataset, seed)][split]["rows"]
                output_path = target_dir / f"{split}_predictions.jsonl"
                with output_path.open("w", encoding="utf-8", newline="\n") as stream:
                    for row in rows:
                        public_row = dict(row)
                        public_row["sample_id"] = sample_mapping[str(row["sample_id"])]
                        public_row["base_profile_id"] = profile_mapping[
                            str(row["base_profile_id"])
                        ]
                        stream.write(
                            json.dumps(public_row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
        privacy_report["datasets"][dataset] = {
            "sample_id_count": len(sample_mapping),
            "profile_id_count": len(profile_mapping),
            "splits": split_reports,
        }
    return root, privacy_report


def write_table_csv(summary: Mapping[str, object], path: Path) -> None:
    rows = []
    for dataset, methods in summary["aggregates"].items():
        for method, aggregate in methods.items():
            if aggregate.get("status") != "available":
                rows.append({"dataset": dataset, "method": method, "status": aggregate["status"]})
                continue
            row = {
                "dataset": dataset,
                "method": method,
                "status": "available",
                "n_seeds": aggregate["n_seeds"],
            }
            for key in METRIC_KEYS:
                row[f"{key}_mean"] = aggregate["statistics"][key]["mean"]
                row[f"{key}_population_sd"] = aggregate["statistics"][key]["population_sd"]
            if "fusion_weight_bse" in aggregate:
                row["fusion_weight_bse_mean"] = aggregate["fusion_weight_bse"]["mean"]
                row["fusion_weight_bse_population_sd"] = aggregate["fusion_weight_bse"][
                    "population_sd"
                ]
            rows.append(row)
    fieldnames = [
        "dataset",
        "method",
        "status",
        "n_seeds",
        *[
            item
            for key in METRIC_KEYS
            for item in (f"{key}_mean", f"{key}_population_sd")
        ],
        "fusion_weight_bse_mean",
        "fusion_weight_bse_population_sd",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_existing_figures(
    predictions_root: Path,
    output_dir: Path,
    n_bins: int,
    aggregates: Mapping[str, object],
) -> dict:
    analysis_dir = Path(__file__).resolve().parent
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    try:
        import plot_confusion
        import plot_reliability
        import plot_ablation
    except ImportError as exc:
        raise AuditError(
            "Figure regeneration requires numpy and matplotlib from requirements.txt; "
            "use --skip-figures only when producing a data-only audit"
        ) from exc

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    confusion_stem = figure_dir / "cbaf_net_confusion"
    confusion = plot_confusion.aggregate_matrices(predictions_root)
    plot_confusion.save_figure(confusion, confusion_stem, 600)
    reliability_stem = figure_dir / "cbaf_net_reliability_confidence"
    curves, reliability_summary = plot_reliability.collect(predictions_root, n_bins)
    plot_reliability.build_figure(curves, reliability_summary, reliability_stem, n_bins)
    plot_reliability.write_summary(reliability_summary, reliability_stem, n_bins)
    artifacts = {
        "confusion_figure_stem": "figures/cbaf_net_confusion",
        "reliability_figure_stem": "figures/cbaf_net_reliability_confidence",
    }
    try:
        ablation_results = plot_ablation.results_from_aggregates(aggregates)
    except ValueError as exc:
        artifacts["ablation_figure_status"] = f"not_available: {exc}"
    else:
        ablation_stem = figure_dir / "cbaf_net_ablation"
        plot_ablation.save_figure(
            plot_ablation.build_figure(ablation_results),
            ablation_stem,
        )
        artifacts["ablation_figure_stem"] = "figures/cbaf_net_ablation"
    return artifacts


def process(args: argparse.Namespace) -> dict:
    main_results = discover_result_dirs(args.result_dir, args.results_root, ("both",))
    require_complete_runs(main_results, "both")
    training = discover_training_summaries(args.training_summary, args.training_root)
    require_complete_training(training)
    ablations = discover_result_dirs(
        args.ablation_result_dir,
        args.ablation_root,
        ("num", "style"),
    ) if args.ablation_result_dir or args.ablation_root else {}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    all_split_data = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            run, split_data = audit_main_run(
                main_results[(dataset, seed, "both")],
                training[(dataset, seed)],
                args.require_declared_checkpoint_hash,
            )
            run_key = f"{dataset}_seed{seed}"
            run["result_artifact"] = f"inputs/{dataset}_seed{seed}"
            run["training_summary_artifact"] = (
                f"training/{dataset}_seed{seed}/training_summary.json"
            )
            run["ablations"] = {}
            for mode, label in (
                ("num", "BSE-numerical-only fusion"),
                ("style", "BSE-style-only fusion"),
            ):
                ablation_dir = ablations.get((dataset, seed, mode))
                if ablation_dir is None:
                    run["ablations"][label] = {
                        "status": "not_available",
                        "reason": f"No {mode!r} result directory was supplied for this run.",
                    }
                else:
                    run["ablations"][label] = audit_ablation(
                        mode,
                        ablation_dir,
                        split_data,
                        dataset,
                        seed,
                        run["verification"]["checkpoint_sha256"],
                    )
            runs[run_key] = run
            all_split_data[(dataset, seed)] = split_data

    preprocessing_signatures = {}
    split_identity_contract = {}
    for dataset in DATASETS:
        signatures = {
            runs[f"{dataset}_seed{seed}"]["verification"]["protocol"][
                "tabular_preprocessing_signature_sha256"
            ]
            for seed in SEEDS
        }
        if len(signatures) != 1:
            raise AuditError(
                f"Training preprocessing signature changes across seeds for {dataset}: "
                f"{sorted(signatures)}"
            )
        preprocessing_signatures[dataset] = next(iter(signatures))
        split_identity_contract[dataset] = {}
        for split in ("val", "test"):
            by_seed = {
                str(seed): runs[f"{dataset}_seed{seed}"]["verification"][
                    f"{split}_ordered_identity"
                ]
                for seed in SEEDS
            }
            reference = by_seed[str(SEEDS[0])]
            for seed in SEEDS[1:]:
                if by_seed[str(seed)] != reference:
                    raise AuditError(
                        f"Cross-seed ordered split identity mismatch for "
                        f"{dataset}/{split}: seed {SEEDS[0]} vs seed {seed}"
                    )
            split_identity_contract[dataset][split] = {
                **reference,
                "verified_equal_across_seeds": list(SEEDS),
            }

    aggregates = {}
    for dataset in DATASETS:
        methods = {}
        for method in PROBABILITY_FIELDS:
            blocks = [runs[f"{dataset}_seed{seed}"]["metrics"]["test"][method] for seed in SEEDS]
            methods[method] = {"status": "available", **aggregate_metric_blocks(blocks)}
        fixed_blocks = [runs[f"{dataset}_seed{seed}"]["fixed_weight_0_5"]["test"] for seed in SEEDS]
        methods["CBAF-Net fixed w=0.5"] = {
            "status": "available",
            **aggregate_metric_blocks(fixed_blocks),
            "fusion_weight_bse": {"mean": 0.5, "population_sd": 0.0, "values": [0.5] * 3},
        }
        main_weights = [runs[f"{dataset}_seed{seed}"]["fusion_weight_bse"] for seed in SEEDS]
        methods["CBAF-Net"]["fusion_weight_bse"] = {
            "mean": fmean(main_weights),
            "population_sd": pstdev(main_weights),
            "values_by_seed_order": main_weights,
        }
        for label in ("BSE-numerical-only fusion", "BSE-style-only fusion"):
            records = [runs[f"{dataset}_seed{seed}"]["ablations"][label] for seed in SEEDS]
            if all(record["status"] == "available" for record in records):
                blocks = [record["metrics"]["test"] for record in records]
                weights = [float(record["fusion_weight_bse"]) for record in records]
                methods[label] = {
                    "status": "available",
                    **aggregate_metric_blocks(blocks),
                    "fusion_weight_bse": {
                        "mean": fmean(weights),
                        "population_sd": pstdev(weights),
                        "values_by_seed_order": weights,
                    },
                }
            else:
                available = [SEEDS[index] for index, record in enumerate(records) if record["status"] == "available"]
                methods[label] = {
                    "status": "not_available",
                    "reason": "A complete three-seed ablation was not supplied; no aggregate was computed.",
                    "available_seeds": available,
                }
        aggregates[dataset] = methods

    bootstrap = build_profile_cluster_bootstrap(
        all_split_data,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    bootstrap_path = output_dir / "profile_cluster_bootstrap.json"
    bootstrap_path.write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions_root, privacy_report = copy_prediction_inputs(
        main_results,
        all_split_data,
        runs,
        output_dir,
    )
    confusion_input = build_confusion_input(all_split_data)
    reliability_input = build_reliability_input(all_split_data, args.bins)
    confusion_path = output_dir / "confusion_input.json"
    reliability_path = output_dir / "reliability_input.json"
    confusion_path.write_text(json.dumps(confusion_input, indent=2), encoding="utf-8")
    reliability_path.write_text(json.dumps(reliability_input, indent=2), encoding="utf-8")

    figure_artifacts = {}
    if not args.skip_figures:
        figure_artifacts = render_existing_figures(
            predictions_root,
            output_dir,
            args.bins,
            aggregates,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "datasets": dict(DATASETS),
            "seeds": list(SEEDS),
            "metric_sd": "population (ddof=0)",
            "numeric_source_of_truth": "validated val/test prediction JSONL artifacts",
            "graph_protocol": dict(EXPECTED_GRAPH_PROTOCOL),
            "checkpoint_format": EXPECTED_CHECKPOINT_FORMAT,
            "preprocessing_version": EXPECTED_PREPROCESSING_VERSION,
            "bse_configuration": {
                "verification": "serialized estimator parameters checked for every main and ablation run",
                "learning_rate": 0.1,
                "min_samples_leaf": 20,
                "max_iter": 300,
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "random_state": "training seed",
            },
            "preprocessing_signature_sha256_by_dataset": preprocessing_signatures,
            "ordered_split_identity_sha256_by_dataset": split_identity_contract,
            "declared_checkpoint_hash_required": args.require_declared_checkpoint_hash,
            "bootstrap": {
                "replicates": args.bootstrap_replicates,
                "base_random_seed": args.bootstrap_seed,
                "sampling_unit": "base_profile_id",
                "aggregate_estimand": (
                    "mean of seeds 42/43/44 for each synchronized profile-cluster replicate"
                ),
            },
        },
        "verification": {
            "status": "passed",
            "main_run_count": len(runs),
            "all_metrics_recomputed": True,
            "all_protocol_records_verified": True,
            "all_checkpoint_files_hashed": True,
            "cross_seed_split_identity_verified": True,
            "public_prediction_ids_reindexed": True,
            "note": (
                "A checkpoint hash marked computed_only_not_declared proves the file hashed during "
                "postprocessing, but not agreement with an independently recorded pre-run digest."
            ),
        },
        "runs": runs,
        "aggregates": aggregates,
        "profile_cluster_bootstrap_aggregate": {
            dataset: bootstrap["datasets"][dataset]["aggregate"] for dataset in DATASETS
        },
        "privacy": privacy_report,
        "artifacts": {
            "canonical_predictions_root": "predictions",
            "confusion_input": "confusion_input.json",
            "confusion_input_sha256": sha256_file(confusion_path),
            "reliability_input": "reliability_input.json",
            "reliability_input_sha256": sha256_file(reliability_path),
            "profile_cluster_bootstrap": "profile_cluster_bootstrap.json",
            "profile_cluster_bootstrap_sha256": sha256_file(bootstrap_path),
            **figure_artifacts,
        },
    }
    assert_no_absolute_paths(summary, "strict_rerun_summary")
    summary_path = output_dir / "strict_rerun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table_path = output_dir / "strict_rerun_table.csv"
    write_table_csv(summary, table_path)
    summary["artifacts"]["table_csv"] = "strict_rerun_table.csv"
    summary["artifacts"]["table_csv_sha256"] = sha256_file(table_path)
    assert_no_absolute_paths(summary, "strict_rerun_summary")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--result-dir", type=Path, action="append", default=[])
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--training-summary", type=Path, action="append", default=[])
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--ablation-result-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/strict_postprocessed"),
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
        help="Profile-cluster bootstrap replicates. Default: 5000.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260930,
        help="Base seed used to derive deterministic dataset-specific bootstrap streams.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--require-declared-checkpoint-hash",
        action="store_true",
        help="Fail unless metrics/training metadata contain a SHA-256 matching the checkpoint file.",
    )
    args = parser.parse_args(argv)
    if not args.result_dir and args.results_root is None:
        parser.error("provide --results-root or six --result-dir values")
    if not args.training_summary and args.training_root is None:
        parser.error("provide --training-root or six --training-summary values")
    if args.bins < 3:
        parser.error("--bins must be at least 3")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = process(args)
    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 2
    output = args.output_dir.resolve() / "strict_rerun_summary.json"
    print(f"Verified {summary['verification']['main_run_count']} strict CBAF-Net runs")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
