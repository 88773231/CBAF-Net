"""Evaluate CBAF-Net from a frozen BotDMM checkpoint.

The tabular expert only consumes label-free exported numerical/style features.
The fusion weight is selected on the validation split and then frozen before
the hold-out test is reported.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.botdmm import BotDMM
from src.bse_config import estimator_configuration, release_bse_parameters
from src.data_loader import (
    QuadBotDataset,
    StrictOneHopNeighborCollator,
    Twibot22Dataset,
    fit_tabular_standardizers,
)
from src.provenance import sha256_file


CHECKPOINT_FORMAT_VERSION = "cbaf-botdmm-checkpoint-v2"


def normalize_dataset_name(dataset):
    """Map the public CLI names to canonical internal dataset IDs."""
    value = str(dataset).strip().lower()
    if value == "twibot22":
        return "twibot22"
    if value == "quadbot":
        return "quadbot"
    raise ValueError(f"Unsupported dataset: {dataset}")


def display_dataset_name(dataset):
    """Return the paper-facing dataset name used in result metadata."""
    return "Twibot22" if dataset == "twibot22" else "Quadbot"


def raw_tabular(data_dir, split, feature_mode="both"):
    num = torch.load(Path(data_dir) / "numerical" / f"{split}_num_properties_tensor.pt",
                     map_location="cpu", weights_only=True).numpy()
    style = torch.load(Path(data_dir) / "numerical" / f"{split}_llm_features.pt",
                       map_location="cpu", weights_only=True).numpy()
    if feature_mode == "num":
        features = num
    elif feature_mode == "style":
        features = style
    else:
        features = np.concatenate([num, style], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_split_metadata(data_dir, split):
    """Load row identities for post-hoc profile-cluster uncertainty audits."""
    path = Path(data_dir) / f"{split}_metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing split metadata required for audit: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows


def validate_declared_checkpoint_sha256(checkpoint, summary):
    """Require and verify the checkpoint digest declared by the training summary."""
    declared = summary.get("checkpoint_sha256")
    if declared is None:
        raise ValueError(
            "training summary is missing checkpoint_sha256; rerun src.train_botdmm "
            "or use --allow_unverified_checkpoint for diagnostics only"
        )
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("training summary contains an invalid checkpoint_sha256")
    try:
        decoded = bytes.fromhex(declared)
    except ValueError as exc:
        raise ValueError("training summary contains an invalid checkpoint_sha256") from exc
    if len(decoded) != 32:
        raise ValueError("training summary contains an invalid checkpoint_sha256")
    actual = sha256_file(checkpoint)
    if actual != declared.lower():
        raise ValueError(
            "checkpoint SHA-256 does not match the digest recorded in training_summary.json"
        )
    return actual


def validate_checkpoint_protocol(checkpoint, dataset, expected_k, preprocessing):
    """Reject checkpoints without the corrected strict graph-training record."""
    summary_path = checkpoint.parent / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"missing corrected training protocol record beside checkpoint: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_dataset = display_dataset_name(dataset)
    if summary.get("dataset") != expected_dataset:
        raise ValueError(
            f"checkpoint dataset is {summary.get('dataset')!r}, expected {expected_dataset!r}"
        )
    config = summary.get("configuration") or {}
    expected = {
        "edge_policy": "strict_split_local_knn",
        "batching": "one_hop_target_neighbor",
        "message_direction": "selected_neighbor_to_target",
        "knn_k": int(expected_k),
        "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint graph protocol mismatch: {mismatches}")
    recorded_preprocessing = config.get("tabular_preprocessing") or {}
    if recorded_preprocessing.get("signature_sha256") != preprocessing.get("signature_sha256"):
        raise ValueError(
            "checkpoint tabular preprocessing signature does not match the current training tensors"
        )
    if recorded_preprocessing.get("version") != preprocessing.get("version"):
        raise ValueError("checkpoint tabular preprocessing version mismatch")
    validate_declared_checkpoint_sha256(checkpoint, summary)
    return summary_path


def load_checkpoint_state(
    checkpoint,
    device,
    dataset,
    seed,
    num_classes,
    model_input_dimensions,
    preprocessing,
    expected_k,
    allow_unverified=False,
):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        if allow_unverified and isinstance(payload, dict):
            return payload, {"format_version": "legacy-raw-state-dict", "verified": False}
        raise ValueError(
            "checkpoint is not a verified CBAF-Net envelope; rerun src.train_botdmm "
            "or use --allow_unverified_checkpoint for diagnostics only"
        )

    expected_dataset = display_dataset_name(dataset)
    expected_graph = {
        "edge_policy": "strict_split_local_knn",
        "knn_k": int(expected_k),
        "batching": "one_hop_target_neighbor",
        "message_direction": "selected_neighbor_to_target",
    }
    checks = {
        "dataset": (payload.get("dataset"), expected_dataset),
        "seed": (payload.get("seed"), int(seed)),
        "num_classes": (payload.get("num_classes"), int(num_classes)),
        "model_input_dimensions": (
            payload.get("model_input_dimensions"),
            model_input_dimensions,
        ),
        "tabular_preprocessing_signature": (
            (payload.get("tabular_preprocessing") or {}).get("signature_sha256"),
            preprocessing.get("signature_sha256"),
        ),
        "graph_protocol": (payload.get("graph_protocol"), expected_graph),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint envelope mismatch: {mismatches}")
    if "state_dict" not in payload:
        raise ValueError("checkpoint envelope is missing state_dict")
    return payload["state_dict"], {
        "format_version": payload["format_version"],
        "verified": True,
        "tabular_preprocessing_signature": preprocessing["signature_sha256"],
    }


def baseline_probabilities(
    dataset,
    data_dir,
    checkpoint,
    device,
    expected_k=10,
    seed=42,
    tabular_standardizers=None,
    allow_unverified_checkpoint=False,
):
    dataset = normalize_dataset_name(dataset)
    dataset_cls = QuadBotDataset if dataset == "quadbot" else Twibot22Dataset
    num_classes = 4 if dataset == "quadbot" else 3
    train_ds = dataset_cls(
        data_dir,
        split="train",
        num_steps=5,
        tabular_standardizers=tabular_standardizers,
    )
    sample = train_ds[0]
    model_input_dimensions = {
        "description": int(sample["des"].shape[0]),
        "tweet": int(sample["tweets"][0].shape[0]),
        "event_action": int(sample["amrs"][0].shape[0]),
        "num_prop": int(sample["num_prop"].shape[0]),
        "llm_features": int(sample["llm_features"].shape[0]),
    }
    model = BotDMM(
        des_size=model_input_dimensions["description"],
        tweet_size=model_input_dimensions["tweet"],
        amr_size=model_input_dimensions["event_action"],
        num_prop_size=model_input_dimensions["num_prop"],
        llm_features_size=model_input_dimensions["llm_features"],
        embedding_dimension=128, feature_dim=128, num_temporal_steps=5,
        dropout=0.3, temperature=0.1, alpha=0.5,
        num_classes=num_classes, ablation_mode="base",
    )
    state, checkpoint_record = load_checkpoint_state(
        checkpoint,
        device,
        dataset,
        seed,
        num_classes,
        model_input_dimensions,
        train_ds.tabular_preprocessing,
        expected_k,
        allow_unverified=allow_unverified_checkpoint,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    outputs = {}
    for split in ("val", "test"):
        ds = dataset_cls(
            data_dir,
            split=split,
            num_steps=5,
            tabular_standardizers=train_ds.tabular_standardizers,
        )
        collator = StrictOneHopNeighborCollator(
            ds,
            expected_k=expected_k,
            require_strict=True,
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["des"].to(device),
                    [x.to(device) for x in batch["tweets"]],
                    [x.to(device) for x in batch["amrs"]],
                    batch["num_prop"].to(device),
                    batch["llm_features"].to(device),
                    [x.to(device) for x in batch["edge_indices"]],
                    compute_auxiliary_losses=False,
                )["logits"][batch["seed_positions"].to(device)]
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.append(batch["seed_labels"].numpy())
        outputs[split] = (np.concatenate(probs), np.concatenate(labels))
    return outputs, checkpoint_record


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, average="macro", zero_division=0)),
        "recall": float(recall_score(y, p, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "mcc": float(matthews_corrcoef(y, p)),
        "confusion_matrix": __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y, p).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        metavar="{Twibot22,Quadbot}",
        required=True,
        help=(
            "Public dataset name. Twibot22 and Quadbot are canonical; "
            "case-insensitive legacy spellings remain accepted."
        ),
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Frozen BotDMM checkpoint for the selected dataset and seed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--feature_mode", choices=["both", "num", "style"], default="both")
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument(
        "--allow_unverified_checkpoint",
        action="store_true",
        help="diagnostic override for checkpoints lacking a corrected training_summary.json",
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    dataset = normalize_dataset_name(args.dataset)
    if args.knn_k < 1:
        raise ValueError("--knn_k must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    tabular_standardizers = fit_tabular_standardizers(args.data_dir)
    preprocessing = tabular_standardizers["report"]
    checkpoint_protocol = None
    checkpoint_sha256 = None
    if not args.allow_unverified_checkpoint:
        checkpoint_protocol = validate_checkpoint_protocol(
            args.checkpoint,
            dataset,
            args.knn_k,
            preprocessing,
        )
        checkpoint_sha256 = json.loads(
            checkpoint_protocol.read_text(encoding="utf-8")
        ).get("checkpoint_sha256")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = raw_tabular(args.data_dir, "train", args.feature_mode)
    x_val = raw_tabular(args.data_dir, "val", args.feature_mode)
    x_test = raw_tabular(args.data_dir, "test", args.feature_mode)
    y_train = torch.load(Path(args.data_dir) / "train_labels.pt", map_location="cpu", weights_only=True).numpy()
    y_val = torch.load(Path(args.data_dir) / "val_labels.pt", map_location="cpu", weights_only=True).numpy()
    y_test = torch.load(Path(args.data_dir) / "test_labels.pt", map_location="cpu", weights_only=True).numpy()

    tabular = HistGradientBoostingClassifier(**release_bse_parameters(args.seed))
    tabular.fit(x_train, y_train)
    bse_configuration = estimator_configuration(tabular, sklearn.__version__)
    tab_val = tabular.predict_proba(x_val)
    tab_test = tabular.predict_proba(x_test)

    base, checkpoint_envelope = baseline_probabilities(
        dataset,
        args.data_dir,
        args.checkpoint,
        device,
        expected_k=args.knn_k,
        seed=args.seed,
        tabular_standardizers=tabular_standardizers,
        allow_unverified_checkpoint=args.allow_unverified_checkpoint,
    )
    base_val, base_test = base["val"][0], base["test"][0]

    best = None
    for weight in np.arange(0.0, 1.0001, 0.05):
        val_prob = (1.0 - weight) * base_val + weight * tab_val
        val_pred = val_prob.argmax(axis=1)
        score = f1_score(y_val, val_pred, average="macro")
        candidate = (float(score), -float(weight), float(weight))
        if best is None or candidate > best[0]:
            best = (candidate, float(weight))
    weight = best[1]
    val_pred = ((1.0 - weight) * base_val + weight * tab_val).argmax(axis=1)
    test_pred = ((1.0 - weight) * base_test + weight * tab_test).argmax(axis=1)
    result = {
        "dataset": display_dataset_name(dataset),
        "seed": args.seed,
        "method": "CBAF-Net",
        "feature_dim": int(x_train.shape[1]),
        "feature_mode": args.feature_mode,
        "bse_configuration": bse_configuration,
        "fusion_weight_bse": weight,
        "baseline_checkpoint": str(args.checkpoint),
        "checkpoint_protocol_verified": checkpoint_protocol is not None,
        "checkpoint_protocol_record": str(checkpoint_protocol) if checkpoint_protocol else None,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_verified": checkpoint_sha256 is not None,
        "checkpoint_envelope": checkpoint_envelope,
        "backbone_tabular_preprocessing": preprocessing,
        "graph_execution": {
            "edge_policy": "strict_split_local_knn",
            "knn_k": int(args.knn_k),
            "batching": "one_hop_target_neighbor",
            "message_direction": "selected_neighbor_to_target",
        },
        "val": metrics(y_val, val_pred),
        "test": metrics(y_test, test_pred),
        "bse_only_test": metrics(y_test, tab_test.argmax(axis=1)),
        "prediction_artifacts": {
            "val": "val_predictions.jsonl",
            "test": "test_predictions.jsonl",
            "profile_cluster_bootstrap": "python analysis/profile_cluster_bootstrap.py --predictions test_predictions.jsonl --out profile_bootstrap.json",
        },
    }
    for split, labels, base_prob, tab_prob in (
        ("val", y_val, base_val, tab_val),
        ("test", y_test, base_test, tab_test),
    ):
        metadata = load_split_metadata(args.data_dir, split)
        if len(metadata) != len(labels):
            raise ValueError(
                f"{split} metadata rows ({len(metadata)}) do not match labels ({len(labels)})"
            )
        metadata_labels = np.asarray(
            [int(row.get("label")) for row in metadata], dtype=np.int64
        )
        if not np.array_equal(metadata_labels, labels.astype(np.int64)):
            raise ValueError(f"{split} metadata labels are not aligned with labels tensor")
        seen_pairs = set()
        for row in metadata:
            pair = (str(row.get("base_profile_id")), int(row.get("label")))
            if pair in seen_pairs:
                raise ValueError(f"{split} contains duplicate base_profile_id/label pair: {pair}")
            seen_pairs.add(pair)
        fused_prob = (1.0 - weight) * base_prob + weight * tab_prob
        output_path = out_dir / f"{split}_predictions.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for index, (row, label, bp, tp, fp) in enumerate(
                zip(metadata, labels, base_prob, tab_prob, fused_prob)
            ):
                payload = {
                    "row_index": index,
                    "sample_id": row.get("sample_id"),
                    "base_profile_id": row.get("base_profile_id"),
                    "label": int(label),
                    "baseline_pred": int(np.argmax(bp)),
                    "bse_pred": int(np.argmax(tp)),
                    "pred": int(np.argmax(fp)),
                    "baseline_prob": [float(value) for value in bp],
                    "bse_prob": [float(value) for value in tp],
                    "fused_prob": [float(value) for value in fp],
                }
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "joblib is required to save the behavioral-statistics expert; "
            "install the repository requirements first"
        ) from exc
    joblib.dump(tabular, out_dir / "behavioral_statistics_expert.joblib")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
