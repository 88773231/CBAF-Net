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
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.botdmm import BotDMM
from src.data_loader import QuadBotDataset, Twibot22Dataset, collate_fn


def normalize_dataset_name(dataset):
    """Map legacy CLI aliases to the canonical internal dataset IDs."""
    value = str(dataset).strip().lower()
    if value in {"twibot20", "twibot22"}:
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


def baseline_probabilities(dataset, data_dir, checkpoint, device):
    dataset = normalize_dataset_name(dataset)
    dataset_cls = QuadBotDataset if dataset == "quadbot" else Twibot22Dataset
    num_classes = 4 if dataset == "quadbot" else 3
    train_ds = dataset_cls(data_dir, split="train", num_steps=5)
    sample = train_ds[0]
    model = BotDMM(
        des_size=768, tweet_size=768, amr_size=768,
        num_prop_size=sample["num_prop"].shape[0],
        llm_features_size=sample["llm_features"].shape[0],
        embedding_dimension=128, feature_dim=128, num_temporal_steps=5,
        dropout=0.3, temperature=0.1, alpha=0.5,
        num_classes=num_classes, ablation_mode="base",
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing BotDMM parameters: "
            + ", ".join(incompatible.missing_keys[:10])
        )
    model.to(device).eval()
    outputs = {}
    for split in ("val", "test"):
        ds = dataset_cls(data_dir, split=split, num_steps=5)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                des, tweets, amrs, num_prop, llm_features, edges, y, *_ = batch
                logits = model(
                    des.to(device), [x.to(device) for x in tweets],
                    [x.to(device) for x in amrs], num_prop.to(device),
                    llm_features.to(device), [x.to(device) for x in edges],
                )["logits"]
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.append(y.numpy())
        outputs[split] = (np.concatenate(probs), np.concatenate(labels))
    return outputs


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
        choices=["quadbot", "twibot22", "twibot20"],
        required=True,
        help="Dataset name; twibot20 is retained as a legacy alias for Twibot22.",
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
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    dataset = normalize_dataset_name(args.dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = raw_tabular(args.data_dir, "train", args.feature_mode)
    x_val = raw_tabular(args.data_dir, "val", args.feature_mode)
    x_test = raw_tabular(args.data_dir, "test", args.feature_mode)
    y_train = torch.load(Path(args.data_dir) / "train_labels.pt", map_location="cpu", weights_only=True).numpy()
    y_val = torch.load(Path(args.data_dir) / "val_labels.pt", map_location="cpu", weights_only=True).numpy()
    y_test = torch.load(Path(args.data_dir) / "test_labels.pt", map_location="cpu", weights_only=True).numpy()

    tabular = HistGradientBoostingClassifier(
        max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=args.seed,
    )
    tabular.fit(x_train, y_train)
    tab_val = tabular.predict_proba(x_val)
    tab_test = tabular.predict_proba(x_test)

    base = baseline_probabilities(dataset, args.data_dir, args.checkpoint, device)
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
        "fusion_weight_bse": weight,
        "baseline_checkpoint": str(args.checkpoint),
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
