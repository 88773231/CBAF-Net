"""Train the release BotDMM backbone with strict one-hop graph batches.

The kNN files contain split-local ``(target, selected_neighbor)`` pairs. Each
seed batch is expanded with every selected neighbor, graph messages run from
neighbor to target, and the supervised loss is evaluated only on seed rows.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.botdmm import BotDMM
from src.data_loader import QuadBotDataset, StrictOneHopNeighborCollator, Twibot22Dataset
from src.provenance import sha256_file


CHECKPOINT_FORMAT_VERSION = "cbaf-botdmm-checkpoint-v2"


def normalize_dataset_name(dataset):
    value = str(dataset).strip().lower()
    if value == "twibot22":
        return "twibot22"
    if value == "quadbot":
        return "quadbot"
    raise ValueError(f"Unsupported dataset: {dataset}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, expected_k, seed):
    collator = StrictOneHopNeighborCollator(
        dataset,
        expected_k=expected_k,
        require_strict=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )
    return loader, collator


def forward_seed_logits(model, batch, device):
    logits = model(
        batch["des"].to(device, non_blocking=True),
        [value.to(device, non_blocking=True) for value in batch["tweets"]],
        [value.to(device, non_blocking=True) for value in batch["amrs"]],
        batch["num_prop"].to(device, non_blocking=True),
        batch["llm_features"].to(device, non_blocking=True),
        [value.to(device, non_blocking=True) for value in batch["edge_indices"]],
        compute_auxiliary_losses=False,
    )["logits"]
    return logits[batch["seed_positions"].to(device, non_blocking=True)]


def metric_dict(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def train_epoch(model, loader, optimizer, device, clip_grad):
    model.train()
    losses = []
    all_labels = []
    all_predictions = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = forward_seed_logits(model, batch, device)
        labels = batch["seed_labels"].to(device, non_blocking=True)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    result = metric_dict(all_labels, all_predictions)
    result["loss"] = float(np.mean(losses))
    return result


def evaluate(model, loader, device, expected_rows):
    model.eval()
    losses = []
    all_indices = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_seed_logits(model, batch, device)
            labels = batch["seed_labels"].to(device, non_blocking=True)
            losses.append(float(F.cross_entropy(logits, labels).cpu()))
            all_indices.extend(batch["seed_global_indices"].tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu())

    if all_indices != list(range(expected_rows)):
        raise RuntimeError("evaluation loader did not emit each split row exactly once in order")
    probabilities = torch.cat(all_probabilities, dim=0).numpy()
    predictions = probabilities.argmax(axis=1)
    result = metric_dict(all_labels, predictions)
    result["loss"] = float(np.mean(losses))
    return result, probabilities


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
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    dataset_name = normalize_dataset_name(args.dataset)
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1 or args.knn_k < 1:
        raise ValueError("batch_size, epochs, patience, and knn_k must be positive")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dataset_cls = QuadBotDataset if dataset_name == "quadbot" else Twibot22Dataset
    num_classes = 4 if dataset_name == "quadbot" else 3

    train_dataset = dataset_cls(args.data_dir, split="train", num_steps=5)
    shared_standardizers = train_dataset.tabular_standardizers
    val_dataset = dataset_cls(
        args.data_dir,
        split="val",
        num_steps=5,
        tabular_standardizers=shared_standardizers,
    )
    test_dataset = dataset_cls(
        args.data_dir,
        split="test",
        num_steps=5,
        tabular_standardizers=shared_standardizers,
    )
    train_loader, train_collator = make_loader(
        train_dataset, args.batch_size, True, args.knn_k, args.seed
    )
    val_loader, val_collator = make_loader(
        val_dataset, args.batch_size, False, args.knn_k, args.seed
    )
    test_loader, test_collator = make_loader(
        test_dataset, args.batch_size, False, args.knn_k, args.seed
    )

    sample = train_dataset[0]
    model = BotDMM(
        des_size=sample["des"].shape[0],
        tweet_size=sample["tweets"][0].shape[0],
        amr_size=sample["amrs"][0].shape[0],
        num_prop_size=sample["num_prop"].shape[0],
        llm_features_size=sample["llm_features"].shape[0],
        embedding_dimension=128,
        feature_dim=128,
        num_temporal_steps=5,
        dropout=0.3,
        temperature=0.1,
        alpha=0.5,
        num_classes=num_classes,
        ablation_mode="base",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best_model.pt"
    preprocessing = train_dataset.tabular_preprocessing
    graph_protocol = {
        "edge_policy": "strict_split_local_knn",
        "knn_k": int(args.knn_k),
        "batching": "one_hop_target_neighbor",
        "message_direction": "selected_neighbor_to_target",
    }
    checkpoint_metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": "Twibot22" if dataset_name == "twibot22" else "Quadbot",
        "seed": int(args.seed),
        "num_classes": int(num_classes),
        "model_input_dimensions": {
            "description": int(sample["des"].shape[0]),
            "tweet": int(sample["tweets"][0].shape[0]),
            "event_action": int(sample["amrs"][0].shape[0]),
            "num_prop": int(sample["num_prop"].shape[0]),
            "llm_features": int(sample["llm_features"].shape[0]),
        },
        "tabular_preprocessing": preprocessing,
        "graph_protocol": graph_protocol,
    }
    history = []
    best_score = -float("inf")
    best_epoch = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.clip_grad)
        val_metrics, _ = evaluate(model, val_loader, device, len(val_dataset))
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(json.dumps(history[-1], sort_keys=True))
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    **checkpoint_metadata,
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("the newly written checkpoint does not satisfy the release contract")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    val_metrics, _ = evaluate(model, val_loader, device, len(val_dataset))
    test_metrics, _ = evaluate(model, test_loader, device, len(test_dataset))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    summary = {
        "dataset": "Twibot22" if dataset_name == "twibot22" else "Quadbot",
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "validation": val_metrics,
        "test": test_metrics,
        "configuration": {
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "objective": "seed_node_cross_entropy",
            **graph_protocol,
            "checkpoint_format": CHECKPOINT_FORMAT_VERSION,
            "tabular_preprocessing": preprocessing,
        },
        "edge_validation": {
            "train": train_collator.validation_reports[0],
            "val": val_collator.validation_reports[0],
            "test": test_collator.validation_reports[0],
        },
        "history": history,
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
