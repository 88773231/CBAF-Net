import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import QuadBotDataset, Twibot20Dataset, collate_fn
from models.botdmm import BotDMM


CLASS_NAMES = ["human", "traditional_bot", "llm_bot", "full_stack_agent"]

MODEL_VARIANT_TO_MODE = {
    "base": "base",
    "botdmm": "base",
    "aemp": "aemp",
    "quadfusion": "aemp",
    "raep_chdf": "base",
    "raep_only": "base",
    "chdf_only": "base",
    "four_expert": "base",
    "four_expert_raep": "base",
}

MODEL_DISPLAY_NAMES = {
    "aemp": "QuadFusion-Net",
    "quadfusion": "QuadFusion-Net",
    "botdmm": "BotDMM",
    "base": "BotDMM",
    "raep_chdf": "BotDMM-RAEP-CHDF",
    "raep_only": "BotDMM+RAEP",
    "chdf_only": "BotDMM+CHDF",
    "four_expert": "BotDMM-FourExpert-Residual",
    "four_expert_raep": "BotDMM-FourExpert-RAEP",
}


def build_dataset(dataset_name, data_dir, split, num_steps):
    if dataset_name == "quadbot":
        return QuadBotDataset(data_dir, split=split, num_steps=num_steps)
    if dataset_name == "twibot20":
        return Twibot20Dataset(data_dir, split=split, num_steps=num_steps)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def unpack_batch(batch, device):
    if len(batch) == 8:
        des, tweets, amrs, num_prop, llm_features, edge_indices, labels, _ = batch
    else:
        des, tweets, amrs, num_prop, llm_features, edge_indices, labels = batch
    return (
        des.to(device),
        [tensor.to(device) for tensor in tweets],
        [tensor.to(device) for tensor in amrs],
        num_prop.to(device),
        llm_features.to(device),
        [edge.to(device) for edge in edge_indices],
        labels.to(device),
    )


def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            des, tweets, amrs, num_prop, llm_features, edge_indices, labels = unpack_batch(batch, device)
            outputs = model(des, tweets, amrs, num_prop, llm_features, edge_indices)
            logits = outputs["logits"]
            total_loss += criterion(logits, labels).item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    avg_method = "binary" if num_classes == 2 else "macro"
    return {
        "loss": float(total_loss / max(len(loader), 1)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average=avg_method)),
        "mcc": float(matthews_corrcoef(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average=avg_method, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average=avg_method, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=CLASS_NAMES[:num_classes],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved QuadFusion-Net checkpoint.")
    parser.add_argument("--dataset", choices=["twibot20", "quadbot"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--model_variant",
        choices=[
            "aemp", "base", "no_memory", "no_event", "no_prototype", "no_domain",
            "botdmm", "quadfusion",
            "raep_chdf", "raep_only", "chdf_only",
            "four_expert", "four_expert_raep",
            "no_structure", "no_content", "no_orthogonal",
        ],
        default="botdmm",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    splits = {
        split: build_dataset(args.dataset, args.data_dir, split, args.num_steps)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        for split, dataset in splits.items()
    }

    sample = splits["train"][0]
    num_classes = 4 if args.dataset == "quadbot" else 3
    model = BotDMM(
        des_size=768,
        tweet_size=768,
        amr_size=768,
        num_prop_size=sample["num_prop"].shape[0],
        llm_features_size=sample["llm_features"].shape[0],
        embedding_dimension=args.embedding_dim,
        feature_dim=args.feature_dim,
        num_temporal_steps=args.num_steps,
        dropout=args.dropout,
        temperature=args.temperature,
        alpha=args.alpha,
        num_classes=num_classes,
        ablation_mode=MODEL_VARIANT_TO_MODE.get(args.model_variant, args.model_variant),
        use_raep=args.model_variant in {"raep_chdf", "raep_only", "four_expert_raep"},
        use_hierarchy=args.model_variant in {"raep_chdf", "chdf_only", "four_expert_raep"},
        use_four_expert=args.model_variant in {"four_expert", "four_expert_raep"},
        four_expert_mix=0.4,
    )
    # Old BotDMM checkpoints predate the optional RAEP/four-expert modules.
    # Non-strict loading keeps the shared trunk while leaving those extensions
    # at their checkpoint-defined or initialized values for analysis runs.
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True),
        strict=False,
    )
    model.to(device)

    result = {
        "baseline": MODEL_DISPLAY_NAMES.get(args.model_variant, args.model_variant),
        "model_variant": args.model_variant,
        "dataset": args.data_dir,
        "checkpoint": args.checkpoint.as_posix(),
        "seed": args.seed,
        "class_names": CLASS_NAMES[:num_classes],
        "splits": {split: len(dataset) for split, dataset in splits.items()},
        "train": evaluate(model, loaders["train"], device, num_classes),
        "val": evaluate(model, loaders["val"], device, num_classes),
        "test": evaluate(model, loaders["test"], device, num_classes),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "checkpoint": result["checkpoint"],
                "test": {
                    key: result["test"][key]
                    for key in ("loss", "accuracy", "macro_f1", "mcc", "precision", "recall")
                },
                "out": args.out.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

