"""Independent BotSSCL random-feature-corruption multiclass adapter.

This is not a faithful reproduction: it uses the study's frozen signed hashes
and statistics rather than the original user representation and binary tasks.
The core augmentation/NT-Xent mechanism follows the inspected primary repository
code4thispaper/BotSSCL at ed3b17588fc2c271947aade266bb1357fc835f61.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data_loader import fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, input_files, load_split, metrics, pooled_inputs,
    seed_everything, write_new_json,
)
from src.provenance import sha256_file


def corrupt_features(anchor, donor, rate, generator):
    if anchor.ndim != 2 or donor.shape != anchor.shape or not 0 <= rate <= 1:
        raise ValueError("invalid corruption inputs")
    count = int(rate * anchor.shape[1])
    if count == 0:
        return anchor.clone()
    selected = torch.rand(anchor.shape, generator=generator).topk(
        count, dim=1, largest=False
    ).indices
    mask = torch.zeros_like(anchor, dtype=torch.bool).scatter_(1, selected, True)
    return torch.where(mask, donor, anchor)


def nt_xent(left, right, temperature=1.0):
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("NT-Xent requires paired batches with at least two rows")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    combined = F.normalize(torch.cat([left, right]), dim=1)
    logits = combined @ combined.T / temperature
    logits.fill_diagonal_(-torch.inf)
    count = left.shape[0]
    targets = (torch.arange(count * 2, device=left.device) + count) % (count * 2)
    return F.cross_entropy(logits, targets)


class CorruptionAdapter(nn.Module):
    def __init__(self, input_dim, class_count, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, class_count)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(self, inputs):
        return self.classifier(self.encoder(inputs))


def probabilities(model, inputs, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).softmax(1)
                          for batch in inputs.split(batch_size)]).numpy()


def fit_adapter(train, val, class_count, seed, config):
    seed_everything(seed)
    train_x = torch.from_numpy(pooled_inputs(train))
    val_x = torch.from_numpy(pooled_inputs(val))
    model = CorruptionAdapter(train_x.shape[1], class_count, config["hidden_dim"])
    optimizer = torch.optim.Adam(
        [*model.encoder.parameters(), *model.projection.parameters()],
        lr=config["learning_rate"],
    )
    generator = torch.Generator().manual_seed(seed)
    pretraining_history = []
    for epoch in range(config["pretraining_epochs"]):
        model.train()
        losses = []
        order = torch.randperm(len(train_x), generator=generator)
        for indices in order.split(config["batch_size"]):
            if len(indices) < 2:
                continue
            anchor = train_x[indices]
            # Donors come exclusively from the training population.
            donors = train_x[torch.randint(len(train_x), (len(indices),), generator=generator)]
            positive = corrupt_features(anchor, donors, config["corruption_rate"], generator)
            loss = nt_xent(
                model.projection(model.encoder(anchor)),
                model.projection(model.encoder(positive)), config["temperature"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        pretraining_history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    optimizer = torch.optim.AdamW(
        [*model.encoder.parameters(), *model.classifier.parameters()],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    best_key, best_state, best_epoch, stale = None, None, None, 0
    finetuning_history = []
    for epoch in range(1, config["finetuning_epochs"] + 1):
        model.train()
        for indices in torch.randperm(len(train_x), generator=generator).split(config["batch_size"]):
            loss = F.cross_entropy(model(train_x[indices]), train["labels"][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scores = metrics(val["labels"].numpy(), probabilities(model, val_x, config["batch_size"]),
                         class_count)
        finetuning_history.append({"epoch": epoch, "validation": scores})
        key = (scores["macro_f1"], scores["mcc"])
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model, probabilities(model, val_x, config["batch_size"]), {
        "selected_epoch": best_epoch, "pretraining": pretraining_history,
        "finetuning": finetuning_history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite adapter evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    class_count = 3 if args.dataset == "Twibot22" else 4
    config = {
        "hidden_dim": 128, "corruption_rate": 0.6, "temperature": 1.0,
        "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 128,
        "pretraining_epochs": 30, "finetuning_epochs": 30, "patience": 6,
    }
    standardizers = fit_tabular_standardizers(str(args.data_dir))
    train = load_split(args.data_dir, "train", 5, standardizers, class_count)
    val = load_split(args.data_dir, "val", 5, standardizers, class_count)
    timing = attach_relative_timing(train, val, args.temporal_dir) if args.temporal_dir else None
    protocol = {
        "dataset": args.dataset, "seeds": [42, 43, 44], "configuration": config,
        "input_aggregation": "temporal_mean_text_action_plus_profile_and_statistics",
        "group_counts": assert_group_disjoint(train, val),
        "relative_timing": timing,
        "source_commit": "ed3b17588fc2c271947aade266bb1357fc835f61",
        "method": "BotSSCL-inspired multiclass corruption adapter",
        "adaptation_status": "architectural_adapter",
        "adaptation_reasons": [
            "frozen study hashes replace original account and language-model representations",
            "same source blocks are used, but text/action time steps are mean-pooled",
            "joint supervised multiclass fine-tuning replaces the binary downstream task",
            "grouped train/validation selection replaces original data splits",
            "bounded 30-epoch pretraining is not the original paper training budget",
        ],
        "input_hashes": {
            path.relative_to(args.data_dir).as_posix(): sha256_file(path)
            for split in ("train", "val") for path in input_files(args.data_dir, split, 5)
        },
        "implementation_sha256": sha256_file(Path(__file__)),
        "source_hashes": {
            relative: sha256_file(Path(__file__).resolve().parents[1] / relative)
            for relative in ("src/botsscl_adapter.py", "src/evidence_repair.py", "src/data_loader.py")
        },
        "test_paths_opened": False, "claim_scope": "exploratory_validation_only",
    }
    write_new_json(args.out_dir / "protocol.json", protocol)
    records = []
    for seed in protocol["seeds"]:
        start = time.monotonic()
        model, probs, history = fit_adapter(train, val, class_count, seed, config)
        directory = args.out_dir / f"seed{seed}"
        directory.mkdir()
        torch.save({"state_dict": model.state_dict(), "configuration": config,
                    "seed": seed, "protocol_sha256": sha256_file(args.out_dir / "protocol.json")},
                   directory / "selected_model.pt")
        record = {"seed": seed, "validation": metrics(val["labels"].numpy(), probs, class_count),
                  "history": history, "elapsed_seconds": time.monotonic() - start}
        write_new_json(directory / "selection.json", record)
        with (directory / "val_predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row, probability in zip(val["metadata"], probs):
                stream.write(json.dumps({
                    "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                    "label": int(row["label"]), "probabilities": probability.tolist(),
                    "prediction": int(probability.argmax()),
                }) + "\n")
        records.append(record)
        print(json.dumps({"dataset": args.dataset, "seed": seed,
                          "validation": record["validation"]}), flush=True)
    write_new_json(args.out_dir / "validation_summary.json", {
        "dataset": args.dataset, "method": protocol["method"], "test_paths_opened": False,
        "adaptation_status": "architectural_adapter", "runs": records,
        "aggregates": {
            key: {"mean": float(np.mean([record["validation"][key] for record in records])),
                  "seed_sd": float(np.std([record["validation"][key] for record in records], ddof=1))}
            for key in ("macro_f1", "accuracy", "mcc")
        },
    })


if __name__ == "__main__":
    main()
