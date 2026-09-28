import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.data_loader import QuadBotDataset, Twibot20Dataset, collate_fn
from models.botdmm import BotDMM


def load_model(dataset, data_dir, checkpoint, variant, device):
    if dataset == "quadbot":
        ds = QuadBotDataset(data_dir, split="train", num_steps=5)
        num_classes = 4
    else:
        ds = Twibot20Dataset(data_dir, split="train", num_steps=5)
        num_classes = 3
    sample = ds[0]
    model = BotDMM(
        des_size=768,
        tweet_size=768,
        amr_size=768,
        num_prop_size=sample["num_prop"].shape[0],
        llm_features_size=sample["llm_features"].shape[0],
        embedding_dimension=128,
        feature_dim=128,
        num_temporal_steps=5,
        dropout=0.3,
        temperature=0.1,
        alpha=0.5,
        num_classes=num_classes,
        use_raep=variant in {"raep", "four_raep"},
        use_hierarchy=variant in {"raep", "four_raep"},
        use_four_expert=variant in {"four", "four_raep"},
        four_expert_mix=0.4,
        freeze_backbone=variant == "four",
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=False)
    return model.to(device).eval()


def collect(model, dataset, split, device):
    if dataset == "quadbot":
        ds = QuadBotDataset(split[0], split="train", num_steps=5)
    else:
        ds = Twibot20Dataset(split[0], split="train", num_steps=5)
    raise RuntimeError("unused")


def predict(model, dataset_name, data_dir, split, device):
    ds_cls = QuadBotDataset if dataset_name == "quadbot" else Twibot20Dataset
    ds = ds_cls(data_dir, split=split, num_steps=5)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 8:
                des, tweets, amrs, num_prop, llm_features, edges, y, src = batch
            else:
                des, tweets, amrs, num_prop, llm_features, edges, y = batch
            out = model(
                des.to(device), [x.to(device) for x in tweets], [x.to(device) for x in amrs],
                num_prop.to(device), llm_features.to(device), [x.to(device) for x in edges]
            )
            logits.append(out["logits"].cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["quadbot", "twibot20"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--raep", required=True)
    p.add_argument("--four", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_model(args.dataset, args.data_dir, args.base, "base", device)
    raep = load_model(args.dataset, args.data_dir, args.raep, "raep", device)
    four = load_model(args.dataset, args.data_dir, args.four, "four", device)
    val = [predict(m, args.dataset, args.data_dir, "val", device) for m in (base, raep, four)]
    test = [predict(m, args.dataset, args.data_dir, "test", device) for m in (base, raep, four)]
    yv, yt = val[0][1].numpy(), test[0][1].numpy()
    val_prob = [torch.softmax(x[0], dim=1) for x in val]
    test_prob = [torch.softmax(x[0], dim=1) for x in test]
    best = None
    for a in range(0, 21):
        for b in range(0, 21 - a):
            c = 20 - a - b
            w = [a / 20.0, b / 20.0, c / 20.0]
            pv = sum(wi * pi for wi, pi in zip(w, val_prob)).argmax(dim=1).numpy()
            score = f1_score(yv, pv, average="macro")
            if best is None or score > best["val_f1"]:
                best = {"weights": w, "val_f1": float(score)}
    w = best["weights"]
    pt = sum(wi * pi for wi, pi in zip(w, test_prob)).argmax(dim=1).numpy()
    best["test_f1"] = float(f1_score(yt, pt, average="macro"))
    best["dataset"] = args.dataset
    best["base_test_f1"] = float(f1_score(yt, test_prob[0].argmax(dim=1).numpy(), average="macro"))
    best["raep_test_f1"] = float(f1_score(yt, test_prob[1].argmax(dim=1).numpy(), average="macro"))
    best["four_test_f1"] = float(f1_score(yt, test_prob[2].argmax(dim=1).numpy(), average="macro"))
    Path(args.out).write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
