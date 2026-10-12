"""Training/validation-only evidence repair on the immutable grouped inputs.

This entry point never constructs a test path. The old held-out test has already
been inspected, so new model results are exploratory until independently checked.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.bse_config import release_bse_parameters
from src.data_loader import (
    apply_column_standardizer, fit_column_standardizer,
    fit_tabular_standardizers, preprocess_tensor,
)
from src.evidence_model import MatchedFeatureMLP, TemporalEvidenceModel
from src.provenance import sha256_file


METHODS = (
    "hgb_statistics", "logistic_pooled", "mlp_matched",
    "temporal_evidence", "without_interaction", "without_gate",
    "without_style", "without_timing",
)


def write_new_json(path, payload):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(labels, probabilities, class_count):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(labels), class_count):
        raise ValueError("probability shape mismatch")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("invalid probabilities")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
        raise ValueError("probabilities must sum to one")
    predicted = probabilities.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=list(range(class_count)),
                                  average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "per_class_f1": f1_score(labels, predicted, labels=list(range(class_count)),
                               average=None, zero_division=0).tolist(),
    }


def input_files(data_dir, split, steps):
    if split not in {"train", "val"}:
        raise ValueError("evidence selection only permits train and val")
    paths = [
        data_dir / f"{split}_metadata.jsonl", data_dir / f"{split}_labels.pt",
        data_dir / "descriptions" / f"{split}_des_tensor.pt",
        data_dir / "numerical" / f"{split}_num_properties_tensor.pt",
        data_dir / "numerical" / f"{split}_llm_features.pt",
    ]
    for step in range(1, steps + 1):
        paths.extend([
            data_dir / f"{split}_text_tensor{step}.pt",
            data_dir / f"{split}_amr_tensor{step}.pt",
        ])
    return paths


def load_split(data_dir, split, steps, standardizers, class_count):
    paths = input_files(data_dir, split, steps)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    with paths[0].open(encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    labels = torch.load(paths[1], map_location="cpu", weights_only=True).long()
    if labels.ndim != 1 or labels.numel() != len(metadata):
        raise ValueError("label and metadata rows do not align")
    if not torch.equal(labels, torch.tensor([int(row["label"]) for row in metadata])):
        raise ValueError("ordered metadata labels differ from tensor labels")
    if set(labels.tolist()) != set(range(class_count)):
        raise ValueError("each split must contain the expected classes")
    pairs = [(str(row["base_profile_id"]), int(row["label"])) for row in metadata]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate profile/class pair")

    def tensor(path):
        value = torch.load(path, map_location="cpu", weights_only=True).float()
        if value.ndim != 2 or value.shape[0] != len(metadata):
            raise ValueError(f"invalid feature shape at {path}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite features at {path}")
        return value

    raw_num, raw_style = tensor(paths[3]), tensor(paths[4])
    values = {
        "description": preprocess_tensor(tensor(paths[2])),
        "text": torch.stack([preprocess_tensor(tensor(paths[5 + 2 * index]))
                             for index in range(steps)], dim=1),
        "action": torch.stack([preprocess_tensor(tensor(paths[6 + 2 * index]))
                               for index in range(steps)], dim=1),
        "tabular": torch.cat([
            apply_column_standardizer(raw_num, standardizers["num_prop"]),
            apply_column_standardizer(raw_style, standardizers["llm_features"]),
        ], dim=1),
        "numerical": apply_column_standardizer(raw_num, standardizers["num_prop"]),
        "raw_tabular": torch.cat([raw_num, raw_style], dim=1),
        "labels": labels, "metadata": metadata,
    }
    return values


def assert_group_disjoint(train, val):
    train_groups = {str(row["base_profile_id"]) for row in train["metadata"]}
    val_groups = {str(row["base_profile_id"]) for row in val["metadata"]}
    if train_groups & val_groups:
        raise ValueError("a profile group crosses train/validation")
    for values, groups in ((train, train_groups), (val, val_groups)):
        if len(values["metadata"]) != len(groups) * len(set(values["labels"].tolist())):
            raise ValueError("profile groups are not class-complete")
    return {"train": len(train_groups), "val": len(val_groups)}


def model_inputs(values, method):
    if method == "without_timing":
        return (
            values["description"], values["text"], values["action"][:, :, :-7],
            values["tabular"][:, :-7],
        )
    return (
        values["description"], values["text"], values["action"],
        values["numerical"] if method == "without_style" else values["tabular"],
    )


def flattened_inputs(values):
    return torch.cat([
        values["description"], values["text"].flatten(1),
        values["action"].flatten(1), values["tabular"],
    ], dim=1).numpy()


def pooled_inputs(values):
    return torch.cat([
        values["description"], values["text"].mean(1),
        values["action"].mean(1), values["tabular"],
    ], dim=1).numpy()


def attach_relative_timing(train, val, temporal_dir):
    hashes = {}
    for split, values in (("train", train), ("val", val)):
        path = temporal_dir / f"{split}_metadata.jsonl"
        with path.open(encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream if line.strip()]
        if metadata != values["metadata"]:
            raise ValueError("timing sidecar identities do not match original metadata")
        hashes[path.name] = sha256_file(path)
    global_train = torch.load(temporal_dir / "train_global.pt", map_location="cpu", weights_only=True)
    step_train = torch.load(temporal_dir / "train_steps.pt", map_location="cpu", weights_only=True)
    global_transform = fit_column_standardizer(global_train)
    step_transform = fit_column_standardizer(step_train.flatten(0, 1))
    for split, values in (("train", train), ("val", val)):
        global_path, step_path = temporal_dir / f"{split}_global.pt", temporal_dir / f"{split}_steps.pt"
        global_values = torch.load(global_path, map_location="cpu", weights_only=True)
        step_values = torch.load(step_path, map_location="cpu", weights_only=True)
        if global_values.shape != (len(values["labels"]), 7):
            raise ValueError("invalid global timing sidecar shape")
        if step_values.shape != (len(values["labels"]), values["action"].shape[1], 7):
            raise ValueError("invalid step timing sidecar shape")
        if not torch.isfinite(global_values).all() or not torch.isfinite(step_values).all():
            raise ValueError("timing sidecars must be finite")
        scaled_global = apply_column_standardizer(global_values, global_transform)
        scaled_steps = apply_column_standardizer(
            step_values.flatten(0, 1), step_transform
        ).reshape_as(step_values)
        values["raw_tabular"] = torch.cat([values["raw_tabular"], global_values], dim=1)
        values["tabular"] = torch.cat([values["tabular"], scaled_global], dim=1)
        values["numerical"] = torch.cat([values["numerical"], scaled_global], dim=1)
        values["action"] = torch.cat([values["action"], scaled_steps], dim=2)
        hashes[global_path.name], hashes[step_path.name] = sha256_file(global_path), sha256_file(step_path)
    receipt = temporal_dir / "export_receipt.json"
    hashes[receipt.name] = sha256_file(receipt)
    return {
        "version": "relative-rounded-gap-v1", "fit_split": "train",
        "global_features": 7, "per_step_features": 7, "sidecar_hashes": hashes,
        "shared_by_all_new_baselines_and_candidates": True,
    }


def make_model(values, method, class_count, config):
    description, text, action, tabular = model_inputs(values, method)
    if method == "mlp_matched":
        return MatchedFeatureMLP(
            description.shape[1] + text.shape[1] * text.shape[2]
            + action.shape[1] * action.shape[2] + tabular.shape[1],
            class_count, hidden_dim=config["mlp_hidden_dim"], dropout=config["dropout"],
        )
    return TemporalEvidenceModel(
        description.shape[1], text.shape[2], action.shape[2], tabular.shape[1],
        class_count, steps=text.shape[1], hidden_dim=config["hidden_dim"],
        dropout=config["dropout"], interaction=method != "without_interaction",
        gate=method != "without_gate",
    )


def predict(model, values, method, batch_size):
    model.eval()
    inputs = model_inputs(values, method)
    probabilities, gate_weights = [], []
    with torch.no_grad():
        for start in range(0, len(values["labels"]), batch_size):
            result = model(*(value[start:start + batch_size] for value in inputs))
            probabilities.append(result["logits"].softmax(1).numpy())
            if "gate_weights" in result:
                gate_weights.append(result["gate_weights"].numpy())
    diagnostic = None
    if gate_weights:
        weights = np.concatenate(gate_weights)
        diagnostic = {
            "expert_order": ["text_action_temporal", "statistics"],
            "classwise_mean_weights": weights.mean(0).tolist(),
            "mean_entropy": float(-(weights * np.log(np.clip(weights, 1e-12, 1))).sum(2).mean()),
        }
    return np.concatenate(probabilities), diagnostic


def train_neural(train, val, method, class_count, seed, config):
    seed_everything(seed)
    model = make_model(train, method, class_count, config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    inputs = model_inputs(train, method)
    generator = torch.Generator().manual_seed(seed)
    history, best_state, best_key, best_epoch, stale = [], None, None, None, 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        order = torch.randperm(len(train["labels"]), generator=generator)
        for indices in order.split(config["batch_size"]):
            result = model(*(value[indices] for value in inputs))
            labels = train["labels"][indices]
            loss = F.cross_entropy(result["logits"], labels)
            if "branch_logits" in result:
                auxiliary = torch.stack([
                    F.cross_entropy(result["branch_logits"][:, :, branch], labels)
                    for branch in range(result["branch_logits"].shape[2])
                ]).mean()
                loss = loss + config["branch_loss_weight"] * auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities, _ = predict(model, val, method, config["batch_size"])
        scores = metrics(val["labels"].numpy(), probabilities, class_count)
        key = (scores["macro_f1"], scores["mcc"])
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)),
                        "validation": scores})
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    probabilities, diagnostic = predict(model, val, method, config["batch_size"])
    return model, probabilities, {
        "best_epoch": best_epoch, "history": history,
        "gate_diagnostics": diagnostic,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def run_selection(data_dir, out_dir, dataset, class_count, seeds, methods, config, temporal_dir=None):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("selection requires a new or empty evidence directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = config["steps"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seeds must be nonempty and unique")
    if any(method not in METHODS for method in methods) or len(set(methods)) != len(methods):
        raise ValueError("methods must be supported and unique")
    if "without_timing" in methods and temporal_dir is None:
        raise ValueError("without_timing requires a relative-timing sidecar comparison")
    standardizers = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", steps, standardizers, class_count)
    val = load_split(data_dir, "val", steps, standardizers, class_count)
    group_counts = assert_group_disjoint(train, val)
    relative_timing = attach_relative_timing(train, val, temporal_dir) if temporal_dir else None
    repository = Path(__file__).resolve().parents[1]
    source_files = ["src/evidence_model.py", "src/evidence_repair.py", "src/data_loader.py",
                    "dataset_pipeline/export_relative_timing.py"]
    protocol = {
        "schema": "cbaf-evidence-repair-validation-v1",
        "dataset": dataset, "class_count": class_count, "seeds": list(seeds),
        "methods": list(methods), "configuration": config,
        "group_counts": group_counts,
        "input_signature": standardizers["report"],
        "relative_timing": relative_timing,
        "source_hashes": {name: sha256_file(repository / name) for name in source_files},
        "input_hashes": {
            path.relative_to(data_dir).as_posix(): sha256_file(path)
            for split in ("train", "val") for path in input_files(data_dir, split, steps)
        },
        "environment": {"torch": str(torch.__version__), "sklearn": sklearn.__version__},
        "device": "cpu", "test_paths_opened": False,
        "old_test_has_been_inspected": True, "claim_scope": "exploratory_validation_only",
    }
    write_new_json(out_dir / "protocol.json", protocol)
    protocol_hash = sha256_file(out_dir / "protocol.json")
    summary = []
    for seed in seeds:
        for method in methods:
            start = time.monotonic()
            run_dir = out_dir / f"{method}_seed{seed}"
            run_dir.mkdir()
            if method in {"hgb_statistics", "logistic_pooled"}:
                if method == "hgb_statistics":
                    estimator = HistGradientBoostingClassifier(**release_bse_parameters(seed))
                    train_input, val_input = train["raw_tabular"].numpy(), val["raw_tabular"].numpy()
                else:
                    estimator = LogisticRegression(C=1.0, max_iter=300, random_state=seed)
                    train_input, val_input = pooled_inputs(train), pooled_inputs(val)
                estimator.fit(train_input, train["labels"].numpy())
                probabilities = estimator.predict_proba(val_input)
                import joblib
                joblib.dump(estimator, run_dir / "selected_model.joblib")
                fit_record = {"configuration": estimator.get_params(deep=False)}
            else:
                model, probabilities, fit_record = train_neural(
                    train, val, method, class_count, seed, config
                )
                torch.save({
                    "format": "cbaf-temporal-evidence-candidate-v1",
                    "method": method, "dataset": dataset, "seed": seed,
                    "configuration": config, "protocol_sha256": protocol_hash,
                    "state_dict": model.state_dict(),
                }, run_dir / "selected_model.pt")
            record = {
                "method": method, "seed": seed, "dataset": dataset,
                "validation": metrics(val["labels"].numpy(), probabilities, class_count),
                "fit": fit_record, "elapsed_seconds": time.monotonic() - start,
                "protocol_sha256": protocol_hash, "test_state": "NOT_OPENED",
                "selection_split": "val", "claim_scope": "exploratory_validation_only",
            }
            write_new_json(run_dir / "selection.json", record)
            with (run_dir / "val_predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row, probability in zip(val["metadata"], probabilities):
                    payload = {
                        "sample_id": row["sample_id"],
                        "base_profile_id": row["base_profile_id"],
                        "label": int(row["label"]),
                        "probabilities": probability.tolist(),
                        "prediction": int(probability.argmax()),
                    }
                    stream.write(json.dumps(payload) + "\n")
            summary.append(record)
            print(json.dumps({key: record[key] for key in
                              ("dataset", "method", "seed", "validation", "elapsed_seconds")}),
                  flush=True)
    aggregates = {}
    for method in methods:
        rows = [row for row in summary if row["method"] == method]
        aggregates[method] = {
            metric: {"mean": float(np.mean([row["validation"][metric] for row in rows])),
                     "seed_sd": float(np.std([row["validation"][metric] for row in rows],
                                              ddof=1)) if len(rows) > 1 else None}
            for metric in ("macro_f1", "accuracy", "mcc")
        }
    result = {"dataset": dataset, "test_state": "NOT_OPENED",
              "claim_scope": "exploratory_validation_only", "aggregates": aggregates,
              "protocol_sha256": protocol_hash, "runs": summary}
    write_new_json(out_dir / "validation_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    torch.set_num_threads(protocol["cpu_threads"])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=protocol["cpu_threads"]):
        run_selection(
            args.data_dir, args.out_dir, args.dataset, 3 if args.dataset == "Twibot22" else 4,
            protocol["seeds"], protocol["methods"], protocol["configuration"], args.temporal_dir,
        )


if __name__ == "__main__":
    main()
