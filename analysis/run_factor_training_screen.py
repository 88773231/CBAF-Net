"""Bounded architecture feasibility using original training records only."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import subset
from analysis.screen_unseen_policy import fit_fold_transforms
from src.botdmm_development import (
    DevelopmentGraphView, make_backbone, predict_backbone, train_backbone,
)
from src.bse_config import release_bse_parameters
from src.data_loader import fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, input_files, load_split, make_model, metrics,
    predict, write_new_json,
)
from src.factor_evidence import (
    METHODS, make_factor_model, predict_factor, train_factor_model,
)
from src.policy_robust import train_policy_model
from src.provenance import sha256_file


ALL_METHODS = (*METHODS, "erm_mlp", "hgb_statistics", "botdmm_timing")
POLICIES = ("burst_scheduler", "cyclic_reposter", "uniform_scheduler")
SOURCES = (
    "analysis/run_factor_training_screen.py", "src/factor_evidence.py",
    "src/evidence_model.py", "src/evidence_repair.py", "src/policy_robust.py",
    "src/botdmm_development.py", "src/botdmm.py", "src/data_loader.py",
    "src/layers/decoupler.py", "src/layers/constraints.py", "src/layers/attention.py",
    "analysis/screen_unseen_policy.py", "analysis/audit_development_shortcuts.py",
    "dataset_pipeline/export_quadbot_v3_features.py", "src/bse_config.py",
)


def training_input_files(data_dir, temporal):
    return [
        *input_files(data_dir, "train", 5),
        data_dir / "graph/train_node_features.pt",
        *[temporal / name for name in (
            "train_metadata.jsonl", "train_global.pt", "train_steps.pt",
        )],
    ]


def policies_from_partitions(rows, receipts):
    """Recover policy groups from old training indices, without reopening master."""
    all_rows, policies = set(range(len(rows))), {}
    for name, receipt in receipts.items():
        fitting, selection = receipt["fitting_indices"], receipt["selection_indices"]
        if (len(fitting) != len(set(fitting)) or len(selection) != len(set(selection))
                or set(fitting) & set(selection)
                or not set(fitting + selection) <= all_rows):
            raise ValueError("invalid prior training partition indices")
        excluded = all_rows - set(fitting + selection)
        if not excluded:
            raise ValueError("empty excluded training policy")
        profile_rows = {}
        for index, row in enumerate(rows):
            profile_rows.setdefault(str(row["base_profile_id"]), set()).add(index)
        for profile, indices in profile_rows.items():
            intersection = excluded & indices
            if intersection and intersection != indices:
                raise ValueError("prior exclusion divides a profile")
            if intersection:
                if profile in policies:
                    raise ValueError("profile assigned multiple policies")
                policies[profile] = name
    if set(policies) != {str(row["base_profile_id"]) for row in rows}:
        raise ValueError("prior partitions do not cover all training profiles")
    return policies


def prepare(data_dir, temporal, partition_root, dataset):
    classes = 3 if dataset == "Twibot22" else 4
    train = load_split(data_dir, "train", 5, fit_tabular_standardizers(str(data_dir)), classes)
    metadata_path = temporal / "train_metadata.jsonl"
    timing_rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    if timing_rows != train["metadata"]:
        raise ValueError("training timing row identity differs")
    global_timing = torch.load(temporal / "train_global.pt", map_location="cpu", weights_only=True)
    step_timing = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    if (global_timing.shape != (len(timing_rows), 7)
            or step_timing.shape != (len(timing_rows), 5, 7)
            or not torch.isfinite(global_timing).all()
            or not torch.isfinite(step_timing).all()):
        raise ValueError("invalid training timing shape or values")
    train["raw_tabular"] = torch.cat([train["raw_tabular"], global_timing], dim=1)
    train["action"] = torch.cat([train["action"], step_timing], dim=2)
    graph = torch.load(
        data_dir / "graph/train_node_features.pt", map_location="cpu", weights_only=True,
    )
    if graph.ndim != 2 or graph.shape[0] != len(timing_rows) or not torch.isfinite(graph).all():
        raise ValueError("training graph features differ")
    receipt_files = {name: partition_root / name / "normalization.json" for name in POLICIES}
    policies = policies_from_partitions(
        train["metadata"], {name: json.loads(path.read_text()) for name, path in receipt_files.items()},
    )
    paths = training_input_files(data_dir, temporal)
    return {
        "train": train, "steps": step_timing, "graph": graph,
        "policies": policies, "class_count": classes,
        "input_hashes": {str(path): sha256_file(path) for path in paths},
        "prior_partition_hashes": {name: sha256_file(path) for name, path in receipt_files.items()},
    }


def fold(material, seed, heldout, k):
    train, policies = material["train"], material["policies"]
    fitting, selection = screen_inner_partition(train["metadata"], policies, heldout, seed)
    challenge = [
        index for index, row in enumerate(train["metadata"])
        if policies[row["base_profile_id"]] == heldout
    ]
    indices = (fitting, selection, challenge)
    if set().union(*map(set, indices)) != set(range(len(train["labels"]))):
        raise ValueError("training-only fold does not cover the original training rows")
    values = [subset(train, part) for part in indices]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert_group_disjoint(values[left], values[right])
    normalization = fit_fold_transforms(
        *values, [material["steps"][part] for part in indices],
    )
    normalization.update({
        "fitting_indices": fitting, "selection_indices": selection,
        "challenge_indices": challenge, "challenge_scope": "original_train_only",
    })
    views = [
        DevelopmentGraphView(value, material["graph"][part], "botdmm_timing", k)
        for value, part in zip(values, indices)
    ]
    return values, views, normalization


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    return {name: sha256_file(code / name) for name in SOURCES}


def screen_inner_partition(rows, policies, heldout, seed):
    selection_profiles = set()
    for policy in POLICIES:
        if policy == heldout:
            continue
        profiles = sorted({
            row["base_profile_id"] for row in rows
            if policies[row["base_profile_id"]] == policy
        }, key=lambda profile: hashlib.sha256(
            f"factor-training-screen:{seed}:{profile}".encode()
        ).digest())
        if len(profiles) < 2:
            raise ValueError("each fitting policy needs distinct fitting/selection profiles")
        selection_profiles.update(profiles[:max(1, len(profiles) // 6)])
    fitting, selection = [], []
    for index, row in enumerate(rows):
        if policies[row["base_profile_id"]] == heldout:
            continue
        (selection if row["base_profile_id"] in selection_profiles else fitting).append(index)
    return fitting, selection


def model_for(method, values, view, classes, config):
    if method in METHODS:
        return make_factor_model(values, classes, config, method)
    if method == "erm_mlp":
        return make_model(values, "mlp_matched", classes, config)
    if method == "botdmm_timing":
        return make_backbone(view, classes, config)
    raise ValueError("unknown neural method")


def predict_method(model, method, values, view, config):
    if method in METHODS:
        return predict_factor(model, values, config["batch_size"])
    if method == "erm_mlp":
        return predict(model, values, "mlp_matched", config["batch_size"])[0]
    if method == "botdmm_timing":
        return predict_backbone(model, view, config["batch_size"], config["knn_k"])
    return model.predict_proba(values["raw_tabular"].numpy())


def run(data_dir, temporal, partitions, dataset, protocol_path, output):
    if output.exists():
        raise FileExistsError("training screen evidence cannot be overwritten")
    settings = json.loads(protocol_path.read_text())
    if settings["seeds"] != [42] or tuple(settings["methods"]) != ALL_METHODS:
        raise ValueError("this bounded screen requires the six fixed methods and seed 42")
    material = prepare(data_dir, temporal, partitions, dataset)
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"],
        "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "device": "cpu",
        "test_paths_opened": False, "original_val_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False,
    }
    output.mkdir(parents=True)
    write_new_json(output / "protocol.json", frozen)
    protocol_hash, config = sha256_file(output / "protocol.json"), settings["configuration"]
    records = []
    for seed in settings["seeds"]:
        for heldout in POLICIES:
            values, views, normalization = fold(material, seed, heldout, config["knn_k"])
            folder = output / f"seed{seed}" / heldout
            write_new_json(folder / "normalization.json", normalization)
            for method in settings["methods"]:
                destination = folder / method
                destination.mkdir()

                def progress(state):
                    print(json.dumps({
                        "event": "epoch", "dataset": dataset, "seed": seed,
                        "heldout_policy": heldout, "method": method, **state,
                    }), flush=True)

                if method in METHODS:
                    model, fitted = train_factor_model(
                        values[0], values[1], material["policies"], method,
                        material["class_count"], seed, config, progress,
                    )
                elif method == "erm_mlp":
                    model, fitted = train_policy_model(
                        values[0], values[1], material["policies"], method,
                        material["class_count"], seed, config,
                    )
                elif method == "botdmm_timing":
                    model, fitted = train_backbone(
                        views[0], views[1], values[1], material["policies"],
                        material["class_count"], seed, config, progress,
                    )
                else:
                    model = HistGradientBoostingClassifier(**release_bse_parameters(seed))
                    model.fit(values[0]["raw_tabular"].numpy(), values[0]["labels"].numpy())
                    fitted = {"configuration": model.get_params(), "parameter_count": None,
                              "selection_rule": "fixed_configuration_no_checkpoint_selection"}
                probabilities = predict_method(model, method, values[2], views[2], config)
                if method == "hgb_statistics":
                    checkpoint = destination / "selected_model.joblib"
                    joblib.dump(model, checkpoint)
                else:
                    checkpoint = destination / "selected_model.pt"
                    torch.save({
                        "state_dict": model.state_dict(), "dataset": dataset,
                        "method": method, "seed": seed, "heldout_policy": heldout,
                        "configuration": config, "normalization": normalization,
                        "protocol_sha256": protocol_hash,
                    }, checkpoint)
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": heldout, "method": method,
                    "fit": fitted, "challenge": metrics(
                        values[2]["labels"].numpy(), probabilities, material["class_count"],
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "protocol_sha256": protocol_hash,
                    "graph_challenge": views[2].graph_receipt if method == "botdmm_timing" else None,
                    **{name + "_profiles": len({row["base_profile_id"] for row in value["metadata"]})
                       for name, value in zip(("fitting", "selection", "challenge"), values)},
                    "original_val_or_test_access": False,
                    "challenge_used_for_checkpoint_selection": False,
                    "claim_scope": "one_seed_training_only_posthoc_feasibility",
                }
                write_new_json(destination / "metrics.json", record)
                with (destination / "challenge_predictions.jsonl").open("x", encoding="utf-8") as stream:
                    for row, probability in zip(values[2]["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "prediction": int(probability.argmax()),
                            "probabilities": probability.tolist(),
                        }) + "\n")
                records.append(record)
                print(json.dumps({
                    "event": "run_complete", "dataset": dataset, "method": method,
                    "heldout_policy": heldout, "seed": seed,
                }), flush=True)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": records, "submission_allowed": False,
        "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, partitions, dataset, root, output):
    if output.exists():
        raise FileExistsError("replay evidence cannot be overwritten")
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("frozen implementation changed before replay")
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != frozen[key]:
            raise ValueError("replay input identity changed")
    config, replays = frozen["configuration"], []
    for seed in frozen["seeds"]:
        for heldout in POLICIES:
            values, views, normalization = fold(material, seed, heldout, config["knn_k"])
            folder = root / dataset / f"seed{seed}" / heldout
            if normalization != json.loads((folder / "normalization.json").read_text()):
                raise ValueError("fold transforms differ during replay")
            for method in frozen["methods"]:
                destination = folder / method
                record = json.loads((destination / "metrics.json").read_text())
                path = destination / (
                    "selected_model.joblib" if method == "hgb_statistics" else "selected_model.pt"
                )
                if sha256_file(path) != record["checkpoint_sha256"]:
                    raise ValueError("checkpoint bytes changed")
                if method == "hgb_statistics":
                    model = joblib.load(path)
                else:
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    for key, expected in (
                        ("dataset", dataset), ("method", method), ("seed", seed),
                        ("heldout_policy", heldout), ("configuration", config),
                        ("normalization", normalization),
                        ("protocol_sha256", sha256_file(root / dataset / "protocol.json")),
                    ):
                        if saved[key] != expected:
                            raise ValueError("checkpoint identity differs: " + key)
                    model = model_for(method, values[2], views[2], material["class_count"], config)
                    model.load_state_dict(saved["state_dict"], strict=True)
                probabilities = predict_method(model, method, values[2], views[2], config)
                rows = [
                    json.loads(line) for line in
                    (destination / "challenge_predictions.jsonl").read_text().splitlines()
                ]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in rows] != [
                    (row["sample_id"], row["base_profile_id"], row["label"])
                    for row in values[2]["metadata"]
                ]:
                    raise ValueError("prediction identities differ")
                difference = float(np.max(np.abs(
                    probabilities - np.asarray([row["probabilities"] for row in rows]),
                )))
                if difference > 1e-6:
                    raise ValueError("checkpoint predictions failed to replay")
                if metrics(values[2]["labels"].numpy(), probabilities, material["class_count"]) != (
                    record["challenge"]
                ):
                    raise ValueError("replayed metrics differ")
                replays.append({
                    "seed": seed, "heldout_policy": heldout, "method": method,
                    "maximum_probability_difference": difference,
                    "checkpoint_sha256": sha256_file(path),
                })
    write_new_json(output / "audit.json", {
        "dataset": dataset, "all_replayed": True, "checkpoint_replays": replays,
        "original_val_or_test_access": False, "submission_allowed": False,
    })


def summarize(root, output):
    if output.exists():
        raise FileExistsError("complete screen summary cannot be overwritten")
    report = {
        "claim_scope": "one_seed_training_only_posthoc_feasibility",
        "submission_allowed": False, "original_val_or_test_access": False, "datasets": {},
    }
    lines = [
        "# Training Only Factor Fusion Feasibility", "",
        "Single seed 42; all outcomes retained. Original training-only policy",
        "challenges are exposed development data, not independent confirmation.", "",
        "| View | Policy | Method | Macro-F1 | Parameters |",
        "|---|---|---|---:|---:|",
    ]
    for dataset in ("Twibot22", "Quadbot"):
        frozen = json.loads((root / dataset / "protocol.json").read_text())
        audit = json.loads((root / (dataset + "_audit") / "audit.json").read_text())
        rows = json.loads((root / dataset / "summary.json").read_text())["runs"]
        expected = {(42, policy, method) for policy in POLICIES for method in ALL_METHODS}
        observed = {(row["seed"], row["heldout_policy"], row["method"]) for row in rows}
        if (observed != expected or len(rows) != len(expected)
                or not audit["all_replayed"] or len(audit["checkpoint_replays"]) != 18):
            raise ValueError("incomplete training screen coverage")
        tables = {}
        for policy in POLICIES:
            tables[policy] = {}
            for row in rows:
                if row["heldout_policy"] != policy:
                    continue
                metric_path = (
                    root / dataset / "seed42" / policy / row["method"] / "metrics.json"
                )
                if row != json.loads(metric_path.read_text()):
                    raise ValueError("run summary differs from original metrics")
                tables[policy][row["method"]] = {
                    "macro_f1": row["challenge"]["macro_f1"],
                    "per_class_f1": row["challenge"]["per_class_f1"],
                    "parameter_count": row["fit"]["parameter_count"],
                }
                lines.append("| {} | {} | {} | {:.4f} | {} |".format(
                    dataset, policy, row["method"], row["challenge"]["macro_f1"],
                    row["fit"]["parameter_count"],
                ))
        report["datasets"][dataset] = {
            "runs_verified": 18, "tables": tables,
            "protocol_sha256": sha256_file(root / dataset / "protocol.json"),
            "input_hashes": frozen["input_hashes"],
        }
    lines.extend([
        "", "## Boundary", "",
        "No original validation/test tensors or raw master archive were opened.",
        "Training policy assignments were reconstructed from prior training",
        "partition indices. Reassigned old training records are not new data.",
        "One seed cannot establish reproducibility, module necessity or SOTA.",
        "These models are not capacity-matched; HGB uses only global statistics,",
        "and adapted BotDMM has an additional unlabelled feature-kNN context.",
        "No tuning or automatic final-model selection is authorized by this screen.", "",
    ])
    write_new_json(output / "development_summary.json", report)
    (output / "development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "replay", "summarize"), required=True)
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"))
    for name in ("data-dir", "temporal-dir", "partition-root", "protocol", "root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=name == "out-dir")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "summarize":
            if not args.root:
                parser.error("summarize requires --root")
            summarize(args.root, args.out_dir)
        else:
            if not all((args.dataset, args.data_dir, args.temporal_dir, args.partition_root)):
                parser.error("train/replay require dataset and all training inputs")
            common = (args.data_dir, args.temporal_dir, args.partition_root, args.dataset)
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay requires --root")
                replay(*common, args.root, args.out_dir)
            else:
                if not args.protocol:
                    parser.error("train requires --protocol")
                run(*common, args.protocol, args.out_dir)


if __name__ == "__main__":
    main()
