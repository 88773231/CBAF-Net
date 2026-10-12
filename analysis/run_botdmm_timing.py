"""Fixed three-seed backbone comparison on previously inspected policy challenges."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from analysis.audit_policy_robust import standardizer_payload
from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition
from src.botdmm_development import (
    METHODS, DevelopmentGraphView, predict_backbone, train_backbone,
)
from src.data_loader import fit_column_standardizer, fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, input_files, load_split, metrics, write_new_json,
)
from src.provenance import sha256_file


def validate_reference(reference, dataset, seeds, config, frozen):
    protocol_path = reference / dataset / "protocol.json"
    prior = json.loads(protocol_path.read_text(encoding="utf-8"))
    if prior["seeds"] != seeds or prior["test_paths_opened"] or prior["outer_challenge_used_for_selection"]:
        raise ValueError("reference seed or development boundary differs")
    if frozen["mlp_reference"]["method"] not in prior["methods"]:
        raise ValueError("reference control is missing")
    for key in frozen["mlp_reference"]["same_recipe_keys"]:
        if prior["configuration"][key] != config[key]:
            raise ValueError("matched MLP recipe differs: " + key)
    summary = json.loads((reference / dataset / "summary.json").read_text())
    if summary["protocol_sha256"] != sha256_file(protocol_path) or summary["test_state"] != "NOT_OPENED":
        raise ValueError("invalid reference summary")
    return prior


def graph_inputs(data_dir, split, rows):
    if split not in ("train", "val"):
        raise ValueError("only development graph inputs may be opened")
    path = data_dir / "graph" / f"{split}_node_features.pt"
    values = torch.load(path, map_location="cpu", weights_only=True).float()
    if values.ndim != 2 or values.shape[0] != rows or not torch.isfinite(values).all():
        raise ValueError("graph feature rows are not valid")
    return values, path


def run(data_dir, temporal, master, output, reference, dataset, protocol_path):
    if output.exists():
        raise FileExistsError("refusing to overwrite the backbone round")
    settings = json.loads(protocol_path.read_text(encoding="utf-8"))
    seeds, methods, config = settings["seeds"], settings["methods"], settings["configuration"]
    if not seeds or len(set(seeds)) != len(seeds) or set(methods) != set(METHODS):
        raise ValueError("protocol needs distinct seeds and both timing conditions")
    class_count = 3 if dataset == "Twibot22" else 4
    prior = validate_reference(reference, dataset, seeds, config, settings)
    standardizers = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", 5, standardizers, class_count)
    val = load_split(data_dir, "val", 5, standardizers, class_count)
    assert_group_disjoint(train, val)
    timing = attach_relative_timing(train, val, temporal)
    train_timing = torch.load(temporal / "train_steps.pt", map_location="cpu", weights_only=True)
    val_timing = torch.load(temporal / "val_steps.pt", map_location="cpu", weights_only=True)
    train_graph, train_graph_path = graph_inputs(data_dir, "train", len(train["labels"]))
    val_graph, val_graph_path = graph_inputs(data_dir, "val", len(val["labels"]))
    policies, provenance = collect_provenance(master)
    code = Path(__file__).resolve().parents[1]
    hashes = {
        path.relative_to(data_dir).as_posix(): sha256_file(path)
        for split in ("train", "val") for path in input_files(data_dir, split, 5)
    }
    if hashes != prior["input_hashes"] or timing != prior["timing"]:
        raise ValueError("backbone and MLP reference individual input bytes differ")
    sources = (
        "analysis/run_botdmm_timing.py", "src/botdmm_development.py", "src/botdmm.py",
        "src/layers/attention.py", "src/layers/decoupler.py", "src/layers/constraints.py",
        "src/data_loader.py", "src/policy_robust.py", "analysis/screen_unseen_policy.py",
        "src/evidence_repair.py", "analysis/audit_development_shortcuts.py",
        "analysis/audit_policy_robust.py", "dataset_pipeline/export_quadbot_v3_features.py",
    )
    frozen = {
        **settings, "dataset": dataset, "class_count": class_count,
        "timing": timing, "input_hashes": hashes,
        "graph_input_hashes": {path.relative_to(data_dir).as_posix(): sha256_file(path)
                              for path in (train_graph_path, val_graph_path)},
        "source_hashes": {name: sha256_file(code / name) for name in sources},
        "master_sha256": sha256_file(master), "provenance": provenance,
        "mlp_reference_protocol_sha256": sha256_file(reference / dataset / "protocol.json"),
        "test_paths_opened": False, "outer_challenge_used_for_selection": False,
        "device": "cpu", "submission_allowed": False,
    }
    output.mkdir(parents=True)
    write_new_json(output / "protocol.json", frozen)
    protocol_hash = sha256_file(output / "protocol.json")
    records = []
    for seed in seeds:
        for heldout in sorted(set(policies.values())):
            fit_indices, select_indices = inner_profile_partition(
                train["metadata"], policies, heldout, seed
            )
            challenge_indices = [
                index for index, row in enumerate(val["metadata"])
                if policies[row["base_profile_id"]] == heldout
            ]
            fitting, selection, challenge = (
                subset(train, fit_indices), subset(train, select_indices),
                subset(val, challenge_indices),
            )
            for left, right in ((fitting, selection), (fitting, challenge), (selection, challenge)):
                assert_group_disjoint(left, right)
            if any(policies[row["base_profile_id"]] == heldout
                   for values in (fitting, selection) for row in values["metadata"]):
                raise ValueError("heldout policy entered fitting/selection")
            table_transform = fit_column_standardizer(fitting["raw_tabular"])
            step_transform = fit_column_standardizer(train_timing[fit_indices].flatten(0, 1))
            fit_fold_transforms(
                fitting, selection, challenge,
                (train_timing[fit_indices], train_timing[select_indices],
                 val_timing[challenge_indices]),
            )
            fold_dir = output / f"seed{seed}" / heldout
            fold_dir.mkdir(parents=True)
            normalization = {
                "global_table": standardizer_payload(table_transform),
                "step_timing": standardizer_payload(step_transform), "clamp": [-5.0, 5.0],
                "fitting_indices": fit_indices, "selection_indices": select_indices,
                "challenge_indices": challenge_indices, "fit_scope": "inner_fitting_profiles_only",
            }
            write_new_json(fold_dir / "normalization.json", normalization)
            for method in methods:
                run_dir = fold_dir / method
                run_dir.mkdir()
                fitting_view = DevelopmentGraphView(
                    fitting, train_graph[fit_indices], method, config["knn_k"]
                )
                selection_view = DevelopmentGraphView(
                    selection, train_graph[select_indices], method, config["knn_k"]
                )
                model, fitted = train_backbone(
                    fitting_view, selection_view, selection, policies, class_count, seed, config,
                    progress=lambda state: print(json.dumps({
                        "event": "epoch", "dataset": dataset, "seed": seed,
                        "heldout_policy": heldout, "method": method, **state,
                    }), flush=True),
                )
                # Challenge features and graph only enter inference after
                # checkpoint selection on the remaining training policies.
                challenge_view = DevelopmentGraphView(
                    challenge, val_graph[challenge_indices], method, config["knn_k"]
                )
                probabilities = predict_backbone(
                    model, challenge_view, config["batch_size"], config["knn_k"]
                )
                checkpoint = run_dir / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "method": method, "seed": seed,
                    "dataset": dataset, "heldout_policy": heldout, "configuration": config,
                    "normalization": normalization, "protocol_sha256": protocol_hash,
                    "graph_challenge": challenge_view.graph_receipt,
                    "input_dimensions": {
                        "description": fitting_view.des_tensor.shape[1],
                        "text": fitting_view.tweets_list[0].shape[1],
                        "action": fitting_view.amrs_list[0].shape[1],
                        "numerical": fitting_view.num_prop.shape[1],
                        "style": fitting_view.llm_features.shape[1],
                    },
                }, checkpoint)
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": heldout, "method": method,
                    "fit": fitted, "challenge": metrics(
                        challenge["labels"].numpy(), probabilities, class_count
                    ),
                    "graph_challenge": challenge_view.graph_receipt,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "fitting_profiles": len(set(fitting_view.base_profile_ids)),
                    "selection_profiles": len(set(selection_view.base_profile_ids)),
                    "challenge_profiles": len(set(challenge_view.base_profile_ids)),
                    "protocol_sha256": protocol_hash, "test_state": "NOT_OPENED",
                    "claim_scope": "posthoc_development_only",
                    "outer_challenge_used_for_selection": False,
                }
                write_new_json(run_dir / "metrics.json", record)
                with (run_dir / "challenge_predictions.jsonl").open("x", encoding="utf-8") as stream:
                    for row, probability in zip(challenge["metadata"], probabilities):
                        stream.write(json.dumps({
                            "sample_id": row["sample_id"], "base_profile_id": row["base_profile_id"],
                            "label": int(row["label"]), "prediction": int(probability.argmax()),
                            "probabilities": probability.tolist(),
                        }) + "\n")
                records.append(record)
                print(json.dumps({
                    "dataset": dataset, "seed": seed, "heldout_policy": heldout, "method": method,
                    "challenge": record["challenge"], "best_epoch": fitted["best_epoch"],
                    "epochs": len(fitted["history"]), "elapsed_seconds": fitted["elapsed_seconds"],
                }), flush=True)
    aggregates = {}
    for policy in sorted(set(policies.values())):
        aggregates[policy] = {}
        for method in methods:
            values = [row["challenge"]["macro_f1"] for row in records
                      if row["heldout_policy"] == policy and row["method"] == method]
            aggregates[policy][method] = {
                "mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)),
            }
    result = {
        "dataset": dataset, "runs": records, "aggregates": aggregates,
        "test_state": "NOT_OPENED", "submission_allowed": False,
        "protocol_sha256": protocol_hash,
    }
    write_new_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.protocol.read_text())
    torch.set_num_threads(frozen["cpu_threads"])
    with threadpool_limits(limits=frozen["cpu_threads"]):
        run(args.data_dir, args.temporal_dir, args.master, args.out_dir,
            args.reference_dir, args.dataset, args.protocol)


if __name__ == "__main__":
    main()
