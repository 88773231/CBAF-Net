"""Fixed module diagnostics with a shared fold builder for training and replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_development_shortcuts import collect_provenance, subset
from analysis.audit_policy_robust import standardizer_payload
from analysis.run_botdmm_timing import graph_inputs
from analysis.screen_unseen_policy import fit_fold_transforms, inner_profile_partition
from analysis.summarize_policy_robust import verified_records
from src.botdmm_development import DevelopmentGraphView, predict_backbone
from src.botdmm_module_controls import METHOD_MODES, make_module_control, train_module_control
from src.data_loader import fit_column_standardizer, fit_tabular_standardizers
from src.evidence_repair import (
    assert_group_disjoint, attach_relative_timing, input_files, load_split, metrics, write_new_json,
)
from src.provenance import sha256_file


def prepare(data_dir, temporal, master, dataset):
    classes = 3 if dataset == "Twibot22" else 4
    transforms = fit_tabular_standardizers(str(data_dir))
    train = load_split(data_dir, "train", 5, transforms, classes)
    val = load_split(data_dir, "val", 5, transforms, classes)
    assert_group_disjoint(train, val)
    timing = attach_relative_timing(train, val, temporal)
    steps = [torch.load(temporal / f"{split}_steps.pt", map_location="cpu", weights_only=True)
             for split in ("train", "val")]
    graphs = [graph_inputs(data_dir, split, len(values["labels"]))[0]
              for split, values in (("train", train), ("val", val))]
    policies, provenance = collect_provenance(master)
    return {
        "train": train, "val": val, "steps": steps, "graphs": graphs,
        "policies": policies, "provenance": provenance, "timing": timing, "class_count": classes,
        "input_hashes": {path.relative_to(data_dir).as_posix(): sha256_file(path)
                         for split in ("train", "val") for path in input_files(data_dir, split, 5)},
        "graph_input_hashes": {
            f"graph/{split}_node_features.pt": sha256_file(data_dir / "graph" / f"{split}_node_features.pt")
            for split in ("train", "val")
        },
    }


def fold(material, seed, heldout, k):
    train, val, policies = (material[name] for name in ("train", "val", "policies"))
    fit_indices, select_indices = inner_profile_partition(train["metadata"], policies, heldout, seed)
    challenge_indices = [index for index, row in enumerate(val["metadata"])
                         if policies[row["base_profile_id"]] == heldout]
    indices = (fit_indices, select_indices, challenge_indices)
    values = [subset(source, rows) for source, rows in zip((train, train, val), indices)]
    for left, right in ((values[0], values[1]), (values[0], values[2]), (values[1], values[2])):
        assert_group_disjoint(left, right)
    if any(policies[row["base_profile_id"]] == heldout
           for part in values[:2] for row in part["metadata"]):
        raise ValueError("held-out policy entered fitting or selection")
    step_values = [source[rows] for source, rows in zip(
        (material["steps"][0], material["steps"][0], material["steps"][1]), indices,
    )]
    table_transform = fit_column_standardizer(values[0]["raw_tabular"])
    step_transform = fit_column_standardizer(step_values[0].flatten(0, 1))
    fit_fold_transforms(*values, step_values)
    normalization = {
        "global_table": standardizer_payload(table_transform),
        "step_timing": standardizer_payload(step_transform), "clamp": [-5.0, 5.0],
        "fitting_indices": fit_indices, "selection_indices": select_indices,
        "challenge_indices": challenge_indices, "fit_scope": "inner_fitting_profiles_only",
    }
    graph_sources = (material["graphs"][0], material["graphs"][0], material["graphs"][1])
    views = [DevelopmentGraphView(part, source[rows], "botdmm_timing", k)
             for part, source, rows in zip(values, graph_sources, indices)]
    return values, views, normalization


def validate_reference(material, prior, settings):
    if (prior["seeds"] != settings["seeds"]
            or prior["configuration"] != settings["configuration"]
            or prior["test_paths_opened"] or prior["outer_challenge_used_for_selection"]):
        raise ValueError("baseline seeds, recipe or access boundary differs")
    for key in ("input_hashes", "graph_input_hashes", "timing"):
        if material[key] != prior[key]:
            raise ValueError("baseline input differs: " + key)


def run(data_dir, temporal, master, dataset, settings_path, reference, out):
    if out.exists():
        raise FileExistsError("module diagnostic outputs may not be overwritten")
    settings = json.loads(settings_path.read_text())
    if (set(settings["methods"]) != set(METHOD_MODES)
            or len(set(settings["seeds"])) != len(settings["seeds"])):
        raise ValueError("protocol must contain both fixed module removals and distinct seeds")
    material = prepare(data_dir, temporal, master, dataset)
    prior, baseline_records = verified_records(reference, dataset)
    validate_reference(material, prior, settings)
    code = Path(__file__).resolve().parents[1]
    for name, expected in prior["source_hashes"].items():
        if sha256_file(code / name) != expected:
            raise ValueError("earlier timing experiment implementation changed: " + name)
    sources = {
        **prior["source_hashes"],
        **{name: sha256_file(code / name) for name in (
            "analysis/run_botdmm_modules.py", "src/botdmm_module_controls.py",
        )},
    }
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        **{key: material[key] for key in (
            "input_hashes", "graph_input_hashes", "timing", "provenance",
        )},
        "master_sha256": sha256_file(master), "source_hashes": sources,
        "baseline_protocol_sha256": sha256_file(reference / dataset / "protocol.json"),
        "test_paths_opened": False, "outer_challenge_used_for_selection": False, "device": "cpu",
    }
    if frozen["master_sha256"] != prior["master_sha256"]:
        raise ValueError("baseline main account archive differs")
    out.mkdir(parents=True)
    write_new_json(out / "protocol.json", frozen)
    protocol_hash = sha256_file(out / "protocol.json")
    config, records = settings["configuration"], []
    for seed in settings["seeds"]:
        for policy in sorted(set(material["policies"].values())):
            values, views, normalization = fold(material, seed, policy, config["knn_k"])
            baseline = baseline_records[(seed, policy, "botdmm_timing")]
            if (views[0].graph_receipt != baseline["fit"]["graph_fit"]
                    or views[1].graph_receipt != baseline["fit"]["graph_inner_selection"]
                    or views[2].graph_receipt != baseline["graph_challenge"]):
                raise ValueError("module control and baseline graphs differ")
            baseline_norm = reference / dataset / f"seed{seed}" / policy / "normalization.json"
            if normalization != json.loads(baseline_norm.read_text()):
                raise ValueError("module control and baseline fold normalizations differ")
            folder = out / f"seed{seed}" / policy
            folder.mkdir(parents=True)
            write_new_json(folder / "normalization.json", normalization)
            for method in settings["methods"]:
                destination = folder / method
                destination.mkdir()
                model, fitted = train_module_control(
                    views[0], views[1], values[1], material["policies"],
                    material["class_count"], seed, config, method,
                    progress=lambda state: print(json.dumps({
                        "event": "epoch", "dataset": dataset, "seed": seed,
                        "heldout_policy": policy, "method": method, **state,
                    }), flush=True),
                )
                probabilities = predict_backbone(model, views[2], config["batch_size"], config["knn_k"])
                checkpoint = destination / "selected_model.pt"
                torch.save({
                    "state_dict": model.state_dict(), "dataset": dataset, "method": method,
                    "seed": seed, "heldout_policy": policy, "configuration": config,
                    "normalization": normalization, "protocol_sha256": protocol_hash,
                }, checkpoint)
                record = {
                    "dataset": dataset, "seed": seed, "heldout_policy": policy, "method": method,
                    "fit": fitted, "challenge": metrics(
                        values[2]["labels"].numpy(), probabilities, material["class_count"],
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "protocol_sha256": protocol_hash, "graph_challenge": views[2].graph_receipt,
                    **{name + "_profiles": len(set(view.base_profile_ids))
                       for name, view in zip(("fitting", "selection", "challenge"), views)},
                    "test_state": "NOT_OPENED", "outer_challenge_used_for_selection": False,
                    "claim_scope": "posthoc_module_diagnostic",
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
                print(json.dumps({"event": "run_complete", "dataset": dataset, "method": method,
                                  "seed": seed, "heldout_policy": policy,
                                  "epochs": len(fitted["history"])}), flush=True)
    write_new_json(out / "summary.json", {
        "dataset": dataset, "runs": records, "test_state": "NOT_OPENED",
        "submission_allowed": False, "protocol_sha256": protocol_hash,
    })


def replay(data_dir, temporal, master, dataset, root, out):
    if out.exists():
        raise FileExistsError("checkpoint replay output may not be overwritten")
    frozen, records = verified_records(root, dataset)
    code = Path(__file__).resolve().parents[1]
    for name, expected in frozen["source_hashes"].items():
        if sha256_file(code / name) != expected:
            raise ValueError("frozen diagnostic implementation changed")
    material = prepare(data_dir, temporal, master, dataset)
    for key in ("input_hashes", "graph_input_hashes", "timing"):
        if material[key] != frozen[key]:
            raise ValueError("frozen replay input differs")
    if sha256_file(master) != frozen["master_sha256"]:
        raise ValueError("account archive changed")
    config, replays = frozen["configuration"], []
    for seed in frozen["seeds"]:
        for policy in sorted(set(material["policies"].values())):
            values, views, normalization = fold(material, seed, policy, config["knn_k"])
            norm_path = root / dataset / f"seed{seed}" / policy / "normalization.json"
            if json.loads(norm_path.read_text()) != normalization:
                raise ValueError("saved fold normalization differs")
            for method in frozen["methods"]:
                row = records[(seed, policy, method)]
                destination = root / dataset / f"seed{seed}" / policy / method
                path = destination / "selected_model.pt"
                if sha256_file(path) != row["checkpoint_sha256"]:
                    raise ValueError("checkpoint bytes changed")
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                expected_identity = {
                    "dataset": dataset, "method": method, "seed": seed, "heldout_policy": policy,
                    "configuration": config, "normalization": normalization,
                    "protocol_sha256": row["protocol_sha256"],
                }
                if any(checkpoint[key] != expected for key, expected in expected_identity.items()):
                    raise ValueError("checkpoint identity or normalization differs")
                if (views[0].graph_receipt != row["fit"]["graph_fit"]
                        or views[1].graph_receipt != row["fit"]["graph_inner_selection"]
                        or views[2].graph_receipt != row["graph_challenge"]):
                    raise ValueError("fold graph reconstruction differs")
                model = make_module_control(views[2], material["class_count"], config, method)
                if sum(parameter.numel() for parameter in model.parameters()) != row["fit"]["parameter_count"]:
                    raise ValueError("checkpoint parameter count differs")
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                probabilities = predict_backbone(model, views[2], config["batch_size"], config["knn_k"])
                saved = [json.loads(line) for line in
                         (destination / "challenge_predictions.jsonl").read_text().splitlines()
                         if line.strip()]
                if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in saved] != [
                    (row["sample_id"], row["base_profile_id"], row["label"]) for row in values[2]["metadata"]
                ]:
                    raise ValueError("prediction row identities differ")
                expected = np.asarray([row["probabilities"] for row in saved])
                difference = float(np.max(np.abs(expected - probabilities)))
                if not np.allclose(expected, probabilities, atol=1e-6, rtol=0):
                    raise ValueError("checkpoint predictions cannot be replayed")
                if sha256_file(path) != row["checkpoint_sha256"]:
                    raise ValueError("replay changed checkpoint bytes")
                replays.append({"seed": seed, "heldout_policy": policy, "method": method,
                                "maximum_probability_difference": difference,
                                "checkpoint_sha256": row["checkpoint_sha256"]})
    out.mkdir(parents=True)
    write_new_json(out / "audit.json", {
        "dataset": dataset, "all_replayed": True, "checkpoint_replays": replays,
        "test_access": False, "submission_allowed": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("Twibot22", "Quadbot"), required=True)
    for name in ("data-dir", "temporal-dir", "master", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--replay-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.replay_root:
            replay(args.data_dir, args.temporal_dir, args.master, args.dataset,
                   args.replay_root, args.out_dir)
        else:
            if not args.protocol or not args.reference_dir:
                parser.error("training requires --protocol and --reference-dir")
            run(args.data_dir, args.temporal_dir, args.master, args.dataset,
                args.protocol, args.reference_dir, args.out_dir)


if __name__ == "__main__":
    main()
