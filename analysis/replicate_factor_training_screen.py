"""Fixed seed extension of the preserved training-only factor screen."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from analysis.run_factor_training_screen import (
    ALL_METHODS, POLICIES, fold, predict_method, prepare, replay, source_hashes,
)
from src.botdmm_development import train_backbone
from src.bse_config import release_bse_parameters
from src.evidence_repair import metrics, write_new_json
from src.factor_evidence import METHODS, train_factor_model
from src.policy_robust import train_policy_model
from src.provenance import sha256_file


NEW_SEEDS = (43, 44)
ALL_SEEDS = (42, *NEW_SEEDS)
DATASETS = ("Twibot22", "Quadbot")
REPLICATION_SOURCES = (
    "analysis/replicate_factor_training_screen.py",
    "configs/factor_training_replication.json",
)
SCOPE = "three_seed_training_only_posthoc_replication"


def replication_hashes():
    code = Path(__file__).resolve().parents[1]
    return {name: sha256_file(code / name) for name in REPLICATION_SOURCES}


def validate_recipe(settings, prior):
    original = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/factor_training_screen.json").read_text()
    )
    if (settings["schema"] != "training-only-factor-replication-v1"
            or tuple(settings["seeds"]) != NEW_SEEDS
            or tuple(settings["methods"]) != ALL_METHODS
            or prior["seeds"] != [42]
            or tuple(prior["methods"]) != ALL_METHODS
            or settings.get("reused_seeds") != [42]
            or set(settings) != set(original) | {"reused_seeds"}):
        raise ValueError("replication requires preserved seed 42 and new seeds 43/44")
    for key in settings:
        if key not in ("schema", "purpose", "seeds", "reused_seeds"):
            if settings[key] != prior.get(key):
                raise ValueError("fixed recipe differs from prior screen: " + key)
    if prior["source_hashes"] != source_hashes():
        raise ValueError("preserved factor implementation changed")


def run(data_dir, temporal, partitions, dataset, protocol, prior_root, output):
    if output.exists():
        raise FileExistsError("replication evidence cannot be overwritten")
    settings = json.loads(protocol.read_text())
    prior = json.loads((prior_root / dataset / "protocol.json").read_text())
    validate_recipe(settings, prior)
    material = prepare(data_dir, temporal, partitions, dataset)
    for key in ("input_hashes", "prior_partition_hashes"):
        if material[key] != prior[key]:
            raise ValueError("preserved training inputs differ: " + key)
    frozen = {
        **settings, "dataset": dataset, "class_count": material["class_count"],
        "input_hashes": material["input_hashes"],
        "prior_partition_hashes": material["prior_partition_hashes"],
        "source_hashes": source_hashes(), "replication_source_hashes": replication_hashes(),
        "prior_protocol_sha256": sha256_file(prior_root / dataset / "protocol.json"),
        "device": "cpu", "test_paths_opened": False, "original_val_paths_opened": False,
        "challenge_used_for_checkpoint_selection": False,
    }
    write_new_json(output / "protocol.json", frozen)
    protocol_hash = sha256_file(output / "protocol.json")
    config, records = settings["configuration"], []
    for seed in NEW_SEEDS:
        for heldout in POLICIES:
            values, views, normalization = fold(material, seed, heldout, config["knn_k"])
            folder = output / f"seed{seed}" / heldout
            write_new_json(folder / "normalization.json", normalization)
            for method in ALL_METHODS:
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
                    fitted = {
                        "configuration": model.get_params(), "parameter_count": None,
                        "selection_rule": "fixed_configuration_no_checkpoint_selection",
                    }
                probabilities = predict_method(model, method, values[2], views[2], config)
                if method == "hgb_statistics":
                    checkpoint = destination / "selected_model.joblib"
                    joblib.dump(model, checkpoint)
                else:
                    checkpoint = destination / "selected_model.pt"
                    torch.save({
                        "state_dict": model.state_dict(), "dataset": dataset, "method": method,
                        "seed": seed, "heldout_policy": heldout, "configuration": config,
                        "normalization": normalization, "protocol_sha256": protocol_hash,
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
                    "claim_scope": SCOPE,
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
                    "event": "run_complete", "dataset": dataset, "seed": seed,
                    "heldout_policy": heldout, "method": method,
                }), flush=True)
    write_new_json(output / "summary.json", {
        "dataset": dataset, "runs": records, "submission_allowed": False,
        "protocol_sha256": protocol_hash,
    })


def replay_extension(data_dir, temporal, partitions, dataset, root, output):
    frozen = json.loads((root / dataset / "protocol.json").read_text())
    if frozen["replication_source_hashes"] != replication_hashes():
        raise ValueError("replication implementation changed before replay")
    replay(data_dir, temporal, partitions, dataset, root, output)


def verified_runs(root, dataset, seeds):
    frozen_path = root / dataset / "protocol.json"
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads((root / dataset / "summary.json").read_text())
    audit = json.loads((root / (dataset + "_audit") / "audit.json").read_text())
    rows = summary["runs"]
    expected = {(seed, policy, method) for seed in seeds for policy in POLICIES for method in ALL_METHODS}
    observed = {(row["seed"], row["heldout_policy"], row["method"]) for row in rows}
    replayed = {
        (row["seed"], row["heldout_policy"], row["method"]): row
        for row in audit["checkpoint_replays"]
    }
    if (observed != expected or len(rows) != len(expected) or set(replayed) != expected
            or len(audit["checkpoint_replays"]) != len(expected) or not audit["all_replayed"]
            or audit["original_val_or_test_access"] or summary["submission_allowed"]
            or frozen["test_paths_opened"] or frozen["original_val_paths_opened"]
            or frozen["source_hashes"] != source_hashes()):
        raise ValueError("incomplete, changed or out-of-bound replication evidence")
    protocol_hash = sha256_file(frozen_path)
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("summary protocol identity differs")
    for row in rows:
        folder = root / dataset / f"seed{row['seed']}" / row["heldout_policy"] / row["method"]
        path = folder / (
            "selected_model.joblib" if row["method"] == "hgb_statistics" else "selected_model.pt"
        )
        key = (row["seed"], row["heldout_policy"], row["method"])
        if (row != json.loads((folder / "metrics.json").read_text())
                or sha256_file(path) != row["checkpoint_sha256"]
                or replayed[key]["checkpoint_sha256"] != row["checkpoint_sha256"]
                or replayed[key]["maximum_probability_difference"] > 1e-6
                or row["protocol_sha256"] != protocol_hash
                or row["original_val_or_test_access"]
                or row["challenge_used_for_checkpoint_selection"]):
            raise ValueError("per-run metric or checkpoint provenance differs")
    return frozen, rows


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("descriptive uncertainty requires all three finite seed scores")
    return {
        "seeds": list(ALL_SEEDS), "values": values.tolist(),
        "mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def summarize(prior_root, root, output):
    if output.exists():
        raise FileExistsError("replication aggregate cannot be overwritten")
    report = {
        "claim_scope": SCOPE, "seeds": list(ALL_SEEDS), "new_fits": 72, "reused_fits": 36,
        "total_fits": 108, "total_checkpoint_replays_verified": 108,
        "submission_allowed": False, "original_val_or_test_access": False, "datasets": {},
    }
    lines = [
        "# Fixed Three-Seed Factor Replication", "",
        "Seed 42 is reused byte-for-byte; seeds 43/44 are new fits of the same recipe.",
        "These are post-hoc, original-training-only policy challenges, not independent confirmation.",
        "Macro-F1 values below are percentages; SD is sample SD across three seeds.",
        "Each seed also changes the deterministic inner fitting/selection profile partition.",
        "The two views share profiles and are not independent datasets.", "",
        "| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        prior, old_rows = verified_runs(prior_root, dataset, (42,))
        frozen, new_rows = verified_runs(root, dataset, NEW_SEEDS)
        validate_recipe(
            {key: frozen[key] for key in json.loads(
                (Path(__file__).resolve().parents[1] / "configs/factor_training_replication.json").read_text()
            )}, prior,
        )
        for key in ("input_hashes", "prior_partition_hashes"):
            if frozen[key] != prior[key]:
                raise ValueError("replication and original input identities differ")
        if (frozen["replication_source_hashes"] != replication_hashes()
                or frozen["prior_protocol_sha256"] != sha256_file(prior_root / dataset / "protocol.json")):
            raise ValueError("replication provenance changed")
        rows = old_rows + new_rows
        indexed = {(row["seed"], row["heldout_policy"], row["method"]): row for row in rows}
        tables = {}
        for policy in POLICIES:
            tables[policy] = {}
            for method in ALL_METHODS:
                selected = [indexed[(seed, policy, method)] for seed in ALL_SEEDS]
                scores = describe([row["challenge"]["macro_f1"] for row in selected])
                counts = {row["fit"]["parameter_count"] for row in selected}
                if len(counts) != 1:
                    raise ValueError("architecture parameter count changed across seeds")
                tables[policy][method] = {
                    "macro_f1": scores, "parameter_count": counts.pop(),
                    "per_class_f1_by_seed": [row["challenge"]["per_class_f1"] for row in selected],
                }
                lines.append("| {} | {} | {} | {} | {:.2f} | {:.2f} |".format(
                    dataset, policy, method,
                    " | ".join("{:.2f}".format(value * 100) for value in scores["values"]),
                    scores["mean"] * 100, scores["seed_sd"] * 100,
                ))
        comparisons = {}
        for policy in POLICIES:
            comparisons[policy] = {}
            for candidate in ("factor_residual", "without_interaction_residual"):
                comparisons[policy][candidate] = {}
                for comparator in ALL_METHODS:
                    if comparator == candidate:
                        continue
                    delta = describe([
                        indexed[(seed, policy, candidate)]["challenge"]["macro_f1"]
                        - indexed[(seed, policy, comparator)]["challenge"]["macro_f1"]
                        for seed in ALL_SEEDS
                    ])
                    comparisons[policy][candidate][comparator] = {
                        **delta, "higher_seed_count": sum(value > 0 for value in delta["values"]),
                        "at_least_one_point_higher_seed_count": sum(
                            value >= 0.01 for value in delta["values"]
                        ),
                    }
        equal_policy = {
            method: describe([
                np.mean([indexed[(seed, policy, method)]["challenge"]["macro_f1"] for policy in POLICIES])
                for seed in ALL_SEEDS
            ]) for method in ALL_METHODS
        }
        report["datasets"][dataset] = {
            "runs_verified": len(rows), "tables": tables, "paired_seed_differences": comparisons,
            "equal_policy_mean_macro_f1": equal_policy,
            "prior_protocol_sha256": sha256_file(prior_root / dataset / "protocol.json"),
            "replication_protocol_sha256": sha256_file(root / dataset / "protocol.json"),
            "input_hashes": frozen["input_hashes"],
        }
    lines.extend([
        "", "## Paired Differences", "",
        "Mean differences are candidate minus comparator, in percentage points.",
        "Counts are positive paired seed differences, not independent significance tests.", "",
        "| View | Policy | Candidate | Comparator | Mean Delta | Delta SD | Positive Seeds |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for dataset, evidence in report["datasets"].items():
        for policy, candidates in evidence["paired_seed_differences"].items():
            for candidate, comparisons in candidates.items():
                for comparator, delta in comparisons.items():
                    lines.append("| {} | {} | {} | {} | {:+.2f} | {:.2f} | {}/3 |".format(
                        dataset, policy, candidate, comparator, delta["mean"] * 100,
                        delta["seed_sd"] * 100, delta["higher_seed_count"],
                    ))
    lines.extend([
        "", "## Boundary", "",
        "All six methods and all three policies are retained, including negative ablations.",
        "No new architecture, hyperparameter search, seed exclusion or automatic promotion is used.",
        "Original validation/test tensors are not model inputs. Old seed-42 outputs remain preserved.",
        "HGB receives global summaries, neural methods receive step inputs, and adapted BotDMM",
        "has additional unlabelled transductive feature-kNN context. Capacity and information differ.",
        "Factor targets encode synthetic construction, not independently annotated account intent.",
        "More seeds quantify recipe sensitivity; they do not establish external validity or SOTA.",
        "This is development evidence and does not authorize replacing manuscript claims.", "",
    ])
    write_new_json(output / "development_summary.json", report)
    (output / "development_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "replay", "summarize"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    for name in ("data-dir", "temporal-dir", "partition-root", "protocol", "prior-root", "root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=name == "out-dir")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.mode == "summarize":
            if not args.root or not args.prior_root:
                parser.error("summarize requires prior-root and root")
            summarize(args.prior_root, args.root, args.out_dir)
        else:
            if not all((args.dataset, args.data_dir, args.temporal_dir, args.partition_root)):
                parser.error("train/replay require dataset and training inputs")
            common = (args.data_dir, args.temporal_dir, args.partition_root, args.dataset)
            if args.mode == "replay":
                if not args.root:
                    parser.error("replay requires root")
                replay_extension(*common, args.root, args.out_dir)
            else:
                if not args.protocol or not args.prior_root:
                    parser.error("train requires protocol and prior-root")
                run(*common, args.protocol, args.prior_root, args.out_dir)


if __name__ == "__main__":
    main()
