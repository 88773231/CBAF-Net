"""Inspect nine retained checkpoints on their fitting populations only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.run_compact_correspondence_stability import SEEDS, source_hashes, validate_settings
from analysis.run_encoder_objective_diagnostic import load_known_fold
from analysis.run_factor_training_screen import POLICIES
from src.compact_correspondence import make_model
from src.correspondence_dependence import PROBES, diagnose
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def run(data_dir, temporal_dir, control_root, study, output):
    if output.exists():
        raise FileExistsError("retain existing dependence evidence")
    protocol = json.loads((study / "protocol.json").read_text())
    settings = json.loads((Path(__file__).parents[1] /
                           "configs/compact_correspondence_stability.json").read_text())
    validate_settings(settings)
    if (any(protocol[key] != value for key, value in settings.items())
            or protocol["source_hashes"] != source_hashes()
            or protocol["excluded_row_feature_transforms_or_model_forwards"]):
        raise ValueError("retained training protocol differs")
    summary = json.loads((study / "summary.json").read_text())
    expected = {(policy, seed) for policy in POLICIES for seed in SEEDS}
    records = [row for row in summary["records"] if row["condition"] == "compact_correspondence"]
    identities = [(row["policy"], row["seed"]) for row in records]
    if (len(identities) != len(expected) or set(identities) != expected
            or summary["challenge_forward_or_scoring"] or summary["submission_allowed"]):
        raise ValueError("retained compact matrix differs")
    records = {(row["policy"], row["seed"]): row for row in records}
    results, unchanged = [], {}
    for policy in POLICIES:
        values, _, normalization, _, input_hashes = load_known_fold(
            data_dir, temporal_dir, control_root, policy)
        if input_hashes != protocol["input_hashes"]:
            raise ValueError("retained fitting inputs differ")
        del values[1]
        fitting = values[0]
        for seed in SEEDS:
            folder = study / policy / f"seed{seed}" / "compact_correspondence"
            path = folder / "selected_model.pt"
            digest = sha256_file(path)
            record = json.loads((folder / "diagnostic.json").read_text())
            if record != records[(policy, seed)] or digest != record["checkpoint_sha256"]:
                raise ValueError("retained checkpoint or record changed")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            expected_envelope = {
                "condition": "compact_correspondence", "policy": policy, "seed": seed,
                "partition_seed": 42, "configuration": settings["configuration"],
                "normalization": normalization, "protocol_sha256": sha256_file(study / "protocol.json"),
            }
            if (set(saved) != {"state_dict", *expected_envelope}
                    or any(saved[key] != value for key, value in expected_envelope.items())):
                raise ValueError("retained checkpoint envelope differs")
            model = make_model(fitting, settings["configuration"], "compact_correspondence")
            model.load_state_dict(saved["state_dict"], strict=True)
            model.eval()
            result = diagnose(model, fitting, settings["configuration"]["batch_size"])
            results.append({"policy": policy, "seed": seed, "checkpoint_sha256": digest, **result})
            unchanged[str(path)] = digest
            print(json.dumps({"completed": len(results), "planned": 9, "policy": policy, "seed": seed}),
                  flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in unchanged.items()):
        raise ValueError("read-only checkpoint changed")
    aggregates = {}
    for probe in PROBES:
        aggregates[probe] = {
            "mean_fitting_cross_entropy_change": float(np.mean([
                row["outputs"][probe]["fitting_cross_entropy"]
                - row["outputs"]["baseline"]["fitting_cross_entropy"] for row in results])),
            "mean_prediction_change_fraction": float(np.mean([
                row["outputs"][probe]["prediction_changes_from_baseline"] / row["fitting_rows"]
                for row in results])),
            "mean_absolute_logit_change": float(np.mean([
                row["outputs"][probe]["mean_absolute_logit_change"] for row in results])),
        }
    root = Path(__file__).parents[1]
    write_new_json(output, {
        "scope": "post_hoc_fitting_only_correspondence_dependence", "dataset": "Quadbot",
        "checkpoints_inspected": 9, "new_fits": 0, "training_seeds": list(SEEDS),
        "retained_protocol_sha256": sha256_file(study / "protocol.json"),
        "source_hashes": {name: sha256_file(root / name) for name in (
            "src/correspondence_dependence.py", "analysis/run_correspondence_dependence.py")},
        "records": results, "probe_aggregates": aggregates, "reference_checkpoints_unchanged": True,
        "known_selection_features_loaded_but_not_forwarded": True,
        "selection_or_challenge_or_test_forwards": False, "gpu_used": False,
        "probes_are_retrained_ablations": False, "causal_effect_established": False,
        "architecture_novelty_established": False, "submission_allowed": False,
    })


def main():
    parser = argparse.ArgumentParser()
    for name in ("data-dir", "temporal-dir", "control-root", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.data_dir, args.temporal_dir, args.control_root, args.study, args.output)


if __name__ == "__main__":
    main()
