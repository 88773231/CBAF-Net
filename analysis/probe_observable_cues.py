"""Replay selected development cue heads; do not fit or inspect held-out data."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analysis.audit_observable_evolution import audit
from analysis.audit_observable_schema import training_identities
from analysis.run_observable_evolution import CONFIGURATION, METADATA_SHA256, POLICIES, SOURCE_SHA256
from dataset_pipeline.observable_cues import observe
from src.data_loader import apply_column_standardizer
from src.evidence_repair import metrics, write_new_json
from src.observable_evolution import CUES, INPUTS, make_model
from src.provenance import sha256_file


CONDITION = "without_both"


def selection_values(source, metadata, receipt, normalization):
    if (source.name != "train.jsonl" or sha256_file(source) != SOURCE_SHA256
            or sha256_file(metadata) != METADATA_SHA256):
        raise ValueError("frozen cue source identities changed")
    identities = training_identities(metadata)
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    indices = receipt["selection_indices"]
    if len(set(indices)) != len(indices) or set(indices) & set(receipt["fitting_indices"]):
        raise ValueError("cue selection indices overlap fitting")
    selected = [rows[index] for index in indices]
    allowed = {row["sample_id"] for row in selected}
    features = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identity = row["sample_id"]
            if identity not in allowed:
                continue
            if identity in features or (str(row["base_profile_id"]), row["label"]) != identities[identity]:
                raise ValueError("selected cue source identity differs")
            features[identity] = observe(row)
    if set(features) != allowed:
        raise ValueError("selected cue source records missing")
    values = {
        "metadata": selected, "labels": torch.tensor([row["label"] for row in selected]),
        **{name: torch.from_numpy(np.stack([features[row["sample_id"]][name] for row in selected]))
           for name in INPUTS},
    }
    for name, transform in normalization.items():
        if name not in ("numeric", "behavior", "style", "timing"):
            raise ValueError("unexpected cue normalization field")
        tensor = values[name]
        arrays = {key: torch.tensor(value, dtype=torch.bool if key == "variable_mask" else torch.float32)
                  for key, value in transform.items()}
        values[name] = apply_column_standardizer(
            tensor.reshape(-1, tensor.shape[-1]), arrays).reshape_as(tensor)
    return values


def summarize(labels, fused, cues):
    labels, fused, cues = np.asarray(labels), np.asarray(fused), np.asarray(cues)
    count = len(labels)
    if (labels.shape != (count,) or count < 1 or fused.shape != (count, 4)
            or cues.shape != (count, len(CUES), 4)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= 4))):
        raise ValueError("aligned four-class cue predictions required")
    for values in (fused, cues):
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(-1), 1, atol=1e-5)):
            raise ValueError("valid cue probability distributions required")
    guesses = cues.argmax(-1)
    fused_correct = fused.argmax(-1) == labels
    cue_correct = guesses == labels[:, None]
    any_correct = cue_correct.any(1)
    pairs = [(left, right) for left in range(len(CUES)) for right in range(left + 1, len(CUES))]
    disagreement = np.stack([guesses[:, left] != guesses[:, right] for left, right in pairs]).mean(0)
    mixed = cues.mean(1)
    jsd = (cues * (np.log(np.clip(cues, 1e-12, 1))
                    - np.log(np.clip(mixed[:, None], 1e-12, 1)))).sum(-1).mean(1)
    return {
        "rows": count, "fused_errors": int((~fused_correct).sum()),
        "fused_errors_with_at_least_one_correct_cue": int((~fused_correct & any_correct).sum()),
        "fused_errors_with_all_cues_wrong": int((~fused_correct & ~any_correct).sum()),
        "fused_errors_with_unanimous_cues": int(
            (~fused_correct & (guesses == guesses[:, :1]).all(1)).sum()),
        "cue_metrics": {cue: metrics(labels, cues[:, index], 4) for index, cue in enumerate(CUES)},
        "cue_error_events_above_90_percent_confidence": {
            cue: int((~cue_correct[:, index] & (cues[:, index].max(1) >= .9)).sum())
            for index, cue in enumerate(CUES)
        },
        "pairwise_disagreement_sum_on_fused_errors": float(disagreement[~fused_correct].sum()),
        "pairwise_disagreement_sum_on_fused_correct": float(disagreement[fused_correct].sum()),
        "jsd_sum_on_fused_errors": float(jsd[~fused_correct].sum()),
        "jsd_sum_on_fused_correct": float(jsd[fused_correct].sum()),
        "any_correct_cue_is_label_using_oracle_not_an_achievable_score": True,
    }


def cue_predictions(model, values, batch_size):
    model.eval()
    fused, cues = [], []
    with torch.no_grad():
        for indices in torch.arange(len(values["labels"])).split(batch_size):
            output = model(*(values[name][indices] for name in INPUTS))
            fused.append(output["logits"].softmax(1))
            cues.append(torch.stack([output["cue_logits"][cue].softmax(1) for cue in CUES], dim=1))
    return torch.cat(fused).numpy(), torch.cat(cues).numpy()


def run(source, metadata, partitions, study, output):
    if output.exists():
        raise FileExistsError("preserve existing cue diagnosis")
    evidence = audit(study)
    receipts = {policy: json.loads((partitions / policy / "normalization.json").read_text())
                for policy in POLICIES}
    records, unique_errors, unique_recoverable, unique_rows = [], set(), set(), set()
    for policy in POLICIES:
        print(json.dumps({"event": "cue_probe", "policy": policy}), flush=True)
        normalizer = json.loads((study / policy / "normalization.json").read_text())
        if (normalizer["partition_sha256"] != sha256_file(partitions / policy / "normalization.json")
                or normalizer["selection_indices"] != receipts[policy]["selection_indices"]
                or normalizer["fitting_indices"] != receipts[policy]["fitting_indices"]
                or normalizer["fit_scope"] != "inner_fitting_profiles_only"):
            raise ValueError("recorded cue fitting-only normalization differs")
        normalization = normalizer["transforms"]
        values = selection_values(source, metadata, receipts[policy], normalization)
        checkpoint = study / policy / CONDITION / "selected_model.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (saved["condition"] != CONDITION or saved["policy"] != policy
                or saved["configuration"] != CONFIGURATION or saved["normalization"] != normalization
                or saved["protocol_sha256"] != sha256_file(study / "protocol.json")):
            raise ValueError("cue checkpoint identity or normalization differs")
        model = make_model(CONFIGURATION, CONDITION)
        model.load_state_dict(saved["state_dict"], strict=True)
        fused, cues = cue_predictions(model, values, CONFIGURATION["batch_size"])
        previous = [json.loads(line) for line in
                    (study / policy / CONDITION / "inner_selection_predictions.jsonl").read_text().splitlines()]
        if [(row["sample_id"], row["base_profile_id"], row["label"]) for row in previous] != [
            (row["sample_id"], row["base_profile_id"], row["label"]) for row in values["metadata"]
        ]:
            raise ValueError("cue replay identities differ")
        difference = float(np.abs(fused - np.asarray([row["probabilities"] for row in previous])).max())
        if difference > 1e-6:
            raise ValueError("cue replay changed selected fused predictions")
        labels = values["labels"].numpy()
        correct, any_cue = fused.argmax(1) == labels, (cues.argmax(2) == labels[:, None]).any(1)
        for row, fused_ok, cue_ok in zip(previous, correct, any_cue):
            identity = row["sample_id"]
            unique_rows.add(identity)
            if not fused_ok:
                unique_errors.add(identity)
                if cue_ok:
                    unique_recoverable.add(identity)
        records.append({
            "policy": policy, "condition": CONDITION, "checkpoint_sha256": sha256_file(checkpoint),
            "maximum_fused_replay_difference": difference, **summarize(labels, fused, cues),
        })
    sum_field = lambda name: sum(record[name] for record in records)
    errors, rows = sum_field("fused_errors"), sum_field("rows")
    report = {
        "scope": "posthoc_exposed_known_policy_inner_selection_checkpoint_heads",
        "reference_summary_sha256": evidence["summary_sha256"], "records": records,
        "row_events_in_overlapping_folds": rows,
        "unique_selection_trajectories": len(unique_rows),
        "unique_trajectories_with_any_fused_error": len(unique_errors),
        "unique_error_trajectories_with_any_correct_cue_in_any_fold": len(unique_recoverable),
        "fused_error_events": errors,
        "fused_error_events_with_any_correct_cue": sum_field("fused_errors_with_at_least_one_correct_cue"),
        "mean_pairwise_disagreement_on_fused_errors": (
            sum_field("pairwise_disagreement_sum_on_fused_errors") / errors if errors else None),
        "mean_pairwise_disagreement_on_fused_correct": (
            sum_field("pairwise_disagreement_sum_on_fused_correct") / (rows - errors)
            if rows > errors else None),
        "mean_jsd_on_fused_errors": sum_field("jsd_sum_on_fused_errors") / errors if errors else None,
        "mean_jsd_on_fused_correct": (
            sum_field("jsd_sum_on_fused_correct") / (rows - errors) if rows > errors else None),
        "checkpoint_population_replays": len(records), "new_fits": 0,
        "normalization_refitted": False, "only_selection_features_extracted": True,
        "original_validation_test_or_excluded_policy_forwards": False,
        "gpu_used": False, "identifiers_features_posts_exported": False,
        "diagnostic_not_module_ablation_or_independent_evaluation": True,
        "cue_dependence_and_causal_origin_not_established": True,
        "oracle_not_achievable_performance": True, "submission_allowed": False,
    }
    write_new_json(output, report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "metadata", "partitions", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.source, args.metadata, args.partitions, args.study, args.output)


if __name__ == "__main__":
    main()
