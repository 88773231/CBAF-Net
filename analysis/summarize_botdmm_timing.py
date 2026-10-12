"""Retain all BotDMM timing results and paired comparisons to the fixed MLP."""

import argparse
from pathlib import Path

import numpy as np

from analysis.paired_profile_cluster_bootstrap import aligned_dataset, analyze_dataset
from analysis.summarize_policy_robust import verified_records
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def summarize(root, reference, out, replicates=2000):
    if out.exists():
        raise FileExistsError("development summary may not be overwritten")
    if replicates < 100:
        raise ValueError("at least 100 paired bootstrap replicates are required")
    report = {
        "claim_scope": "posthoc_development_only", "test_access": False,
        "submission_allowed": False, "bootstrap_replicates": replicates,
        "bootstrap_scope": "paired_profiles_fixed_seeds_descriptive_not_confirmation",
        "datasets": {},
    }
    lines = [
        "# BotDMM Timing Development Comparison", "",
        "This is a bounded development comparison on previously inspected",
        "validation policy challenges. It is not untouched confirmation, a full",
        "original-paper reproduction, or evidence of SOTA.", "",
    ]
    for dataset in ("Twibot22", "Quadbot"):
        frozen, records = verified_records(root, dataset)
        prior, prior_records = verified_records(reference, dataset)
        if (frozen["seeds"] != prior["seeds"] or frozen["input_hashes"] != prior["input_hashes"]
                or frozen["timing"] != prior["timing"]
                or frozen["master_sha256"] != prior["master_sha256"]
                or frozen["mlp_reference_protocol_sha256"]
                != sha256_file(reference / dataset / "protocol.json")):
            raise ValueError("backbone and MLP reference inputs/protocol do not align")
        for key in frozen["mlp_reference"]["same_recipe_keys"]:
            if frozen["configuration"][key] != prior["configuration"][key]:
                raise ValueError("reference training recipe differs: " + key)
        policies = sorted({key[1] for key in records})
        if policies != sorted({key[1] for key in prior_records}):
            raise ValueError("reference heldout policies differ")
        methods = [*frozen["methods"], "erm_mlp"]
        tables, paired = {}, {}
        lines.extend([f"## {dataset}", ""])
        for index, policy in enumerate(policies):
            tables[policy] = {}
            block = {"seeds": {}}
            lines.extend([f"### {policy}", "", "| Method | Macro-F1 mean +/- seed SD | Parameters |",
                          "|---|---:|---:|"])
            for seed in frozen["seeds"]:
                block["seeds"][str(seed)] = {
                    method: {
                        "path": str((reference if method == "erm_mlp" else root)
                                    / dataset / f"seed{seed}" / policy / method
                                    / "challenge_predictions.jsonl"),
                        "pred_key": "prediction",
                    } for method in methods
                }
                control = prior_records[(seed, policy, "erm_mlp")]
                for method in frozen["methods"]:
                    row = records[(seed, policy, method)]
                    if any(row[key] != control[key] for key in (
                        "fitting_profiles", "selection_profiles", "challenge_profiles"
                    )):
                        raise ValueError("fold profile counts differ from reference")
            data = aligned_dataset(root / "unused_manifest.json", dataset, block)
            if len(data["classes"]) != frozen["class_count"]:
                raise ValueError("challenge classes differ from frozen protocol")
            paired[policy] = analyze_dataset(
                data, ["erm_mlp", "botdmm_without_timing"], replicates, 20261013 + index,
            )
            for method in methods:
                selected = prior_records if method == "erm_mlp" else records
                rows = [selected[(seed, policy, method)] for seed in frozen["seeds"]]
                parameters = {row["fit"]["parameter_count"] for row in rows}
                if len(parameters) != 1:
                    raise ValueError("method capacity differs across seeds")
                tables[policy][method] = {
                    "macro_f1_mean": float(np.mean([row["challenge"]["macro_f1"] for row in rows])),
                    "macro_f1_seed_sd": float(np.std(
                        [row["challenge"]["macro_f1"] for row in rows], ddof=1)),
                    "parameter_count": parameters.pop(),
                }
                item = tables[policy][method]
                lines.append(f"| {method} | {item['macro_f1_mean']:.4f} "
                             f"+/- {item['macro_f1_seed_sd']:.4f} | {item['parameter_count']} |")
            lines.append("")
            for comparison in ("botdmm_timing_minus_erm_mlp",
                               "botdmm_timing_minus_botdmm_without_timing"):
                delta = paired[policy]["aggregate_comparisons"][comparison]["macro_f1"]
                lines.append(f"- {comparison}: {delta['estimate'] * 100:+.2f} percentage points; "
                             f"descriptive profile interval "
                             f"[{delta['lower_95'] * 100:+.2f}, {delta['upper_95'] * 100:+.2f}].")
            lines.append("")
        report["datasets"][dataset] = {
            "runs_verified": len(records), "reference_runs_verified": len(prior_records),
            "tables": tables, "paired_intervals": paired,
            "source_protocol_sha256": sha256_file(root / dataset / "protocol.json"),
        }
    lines.extend([
        "## Boundaries", "",
        "Both BotDMM conditions use the same local release backbone and identical",
        "fold-local feature-kNN graphs. They differ in seven global and seven",
        "per-step timing inputs; the input projections therefore differ in size.",
        "These graphs are not real social edges. Challenge inference is unlabelled",
        "transductive within that challenge, not inductive.", "",
        "The MLP reference uses the same individual inputs, seeded profile folds,",
        "optimizer recipe and checkpoint rule, but not the graph-neighbor channel.",
        "The architectures and capacities are not matched. Auxiliary losses are",
        "disabled as in local release training; original-paper results are not",
        "reproduced by this adapted schedule.", "",
        "These descriptive intervals do not undo prior challenge exposure or",
        "multiple explored mechanisms. Every policy, seed and timing removal is",
        "retained regardless of the direction of its result. Old paper results,",
        "original tensors and the working manuscript remain unchanged.", "",
    ])
    out.mkdir(parents=True)
    write_new_json(out / "development_summary.json", report)
    with (out / "development_summary.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "reference-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    summarize(args.root, args.reference_dir, args.out_dir, args.replicates)


if __name__ == "__main__":
    main()
