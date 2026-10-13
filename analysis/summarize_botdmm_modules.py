"""Descriptive module diagnostics, preserving every policy and seed."""

import argparse
from pathlib import Path

import numpy as np

from analysis.paired_profile_cluster_bootstrap import aligned_dataset, analyze_dataset
from analysis.summarize_policy_robust import verified_records
from src.evidence_repair import write_new_json
from src.provenance import sha256_file


def summarize(root, baseline, out, replicates=2000):
    if out.exists():
        raise FileExistsError("module summary may not be overwritten")
    if replicates < 100:
        raise ValueError("at least 100 descriptive profile replicates required")
    report = {
        "claim_scope": "posthoc_module_diagnostic", "submission_allowed": False,
        "test_access": False, "bootstrap_replicates": replicates,
        "bootstrap_scope": "paired_profiles_fixed_seeds_not_confirmatory", "datasets": {},
    }
    lines = [
        "# Fixed BotDMM Module Diagnostics", "",
        "These are module removals from the local timing-adapted backbone,",
        "not a new model or original-paper full reproduction. All validation",
        "challenges have already been inspected. No result establishes SOTA.", "",
    ]
    for dataset in ("Twibot22", "Quadbot"):
        frozen, records = verified_records(root, dataset)
        prior, controls = verified_records(baseline, dataset)
        for key in ("configuration", "input_hashes", "graph_input_hashes",
                    "timing", "seeds", "master_sha256"):
            if frozen[key] != prior[key]:
                raise ValueError("module diagnostic and baseline differ: " + key)
        if frozen["baseline_protocol_sha256"] != sha256_file(baseline / dataset / "protocol.json"):
            raise ValueError("baseline protocol identity differs")
        methods = ["botdmm_timing", *frozen["methods"]]
        policies = sorted({key[1] for key in records})
        if policies != sorted({key[1] for key in controls}):
            raise ValueError("baseline policy set differs")
        tables, paired = {}, {}
        lines.extend([f"## {dataset}", ""])
        for index, policy in enumerate(policies):
            block = {"seeds": {}}
            lines.extend([f"### {policy}", "", "| Method | Macro-F1 mean +/- seed SD |",
                          "|---|---:|"])
            tables[policy] = {}
            for seed in frozen["seeds"]:
                block["seeds"][str(seed)] = {
                    method: {
                        "path": str((baseline if method == "botdmm_timing" else root)
                                    / dataset / f"seed{seed}" / policy / method
                                    / "challenge_predictions.jsonl"),
                        "pred_key": "prediction",
                    } for method in methods
                }
                control = controls[(seed, policy, "botdmm_timing")]
                for method in frozen["methods"]:
                    row = records[(seed, policy, method)]
                    if any(row[key] != control[key] for key in (
                        "fitting_profiles", "selection_profiles", "challenge_profiles"
                    )):
                        raise ValueError("paired fold profile counts differ")
                    if (row["fit"]["parameter_count"] != control["fit"]["parameter_count"]
                            or row["fit"]["graph_fit"] != control["fit"]["graph_fit"]
                            or row["fit"]["graph_inner_selection"]
                            != control["fit"]["graph_inner_selection"]
                            or row["graph_challenge"] != control["graph_challenge"]):
                        raise ValueError("backbone layout or graph context differs")
            data = aligned_dataset(root / "unused_manifest.json", dataset, block)
            if len(data["classes"]) != frozen["class_count"]:
                raise ValueError("challenge class count differs")
            paired[policy] = analyze_dataset(data, ["botdmm_timing"], replicates, 20261013 + index)
            for method in methods:
                source = controls if method == "botdmm_timing" else records
                rows = [source[(seed, policy, method)] for seed in frozen["seeds"]]
                tables[policy][method] = {
                    "macro_f1_mean": float(np.mean([row["challenge"]["macro_f1"] for row in rows])),
                    "macro_f1_seed_sd": float(np.std(
                        [row["challenge"]["macro_f1"] for row in rows], ddof=1)),
                    "per_class_f1_mean": np.mean(
                        [row["challenge"]["per_class_f1"] for row in rows], axis=0,
                    ).tolist(),
                }
                table = tables[policy][method]
                lines.append(f"| {method} | {table['macro_f1_mean']:.4f} "
                             f"+/- {table['macro_f1_seed_sd']:.4f} |")
            lines.append("")
            for method in frozen["methods"]:
                delta = paired[policy]["aggregate_comparisons"][
                    method + "_minus_botdmm_timing"
                ]["macro_f1"]
                lines.append(f"- {method} minus full backbone: "
                             f"{delta['estimate'] * 100:+.2f} percentage points; descriptive interval "
                             f"[{delta['lower_95'] * 100:+.2f}, {delta['upper_95'] * 100:+.2f}].")
            lines.append("")
        report["datasets"][dataset] = {
            "runs_verified": len(records), "tables": tables, "paired_intervals": paired,
            "protocol_sha256": sha256_file(root / dataset / "protocol.json"),
        }
    lines.extend([
        "## Interpretation Boundary", "",
        "Both controls retain timing and the exact same fold-local kNN context.",
        "The structure control masks the output using the existing no_structure",
        "mode; its inactive parameters remain in the checkpoint. Context rows",
        "still enter training BatchNorm. This is not a completely graph-free",
        "training model or a claim of lower computation.", "",
        "The orthogonal control uses the existing no_orthogonal mode, retaining",
        "branch normalization. Jointly removing orthogonal processing after",
        "structure masking is algebraically redundant and is not run.", "",
        "The optimizer, seeded initialization, sampler, profile folds and inner",
        "selection rule are unchanged. All outcomes are retained. Descriptive",
        "profile intervals cannot remove prior outer challenge exposure.",
        "No additional architecture or weight search is authorized by this table.",
        "The main data, old released results and working manuscript are unchanged.", "",
    ])
    out.mkdir(parents=True)
    write_new_json(out / "development_summary.json", report)
    with (out / "development_summary.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "baseline-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    summarize(args.root, args.baseline_dir, args.out_dir, args.replicates)


if __name__ == "__main__":
    main()
