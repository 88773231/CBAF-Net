# Strict rerun postprocessing

Run this only after all six corrected backbone and CBAF-Net jobs finish:

- `Twibot22`, seeds 42, 43, and 44;
- `Quadbot`, seeds 42, 43, and 44.

Each CBAF result directory must contain `metrics.json`,
`val_predictions.jsonl`, and `test_predictions.jsonl`. Each backbone training
directory must contain adjacent `training_summary.json` and `best_model.pt`
files. The optional numerical-only and style-only CBAF result directories use
the same prediction contract and set `feature_mode` to `num` or `style`.

From the repository root, run:

```bash
python analysis/postprocess_strict_rerun.py \
  --results-root experiments/strict_cbaf \
  --training-root experiments/strict_backbones \
  --ablation-root experiments/strict_cbaf_ablations \
  --output-dir results/strict_postprocessed \
  --bootstrap-replicates 5000 \
  --bootstrap-seed 20260930 \
  --require-declared-checkpoint-hash
```

The root arguments search recursively, so the six directories do not need a
particular parent layout. Repeated `--result-dir`, `--training-summary`, and
`--ablation-result-dir` arguments are available when explicit paths are safer.

The command rejects incomplete main runs, protocol mismatches, checkpoint hash
mismatches, malformed probabilities, changed sample order, and reported metrics
that cannot be reproduced from the prediction files. For each dataset and each
of the validation/test splits, the ordered labels, `sample_id` values,
`base_profile_id` values, and their joint tuples are SHA-256 hashed. All four
hashes must match exactly across seeds 42, 43, and 44. Missing optional
ablations are recorded as `not_available`; no values are substituted.

The output directory contains:

- `strict_rerun_summary.json`: complete per-seed audit plus means and population
  standard deviations for Accuracy, Macro-Precision, Macro-Recall, Macro-F1,
  MCC, and validation-selected fusion weights;
- `strict_rerun_table.csv`: compact manuscript-table input;
- `confusion_input.json`: seed-wise counts and mean row-normalized matrices;
- `reliability_input.json`: seed-wise reliability bins and population summaries;
- `profile_cluster_bootstrap.json`: deterministic 95% percentile intervals for
  BotDMM, BSE-only, CBAF-Net, and paired `CBAF-Net - BotDMM` differences;
- `predictions/`: the six validated, canonical figure-input directories;
- `figures/`: regenerated confusion and reliability figures, plus the ablation
  figure when all numerical-only and style-only runs are available.

Use `--skip-figures` for a data-only audit. The JSON/CSV results are still
produced. The fixed `w=0.5` control is always recomputed from the frozen
backbone and BSE probabilities. Numerical-only and style-only fusion results
are included only when complete artifacts exist for all three seeds.

For every main and ablation run, the command loads
`behavioral_statistics_expert.joblib`, reads the complete
`get_params(deep=False)` mapping, and rejects any deviation from the frozen BSE
configuration. This includes explicit `learning_rate=0.1` and
`min_samples_leaf=20` values rather than reliance on library defaults.

## Profile-cluster intervals

The bootstrap samples `base_profile_id` clusters with replacement. Every
replicate uses the same sampled profile sequence for BotDMM, BSE-only, and
CBAF-Net, which makes the reported CBAF-Net-minus-BotDMM interval paired. The
same sequence is also applied to seeds 42, 43, and 44. Per-seed intervals are
reported separately. For the aggregate interval, each replicate's metric is
computed independently for each fixed seed and the three values are averaged.
The three training seeds are not resampled. Macro metrics retain the full task
class denominator even when a bootstrap replicate omits a class.

Dataset-specific bootstrap RNG seeds are deterministically derived from
`--bootstrap-seed`. Repeating the command with unchanged inputs and arguments
therefore produces byte-identical `profile_cluster_bootstrap.json` output.

## Public identifiers and paths

The copied prediction files do not expose source `sample_id` or
`base_profile_id` values. They use dataset-local sequential identifiers based
on first occurrence. The same mapping is applied across all three seeds, and
the equality partition of the profile sequence is verified before writing, so
cluster membership and row order remain unchanged. The private mapping is not
written anywhere.

Checkpoint and artifact references in the public metrics and summary are
repository-relative logical paths. Source-machine, container, and user-profile
absolute paths are intentionally omitted.
