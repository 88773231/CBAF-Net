# Reproducibility Manifest

This manifest defines the single publication protocol for CBAF-Net. Numeric
results are generated from audited prediction artifacts; they are not copied
into this document by hand.

## Publication protocol

- Datasets: `Twibot22` (three classes) and `Quadbot` (four classes).
- Evaluation seeds: 42, 43, and 44.
- Splits: profile-group disjoint train, validation, and test partitions.
- Graph: split-local directed cosine kNN with `k=10`; self-pairs and candidates
  sharing `base_profile_id` are excluded.
- Runtime graph direction: selected neighbor to target.
- Batching: every seed target is expanded with all ten one-hop neighbors;
  supervised loss and reported metrics are computed only on seed targets.
- Backbone tabular preprocessing: per-column population standardization fitted
  on training data only. Zero-variance columns retain their schema positions
  and map to zero.
- Behavioral-statistics expert: fitted on the raw exported numerical/style
  values from the training split.
- Fusion: selected using validation data and frozen before test inference.
- Primary metric: Macro-F1. Accuracy, macro-precision, macro-recall, and MCC are
  also recomputed from prediction files.
- Across-seed variation: population standard deviation over three seeds.
- Test uncertainty: paired resampling of complete profile groups, preserving
  all controlled variants of a profile in each bootstrap draw.

There is no separate "default" graph result in the publication protocol. Any
checkpoint or prediction produced by the earlier induced-mini-batch graph path
is incompatible with this release.

## Generated result artifacts

The canonical postprocessor consumes exactly six main runs and their adjacent
training summaries. It verifies graph/preprocessing records, checkpoint hashes,
sample identity across seeds, prediction alignment, and recomputed metrics.
The public output contains:

| Package location | Role |
|---|---|
| `results/strict_postprocessed/strict_rerun_summary.json` | Audited per-seed records and aggregate statistics. |
| `results/strict_postprocessed/strict_rerun_table.csv` | Compact manuscript-table input. |
| `results/strict_postprocessed/figures/` | Confusion, reliability, and ablation figures regenerated from audited predictions. |
| `results/strict_postprocessed/predictions/` | Reindexed validation/test probabilities for metric recomputation. |
| `results/strict_postprocessed/SHA256SUMS` | Integrity hashes for distributed result artifacts. |

No raw data, unrestricted generated records, original account identifiers, or
model checkpoints are distributed in this results package.

## Recorded training configuration

- Backbone: `embedding_dimension=128`, `feature_dim=128`, five temporal steps,
  dropout `0.30`, and deterministic temporal mean pooling.
- Optimizer: AdamW with learning rate `5e-5` and weight decay `5e-4`.
- Loader: seed batch size `64`, maximum `60` epochs, and validation Macro-F1
  early stopping with patience `10`.
- BSE: `HistGradientBoostingClassifier(learning_rate=0.1,
  min_samples_leaf=20, max_iter=300, max_leaf_nodes=15,
  l2_regularization=1.0, random_state=seed)`. All shallow estimator parameters
  are recorded and verified against the serialized estimator before release.
- Auxiliary contrastive, prototype, domain-adversarial, and gate losses are not
  part of the reported CBAF-Net objective.

The exact Python, package, CUDA, cuDNN, GPU, and operating-system versions are
stored with the frozen release environment manifest.

## Reproduction boundary

The public code supports inspection of the tensor-view schema, graph contract,
training path, and evaluation pipeline. It does not by itself reconstruct the
controlled generated classes; that requires restricted source material and
retained generation records that are not distributed here. The public results
package supports metric, figure, calibration, and profile-cluster bootstrap
recomputation without exposing restricted records. Neither package establishes
cross-platform, cross-generator, temporal, or live account generalization.

- The anonymous review package omits the public repository URL and author
  account identity. A permanent code archive URL can be added after review.
- The corrected public reference is tagged `v1.1.0`. The pre-existing
  `v1.0.0` tag is not modified.
- The MIT license covers only author-owned code and documentation. It does not
  relicense TwiBot-22-derived material, third-party models, or generated data.
- Funding, competing-interest, ethics, and source-access declarations remain
  author-level submission metadata and must be verified before submission.
