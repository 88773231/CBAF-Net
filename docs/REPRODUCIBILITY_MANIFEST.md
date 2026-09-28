# Reproducibility Manifest

## Frozen protocol

- Evaluation seeds: 42, 43, 44.
- Model selection: validation-selected, global decision-level probability fusion.
- The method is not a per-sample dynamic router.
- Primary metric: Macro-F1; accuracy and MCC are also recorded.
- Test uncertainty: paired profile-level resampling, with 280 test profile groups.
- Exact formal-versus-strict comparison: 5,000 bootstrap resamples per dataset and
  seed using Python `random.Random`.
- Population standard deviation is reported across the three training seeds.

## Frozen Macro-F1 summary

| Protocol | Dataset | BotDMM | CBAF-Net |
|---|---|---:|---:|
| Formal | Twibot22 | 0.7059 +/- 0.0060 | 0.7255 +/- 0.0020 |
| Formal | Quadbot | 0.7723 +/- 0.0022 | 0.7884 +/- 0.0025 |
| Strict edge policy | Twibot22 | 0.7060 +/- 0.0060 | 0.7246 +/- 0.0009 |
| Strict edge policy | Quadbot | 0.7726 +/- 0.0026 | 0.7884 +/- 0.0025 |

## Artifact provenance

| Package location | Origin and role |
|---|---|
| `reports/formal_fusion_controls_profile_audit.*` | Recomputed from frozen formal probability files; no retraining. |
| `reports/formal_vs_strict_profile_bootstrap_exact.json` | Exact 5,000-resample paired profile comparison. |
| `audits/formal/release_contract_*.json` | Frozen release contract, composition, split, and feature checks. |
| `audits/formal/shortcut_audit.json` | Frozen metadata/style diagnostic and alignment contract. |
| `audits/strict/edge_summary.json` | Strict edge counts and zero same-profile fraction. |
| `configs/formal_training_config.json` | Machine-readable formal training, BSE, and fusion-selection configuration. |
| `results/formal/` | Formal metrics and reindexed validation/test probabilities. |
| `results/strict/` | Strict-edge metrics and reindexed validation/test probabilities. |

## Recomputable analyses

The distributed prediction JSONL files contain `row_index`, anonymous
`profile_group`, gold label, BotDMM prediction/probability, BSE
prediction/probability, and fused prediction/probability. These values are
sufficient to recompute confusion matrices, accuracy, Macro-F1, MCC, fixed-average
fusion, validation-selected fusion evaluation, and paired profile-level bootstrap
comparisons. The package does not contain training data or checkpoints, so it does
not yet support end-to-end retraining.

## Recorded training configuration

The frozen evaluator and manuscript record the following values:

- BotDMM backbone: `embedding_dimension=128`, `feature_dim=128`,
  `num_temporal_steps=5`, `dropout=0.30`, `temperature=0.10`, `alpha=0.50`.
- Backbone loader: batch size `64`; maximum `60` epochs; validation Macro-F1
  early-stopping patience `10`; evaluation seeds `42`, `43`, and `44`.
- BSE: `HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=15,
  l2_regularization=1.0, random_state=seed)` fitted only on the training split.
- Fusion: candidate weights `(0, 0.05, Ellipsis, 1.0)`; choose validation Macro-F1
  maximum and freeze the scalar before test evaluation.
- Formal backbone objective: the recorded evaluator uses the BotDMM base path
  (`ablation_mode=base`, `enhanced=False`); no unreported auxiliary loss is
  included in the CBAF-Net definition.

## Strict edge policy

For each split independently, graph vectors form directed cosine kNN edges with
`k=10`. The strict sensitivity view excludes candidate neighbors sharing the same
profile group and still emits exactly ten outgoing edges per row. No cross-split
edges are permitted. See `audits/strict/EDGE_POLICY.md` and
`audits/strict/edge_summary.json`.

## Package transformation

Prediction values are copied without numerical modification. Internal sample IDs
are removed. Internal base-profile IDs are deterministically reindexed within each
dataset as package-local `profile_group` values. Path-bearing audit fields are
rewritten to `not_distributed/...`; scientific counts and metrics are unchanged.

## Reproduction boundary and author actions

- This anonymous package supports recomputation of the reported metrics and audits from frozen prediction artifacts; it does not support end-to-end retraining.
- A public code archive URL and immutable commit or release tag have not yet been supplied by the authors.
- Exact Python, PyTorch, CUDA, scikit-learn, graph-library, operating-system, GPU/CPU, memory, and runtime details are not recoverable from the frozen prediction package and must be exported from the formal training environment before an end-to-end release claim is made.
- Controlled-generator checkpoints, prompts, decoding settings, simulator versions, seed policy, and filtering or deduplication configuration must be published or access-bounded explicitly in the final code/data release.
- Data licenses, permitted access, privacy, competing-interest, funding, and institutional ethics declarations remain author-verified submission metadata rather than anonymous prediction-package fields.
