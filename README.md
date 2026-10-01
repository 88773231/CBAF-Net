# CBAF-Net Reproducibility Release

This repository accompanies the manuscript "CBAF-Net: Auditable Fusion for Controlled Multiclass Detection of LLM-Driven and Agentic Social Accounts"

CBAF-Net combines probabilities from a frozen BotDMM checkpoint with an independently fitted HistGradientBoosting behavioral-statistics expert. The scalar mixture weight is selected on the validation split and then frozen before test evaluation.

## Repository layout

- `src/run_cbaf.py`: canonical CBAF-Net evaluator used by the paper.
- `src/botdmm.py`: frozen BotDMM backbone definition required to load the reported checkpoints.
- `src/data_loader.py`: tensor-view loaders and batching logic.
- `src/layers/`: backbone layers used by `src/botdmm.py`.
- `analysis/`: profile-cluster bootstrap and paper-figure recomputation scripts.
- `dataset_pipeline/`: schema validation, feature export, view construction, and release audits.
- `configs/`: a non-sensitive controlled-view provenance example.
- `docs/`: data card and reproducibility boundary.

The backbone's temporal reducer is deterministic mean pooling followed by
LayerNorm and dropout. The implementation names it `TemporalMeanPooling`; the
existing `temporal_pooling.*` checkpoint parameter keys are unchanged.

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The exact environment used for the reported runs is documented separately with the submission artifacts. The version ranges above are intended for code inspection and local reproduction, not as a claim that every dependency combination reproduces bit-identical checkpoints.

## Run CBAF-Net

The evaluator requires a legally accessible strict tensor view and its matching
BotDMM checkpoint. Strict views exclude kNN candidates sharing the same
`base_profile_id`. The loader expands each target batch with all ten selected
neighbors and sends graph messages from selected neighbors to their target
rows. It never searches the test labels for a fusion weight.

Train a release backbone checkpoint with the same graph execution contract:

```bash
python -m src.train_botdmm \
  --dataset Twibot22 \
  --data_dir /path/to/strict/Twibot22 \
  --seed 42 \
  --out_dir checkpoints/Twibot22_seed42
```

The training command validates ten neighbors per target, split-local indices,
and zero same-profile edges before the first optimizer step. Supervised loss is
computed only for seed rows; neighbor rows provide one-hop graph context. The
backbone fits per-column means and population standard deviations on the raw
training numerical/style tensors, reuses those statistics for validation and
test, and maps training-constant columns to zero. The BSE continues to consume
the raw exported values because its tree splits do not require scaling.

```bash
python -m src.run_cbaf \
  --dataset Twibot22 \
  --data_dir /path/to/strict/Twibot22 \
  --checkpoint /path/to/Twibot22_seed42/best_model.pt \
  --seed 42 \
  --out_dir results/predictions/Twibot22_seed42
```

Use `--dataset Quadbot` with the matching four-class checkpoint. Lowercase
spellings remain accepted only for command-line compatibility. The public view
names and default output directories are `Twibot22` and `Quadbot`; internal
schema keys such as `quadbot_3class`, `quadbot_4class`, and `quadbot-v3-*` are
retained solely to read earlier controlled-pipeline records. Repeat with seeds
42, 43, and 44 to reproduce the paper's aggregation protocol.

Checkpoints trained with the earlier induced-mini-batch edge path or row-wise
tabular normalization are not interchangeable with corrected checkpoints. New
`best_model.pt` files are versioned envelopes containing the model state, graph
protocol, dataset/seed identity, and the train-fitted preprocessing signature.
All graph-dependent metrics must be regenerated after adopting this execution
contract. The evaluator also requires the `training_summary.json` written beside
a corrected checkpoint. That summary records the SHA-256 digest of the final
`best_model.pt`; the evaluator recomputes it and rejects a mismatched checkpoint.
`--allow_unverified_checkpoint` exists only for
diagnostic audits; outputs produced with that override must not be reported as
corrected results.

Run the graph regression suite with:

```bash
python -m unittest discover -s tests -v
```

The output directory contains:

- `metrics.json`;
- `val_predictions.jsonl`;
- `test_predictions.jsonl`;
- `behavioral_statistics_expert.joblib`.

## Recompute audits and figures

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

This is the canonical release chain. It audits all six runs, verifies the
serialized behavioral-statistics expert, recomputes metrics and 5,000-draw
profile-cluster intervals, reindexes public prediction identifiers, and writes
the confusion and reliability figures. It also writes the ablation figure when
all numerical-only and style-only runs are supplied. No figure step retrains a
model or tunes on the test split.

The `v1.1.0` release includes the audited public outputs under
`results/strict_postprocessed/`. Run `sha256sum -c SHA256SUMS` from that
directory to verify every distributed result artifact before recomputation.

The BSE is frozen as `HistGradientBoostingClassifier` with
`learning_rate=0.1`, `min_samples_leaf=20`, `max_iter=300`,
`max_leaf_nodes=15`, and `l2_regularization=1.0`. The evaluator records every
shallow estimator parameter, and the postprocessor verifies that complete
mapping against each serialized `.joblib` artifact.

## Data and claim boundary

`Twibot22` and `Quadbot` are hybrid controlled benchmarks. Human trajectories are historical replays derived from TwiBot-22 material; the automated strategy variants are controlled transformations or simulations. LLM Bot and Full-stack Agent are not verified labels for live platform accounts.

The repository does not contain raw TwiBot-22 records, original account identifiers, private checkpoints, credentials, API keys, or unrestricted generated records. Users must obtain restricted source material under the original provider terms. The reported results do not establish cross-platform, cross-generator, inductive-graph, or real-account generalization.

`configs/provenance.example.json` documents the intended paired-view lineage
without asserting unverified source terms or generator details. Every field
marked `verification_required` must be completed from the authors' retained
records before claiming end-to-end public reproducibility.

## License

The MIT license in `LICENSE` covers only code and documentation owned by the authors. It does not relicense third-party source data, pretrained models, or checkpoints.
