# CBAF-Net Reproducibility Release

This repository accompanies the working manuscript "CBAF Net Behavioral
Statistics Fusion for Controlled Multiclass Social Bot Detection".

CBAF-Net combines probabilities from a frozen BotDMM checkpoint with an independently fitted HistGradientBoosting behavioral-statistics expert. The scalar mixture weight is selected on the validation split and then frozen before test evaluation.
Here BotDMM is the matched multiclass, cross-entropy-only adaptation on
signed-hash inputs and a compatibility graph, not a faithful reproduction of
all original objectives and inputs. No new semantic encoder, routing
network, graph operator, or four-expert architecture is claimed.

## Evidence and version boundary

Latest work order (2026-10-15): stop method expansion after the completed
final bounded trial and prioritize evidence/manuscript/release closeout,
with a CCF-C fallback accepted. All 12 representation fits and replays
completed. Semantic features help the MLP by 0.4758 percentage points but
harm the cue-encoder control by 0.0683; the predeclared complete-study
continuation rules fail. See `docs/SEMANTIC_REPRESENTATION_RESULTS.md` and
the aggregate `results/semantic_representation/selection_summary.json`.
These are overlapping known-policy selection results, not new test
scores or a replacement original method. No training or paid GPU job
remains. The supported final method remains the original probability
mixture, with four-class Quadbot primary and three-class Twibot22 auxiliary.
The manuscript and synchronized revision are tracked separately from the
immutable `v1.1.0` evidence. Actual alignment status is recorded in
`docs/CLOSEOUT_20261015.md`; an aggregate development export alone does not
establish it.

Revision work is summarized in `docs/SUBMISSION_CONSOLIDATION.md`. The frozen
SAM/contrast candidate has a 96.01% four-class development Macro-F1, not a
replacement test score. Its paired module effects and three-class transfer
remain mixed; see `docs/FLAT_POLICY_RESULTS.md`. The completed 36-fit matched-input
BotSSCL-mechanism control scores 98.50% on Twibot22 and 94.81% on Quadbot,
compared with the frozen candidate's 93.09% and 96.01%. It is a mechanism
adaptation, not a faithful original-paper reproduction. These exposed
development results fail the requirement of superiority on both views; see
`docs/BOTSSCL_POLICY_CONTROL_RESULTS.md`. No revision work authorizes a new
submission or changes the released method described below.

A read-only, locally verified prediction diagnosis is recorded in
`docs/CONTROL_PREDICTION_DIAGNOSIS.md`. Candidate/control row and fitting
normalization identities match, while architecture choices remain
confounded. The 2026-10-14 user-authorized next step is a four-class,
known-policy same-encoder objective audit in
`docs/ENCODER_OBJECTIVE_DIAGNOSTIC_PLAN.md`, with nine new fits and three
reused supervised references. See its execution receipt for actual status.
Neither diagnostic promotes a candidate or replaces a test result.

The nine-fit audit completed with 12 known-policy checkpoint replays,
165 passing local tests and five passing targeted remote tests. The
encoder swap is not promoted; see
`docs/ENCODER_OBJECTIVE_DIAGNOSTIC_RESULTS.md`. No new challenge score,
manuscript replacement or public release was produced.

A separate five-block correspondence/transition hypothesis is implemented in
`src/trajectory_correspondence.py`; its frozen four-class feasibility plan is
`docs/TRAJECTORY_CORRESPONDENCE_PLAN.md`. Its twelve fits and twelve checkpoint
replays are complete and locally verified. The full architecture fails the
transition-removal gate: known-policy mean 98.71% versus 98.85% without
transition. See `docs/TRAJECTORY_CORRESPONDENCE_RESULTS.md`. Neither the best
removal nor these inner-selection scores replace the paper/release method.

The subsequent compact, transition-free check completed 30 new fits,
six cached controls and 36 checkpoint replays. All 188 local tests pass;
the receipt audit verifies every cell. Compact known-policy mean is
98.64%, versus 98.60% for correspondence removal, 98.17% supervised
encoder and 98.12% BotSSCL mechanism. The +0.047-point module mean is
not a consistently helpful effect; one seed mean reverses. See
`docs/COMPACT_CORRESPONDENCE_RESULTS.md`. The user's latest priority
is prompt completion with a lower-tier venue if necessary:
`docs/SUBMISSION_SPRINT.md` stops open-ended architecture searches.
No final-test, independent-evaluation, novelty or submission claim follows.

The publication baseline is the immutable `v1.1.0` strict rerun. `Twibot22`
(5,031 trajectories, three classes) is a nested view of the same 1,677 paired
profile groups as `Quadbot` (6,708 trajectories, four classes), not an
independent second source benchmark. The three-class view is not the original
TwiBot-22 benchmark. The LLM Bot and Full-stack Agent classes are controlled
simulations, not verified live-platform account labels.

Under the strict graph protocol, mean test Macro-F1 for BotDMM, BSE-only, and
CBAF-Net is respectively 0.7317, 0.7316, and 0.7332 on `Twibot22`, and
0.7844, 0.7911, and 0.7915 on `Quadbot`. Paired profile-cluster confidence
intervals for CBAF-Net's Macro-F1 differences from either standalone method
include zero. BSE-only has higher Accuracy and MCC than the fusion on both
views. These results do not establish that fusion is necessary, a new
state-of-the-art detector, or transfer to new generators or platforms.

`results/strict_postprocessed/` remains the unchanged `v1.1.0` result set.
The post-rejection paired comparison against BSE-only is separately
recomputable using `analysis/paired_profile_cluster_bootstrap.py` and
`configs/strict_paired_comparison.json`. Validation-only experimental model
branches are not part of the reported CBAF-Net method.

The development repair work is documented separately in
`docs/EVIDENCE_REPAIR.md`. Its temporal-input sidecars and candidate models are
exploratory, use train/validation only, and do not replace the published test
table. An input repair must be shared by every new comparator; it cannot be
presented as an architecture gain against a comparator denied that input.

The fixed policy-objective study and its complete negative/mixed outcomes
are separate development artifacts in `docs/POLICY_ROBUST_STUDY.md` and
`docs/POLICY_ROBUST_RESULTS.md`. The combination is not promoted to the paper
method: it helps one four-class burst-policy challenge but harms the
three-class counterpart. No main test table or release baseline is replaced.

## Repository layout

- `src/run_cbaf.py`: canonical CBAF-Net evaluator used by the paper.
- `src/botdmm.py`: frozen BotDMM backbone definition required to load the reported checkpoints.
- `src/data_loader.py`: tensor-view loaders and batching logic.
- `src/layers/`: backbone layers used by `src/botdmm.py`.
- `analysis/`: profile-cluster bootstrap and paper-figure recomputation scripts.
- `dataset_pipeline/`: schema validation, feature export, view construction, and release audits.
- `configs/`: a non-sensitive controlled-view provenance example.
- `docs/`: data card and reproducibility boundary.
- `docs/figures/cbaf_architecture.svg`: editable method schematic; the PNG is
  its manuscript rendering. The account scene, snippets, and plotted
  probabilities/validation curve are illustrative, not measured results.
  `docs/figures/README.md` records the drawing's scope and visual references.

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

To recompute the paired differences against both BotDMM and BSE-only from
the published prediction files without retraining:

```bash
python analysis/paired_profile_cluster_bootstrap.py \
  --manifest configs/strict_paired_comparison.json \
  --out results/paired_profile_cluster_bootstrap.json \
  --replicates 5000 \
  --seed 20261010
```

The output is an additional post-rejection audit; it does not replace the
original `v1.1.0` result manifest or change any prediction.

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
The TwiBot-22 paper identifies a CC BY-NC-ND 4.0 dataset license, distinct
from its MIT code license. See `docs/SOURCE_PERMISSIONS.md` for the public
release boundary and pending author confirmations.
