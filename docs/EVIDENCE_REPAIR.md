# Evidence Repair Scope

The reported `v1.1.0` CBAF-Net method and results are unchanged. The files
`src/evidence_model.py`, `src/evidence_repair.py`, and `src/botsscl_adapter.py`
are experimental development code, not an implemented revision of the paper's
reported method or evidence of state-of-the-art performance.

## Input Failure

The original feature export retains post/action types and text, but not
inter-post timing. A traditional uniform scheduler can therefore differ from
its human historical replay in timestamps while having identical exported
description, text, action, numerical, and style inputs. This is a representation
problem, not something a more elaborate classifier can resolve.

`dataset_pipeline/export_relative_timing.py` derives seven global and seven
per-step relative-gap statistics from the existing model-facing records. It
does not change accounts, labels, posts, grouped splits, or old feature files.
Absolute dates, epoch times, source fields, generator identifiers, and
subsecond serialization precision are excluded. Missing timestamps fail closed.
The new input is an explicit sidecar version, not a silent change to `v1.1.0`.

All new candidate models and comparators derive timing from the same source
records, but their input summaries differ. HGB uses the seven global timing
statistics; the temporal model and matched MLP also use the per-step statistics.
The pooled comparators use temporal means. These boundaries must accompany the
table. Comparisons against an old model lacking timing are not an
architecture-only improvement claim.

## Candidate Mechanisms

The temporal evidence candidate uses independent projections for each of the
exporter's five step-specific hashing coordinate systems. It models the
elementwise product and absolute difference between text and action
representations, integrates profile context, and encodes the result with a GRU.
A learned classwise gate combines the temporal branch with a neural statistics
branch. These are hypotheses to test, not established original contributions.

The removal experiments retrain models without cross-view interaction, without
the classwise gate, without surface/style statistics, and without relative
timing. Feature removal and architectural module removal are reported
separately. An ablation is allowed to outperform the full model.

## Comparators

- HistGradientBoosting on the same raw statistics, including relative timing.
- Multiclass logistic regression on profile, temporal-mean text/action, and
  standardized statistics. Temporal pooling is explicitly an input summary.
- A supervised MLP on the complete flattened text/action history and statistics.
- An independently implemented BotSSCL-inspired corruption/NT-Xent adapter
  on temporal-mean text/action, profile, and statistics, with joint multiclass
  fine-tuning. This input summary is not the full temporal history used by the
  candidate and matched MLP. The primary repository commit inspected is
  `ed3b17588fc2c271947aade266bb1357fc835f61`. The original representation,
  binary tasks, and training budget are not reproduced. It must remain labelled
  an architectural adapter, never a faithful BotSSCL result.

Graph-dependent recent methods are not run on invented social relations.
The existing kNN input is a feature-neighborhood graph, not a follower graph.
An independent external benchmark comparison remains a separate requirement.

## Frozen Development Protocol

`configs/evidence_repair_protocol.json` freezes seeds 42, 43, and 44, model
capacity, optimizer, training budget, stopping rule, and ablation list before
new validation results. The selection runner never constructs a test path.
Every run records ordered input hashes, source hashes, selected epochs,
validation predictions, and checkpoint artifacts. Output artifacts cannot be
overwritten.

The original held-out test has already been inspected repeatedly. Separating
new selection from test access prevents further direct test tuning; it cannot
retroactively restore an untouched final test. New validation results and
within-validation intervals are exploratory and must not be inserted into the
paper's held-out table as confirmatory findings.

`analysis/screen_unseen_policy.py` additionally excludes one traditional
scheduler policy from every fitting profile variant. It selects neural
checkpoints on a separate inner profile split drawn only from the remaining
original training profiles, then evaluates the excluded policy's original
validation profiles. This one-seed screen is a post-hoc development diagnostic,
not an untouched final test or cross-generator result.

## Remaining Gates

1. Determine whether the candidate has a reproducible advantage over the
   strongest matched simple comparator and whether its architectural modules
   contribute. Do not prescribe the outcome.
2. Diagnose source/style and scheduler-policy dependence using development
   probes. The current one-generator archive cannot support cross-generator
   claims, irrespective of how many profile splits are used.
3. Freeze a confirmatory protocol and obtain genuinely independent evidence
   before making external-real-account or broad SOTA claims.
4. Confirm source permissions, authorship/contributions, and approval of the
   final manuscript with the authors.
5. Only then revise the reported method, main results, figures, and submission
   materials together. No submission is authorized by a passing software test.
