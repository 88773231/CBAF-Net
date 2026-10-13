# Training Only Factor Fusion Outcomes

All 36 fixed one-seed fits and 36 checkpoint replays completed on
October 13, 2026. Original account records, labels, grouped dataset splits,
old tensors, 34 published result files and the working manuscript are preserved.

The mechanism and fixed boundary are in `FACTOR_TRAINING_SCREEN.md`.
Only original training tensors and existing training timing sidecars were
model inputs. Training-only profile partitions were stratified within the
remaining policies; original validation/test tensors were not used.

This is one seed (42) on previously used training profiles. It is neither
independent confirmation nor a replacement final method or SOTA result.

## Complete Macro-F1 Table

All values below are percentages. There is no across-seed uncertainty
estimate because this bounded feasibility screen has only one seed.

| View | Policy | Factor + residual | Flat fusion + residual | Factors without residual | ERM MLP | HGB | Adapted BotDMM |
|---|---|---:|---:|---:|---:|---:|---:|
| Twibot22 | burst_scheduler | 79.20 | 55.49 | 90.78 | 75.86 | 55.45 | 55.15 |
| Twibot22 | cyclic_reposter | 99.38 | 99.20 | 99.02 | 97.79 | 99.82 | 99.82 |
| Twibot22 | uniform_scheduler | 99.55 | 99.37 | 98.47 | 95.40 | 99.55 | 99.37 |
| Quadbot | burst_scheduler | 67.48 | 70.32 | 94.34 | 90.78 | 65.78 | 65.35 |
| Quadbot | cyclic_reposter | 97.46 | 97.86 | 95.85 | 96.99 | 98.73 | 98.80 |
| Quadbot | uniform_scheduler | 98.30 | 98.44 | 97.16 | 97.00 | 98.30 | 98.85 |

## Differences From All Removals and Controls

Signs are full candidate minus the listed comparator, in percentage points.

| View | Policy | Comparator | Full minus comparator |
|---|---|---|---:|
| Twibot22 | burst_scheduler | without_factor_fusion | +23.71 |
| Twibot22 | burst_scheduler | without_interaction_residual | -11.58 |
| Twibot22 | burst_scheduler | erm_mlp | +3.34 |
| Twibot22 | burst_scheduler | hgb_statistics | +23.75 |
| Twibot22 | burst_scheduler | botdmm_timing | +24.05 |
| Twibot22 | cyclic_reposter | without_factor_fusion | +0.18 |
| Twibot22 | cyclic_reposter | without_interaction_residual | +0.35 |
| Twibot22 | cyclic_reposter | erm_mlp | +1.59 |
| Twibot22 | cyclic_reposter | hgb_statistics | -0.45 |
| Twibot22 | cyclic_reposter | botdmm_timing | -0.45 |
| Twibot22 | uniform_scheduler | without_factor_fusion | +0.18 |
| Twibot22 | uniform_scheduler | without_interaction_residual | +1.08 |
| Twibot22 | uniform_scheduler | erm_mlp | +4.15 |
| Twibot22 | uniform_scheduler | hgb_statistics | +0.00 |
| Twibot22 | uniform_scheduler | botdmm_timing | +0.18 |
| Quadbot | burst_scheduler | without_factor_fusion | -2.85 |
| Quadbot | burst_scheduler | without_interaction_residual | -26.86 |
| Quadbot | burst_scheduler | erm_mlp | -23.30 |
| Quadbot | burst_scheduler | hgb_statistics | +1.70 |
| Quadbot | burst_scheduler | botdmm_timing | +2.12 |
| Quadbot | cyclic_reposter | without_factor_fusion | -0.40 |
| Quadbot | cyclic_reposter | without_interaction_residual | +1.60 |
| Quadbot | cyclic_reposter | erm_mlp | +0.47 |
| Quadbot | cyclic_reposter | hgb_statistics | -1.28 |
| Quadbot | cyclic_reposter | botdmm_timing | -1.34 |
| Quadbot | uniform_scheduler | without_factor_fusion | -0.14 |
| Quadbot | uniform_scheduler | without_interaction_residual | +1.14 |
| Quadbot | uniform_scheduler | erm_mlp | +1.30 |
| Quadbot | uniform_scheduler | hgb_statistics | -0.01 |
| Quadbot | uniform_scheduler | botdmm_timing | -0.55 |

## Parameter Counts

| View | Method | Parameters |
|---|---|---:|
| Twibot22 | factor_residual | 1340935 |
| Twibot22 | without_factor_fusion | 1341706 |
| Twibot22 | without_interaction_residual | 1274628 |
| Twibot22 | erm_mlp | 1089667 |
| Twibot22 | hgb_statistics | None |
| Twibot22 | botdmm_timing | 1423111 |
| Quadbot | factor_residual | 1341193 |
| Quadbot | without_factor_fusion | 1342221 |
| Quadbot | without_interaction_residual | 1274757 |
| Quadbot | erm_mlp | 1089796 |
| Quadbot | hgb_statistics | None |
| Quadbot | botdmm_timing | 1423304 |

## Diagnostic Decision

The full candidate is not a successful two-module method. On the burst
condition, omitting the residual increased Macro-F1 from 79.20% to 90.78%
in Twibot22 and from 67.48% to 94.34% in Quadbot. Retaining that residual
as a supported contribution would contradict this evidence.

The no-residual factor variant is a hypothesis worth reproducing, not an
automatically selected final model. It still trails adapted BotDMM on the
cyclic and uniform conditions in both views, including a 2.95-point cyclic
gap in Quadbot. No variant dominates the controls on every condition.

A useful next step is a prespecified replication of all six existing
conditions across seeds 42, 43 and 44, keeping this exact recipe and all
outcomes. It must not be portrayed as untouched confirmation. No additional
residual designs or weight search are justified by these challenge scores.

## Interpretation Boundary

Every removal and comparator is retained. A better removal is not hidden.
The flat-fusion control keeps factor auxiliary supervision; it removes
only taxonomic score combination. The residual control physically omits
the product/difference residual. Taxonomy targets describe archive
construction, not independently annotated intent or real-world factors.

The two evidence towers overlap because event hashes contain post text.
Step projections align different hashing namespaces, not pretrained
semantic embeddings. Parameter counts and graph information differ.
HGB receives global summaries; neural models receive the full step inputs.
Adapted BotDMM receives extra fold-local unlabelled feature-kNN context,
not social edges. Do not describe the methods as capacity matched.

The original validation-policy challenges are not retuned here. The
training-only challenge profiles participated in previous development;
their reassignment is not independent evaluation. A gain here cannot
establish generalization, reproducibility, model novelty or SOTA.

A final method still requires a frozen multi-seed evaluation protocol,
fair recent-method comparisons and genuinely independent confirmation.
No one-seed screen authorizes changing the paper's final claims or submission.
Raw profiles, predictions and checkpoints remain private.
