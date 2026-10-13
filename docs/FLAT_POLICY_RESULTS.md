# Flat Policy Training Results

## Scope and Verification

The frozen `configs/flat_policy_screen.json` recipe was trained at
commit `3637eed713cd85478d9cc1b680ac75d86615fcac`. The study finished
October 13, 2026 at 22:54:26 Asia/Shanghai, approximately 5 minutes
41 seconds after launch. All 102 new fits and selected-checkpoint
replays completed: 54 main fits and 48 seed/partition-axis fits.
All 72 matching cached results were verified. The complete local and
remote regression suites both passed 145 tests. All 10 collected
evidence files matched their completion receipt's SHA-256 hashes.

Quadbot was frozen as the primary task and Twibot22 as transfer.
Primary Macro-F1 pools all three excluded-policy prediction sets
once per profile/class within each seed. All planned conditions are
retained; the axis probes cannot replace the main aggregate.
The main seeds 42/43/44 also change inner profile partitions.
All scores below are percentages; SD is sample SD across three runs.

These challenges already have development exposure. No original
validation/test tensor was evaluated. New axes and confidence intervals
do not restore independent evidence. Both controlled views share
profiles, so they are not independent real-world datasets. Dataset
master, prior studies, release results and working Word were preserved.

The inference model is the same ordinary MLP as the matched control.
The rejected pair margin and cosine head are explicitly absent.
SAM and supervised contrast are established methods, not new inference
modules or a new general algorithm.

## Primary Overall Macro-F1

| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---:|---:|---:|---:|---:|
| Twibot22 | flat_policy | 91.60 | 93.92 | 93.75 | 93.09 | 1.29 |
| Twibot22 | without_policy_contrast | 95.63 | 93.89 | 90.38 | 93.30 | 2.67 |
| Twibot22 | two_pass_erm | 94.42 | 93.11 | 93.88 | 93.80 | 0.66 |
| Twibot22 | without_flatness | 94.00 | 93.99 | 95.88 | 94.62 | 1.09 |
| Twibot22 | erm_mlp | 90.32 | 89.78 | 95.08 | 91.73 | 2.92 |
| Twibot22 | hgb_statistics | 88.33 | 88.33 | 88.33 | 88.33 | 0.00 |
| Twibot22 | botdmm_timing | 88.15 | 88.11 | 88.20 | 88.15 | 0.04 |
| Quadbot | flat_policy | 96.08 | 95.81 | 96.13 | 96.01 | 0.17 |
| Quadbot | without_policy_contrast | 92.41 | 96.54 | 96.24 | 95.06 | 2.30 |
| Quadbot | two_pass_erm | 94.68 | 95.48 | 95.89 | 95.35 | 0.61 |
| Quadbot | without_flatness | 94.71 | 96.26 | 92.58 | 94.52 | 1.85 |
| Quadbot | erm_mlp | 94.98 | 96.90 | 93.01 | 94.96 | 1.95 |
| Quadbot | hgb_statistics | 90.18 | 90.10 | 90.34 | 90.21 | 0.12 |
| Quadbot | botdmm_timing | 90.27 | 90.54 | 90.43 | 90.41 | 0.14 |

`without_flatness` is the exact cached ordinary MLP/contrast-only recipe,
originally named `without_pair_margin` in the paired-policy study.
That source identity, checkpoint and prediction provenance remain
explicit. This equivalent removal is not a renamed final method.
One-pass ERM removes both. Two-pass ERM performs two ordinary updates
per same batch as a per-batch gradient-evaluation control.

## Paired Overall Differences

Full minus comparator, in percentage points. Calculations use
unrounded scores. Positive counts are descriptive, not significance tests.

| View | Comparator | Mean Delta | Delta SD | Positive Seeds |
|---|---|---:|---:|---:|
| Twibot22 | without_policy_contrast | -0.21 | 3.71 | 2/3 |
| Twibot22 | two_pass_erm | -0.71 | 1.89 | 1/3 |
| Twibot22 | without_flatness | -1.53 | 1.27 | 0/3 |
| Twibot22 | erm_mlp | +1.36 | 2.74 | 2/3 |
| Twibot22 | hgb_statistics | +4.76 | 1.29 | 3/3 |
| Twibot22 | botdmm_timing | +4.94 | 1.30 | 3/3 |
| Quadbot | without_policy_contrast | +0.94 | 2.38 | 1/3 |
| Quadbot | two_pass_erm | +0.66 | 0.65 | 3/3 |
| Quadbot | without_flatness | +1.49 | 2.00 | 2/3 |
| Quadbot | erm_mlp | +1.04 | 2.11 | 2/3 |
| Quadbot | hgb_statistics | +5.80 | 0.10 | 3/3 |
| Quadbot | botdmm_timing | +5.60 | 0.29 | 3/3 |

## Seed and Partition Axes

Fixed-partition pairs are (training seed, partition seed)
`(42,42)/(43,42)/(44,42)`. Fixed-training-seed pairs are
`(42,42)/(42,43)/(42,44)`. The common 42/42 results are reused.
Training seed changes initialization, minibatches and dropout together;
the axes do not isolate pure initialization or measure a full factorial
interaction. They are exposed development probes, not fresh tests.

| View | Axis | Method | Pair 1 | Pair 2 | Pair 3 | Mean | Seed SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | fixed_partition | flat_policy | 91.60 | 90.63 | 94.46 | 92.23 | 2.00 |
| Twibot22 | fixed_partition | erm_mlp | 90.32 | 92.93 | 89.01 | 90.75 | 2.00 |
| Twibot22 | fixed_training_seed | flat_policy | 91.60 | 95.57 | 95.80 | 94.33 | 2.37 |
| Twibot22 | fixed_training_seed | erm_mlp | 90.32 | 95.14 | 93.73 | 93.06 | 2.48 |
| Quadbot | fixed_partition | flat_policy | 96.08 | 95.90 | 95.61 | 95.86 | 0.24 |
| Quadbot | fixed_partition | erm_mlp | 94.98 | 93.92 | 94.66 | 94.52 | 0.55 |
| Quadbot | fixed_training_seed | flat_policy | 96.08 | 95.35 | 95.01 | 95.48 | 0.55 |
| Quadbot | fixed_training_seed | erm_mlp | 94.98 | 93.38 | 90.37 | 92.91 | 2.34 |

Quadbot full-minus-ERM means are +1.34 points (paired SD 0.56)
on the fixed-partition axis and +2.57 (paired SD 1.85) on the
fixed-training-seed axis, positive in 3/3 pairs on each axis.
Twibot22 differences are +1.48 (paired SD 3.88), positive in 2/3,
and +1.26 (paired SD 0.82), positive in 3/3, respectively.
Do not pool these overlapping axes as independent additional samples.

## Conditional Profile Uncertainty

A separate read-only analysis resamples 1,118 profile groups with all
class variants and all three selected-model predictions kept together.
Use 2,000 shared draws, RNG seed 20261013, percentile 95% intervals
of the mean paired difference across the three already-selected runs.
No p-values or multiplicity adjustment are reported.

| View | Comparator | Point Delta | Conditional 95% Interval |
|---|---|---:|---:|
| Twibot22 | without_policy_contrast | -0.21 | [-0.60, +0.17] |
| Twibot22 | two_pass_erm | -0.71 | [-1.21, -0.25] |
| Twibot22 | without_flatness | -1.53 | [-1.99, -1.11] |
| Twibot22 | erm_mlp | +1.36 | [+0.97, +1.76] |
| Twibot22 | hgb_statistics | +4.76 | [+3.86, +5.70] |
| Twibot22 | botdmm_timing | +4.94 | [+4.06, +5.84] |
| Quadbot | without_policy_contrast | +0.94 | [+0.69, +1.17] |
| Quadbot | two_pass_erm | +0.66 | [+0.25, +1.07] |
| Quadbot | without_flatness | +1.49 | [+1.14, +1.83] |
| Quadbot | erm_mlp | +1.04 | [+0.76, +1.32] |
| Quadbot | hgb_statistics | +5.80 | [+5.01, +6.63] |
| Quadbot | botdmm_timing | +5.60 | [+4.80, +6.39] |

These intervals are conditional on the three selected seed/fold models.
They exclude method-development selection, new training seeds or folds,
new generator populations and new scheduler populations. They must not
replace the large paired seed variability or imply independent
significance after repeated development on these profiles. The contrast
interval excluding zero in Quadbot does not cancel the fact that only
one of three paired training runs has a positive contrast difference.

Bootstrap JSON SHA-256:
`f802a36fcf26f162a61b797af54f50667dcd9d89c1546b86939ddab4d96b97d8`.

## All Main Policy Conditions

| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | burst | flat_policy | 78.68 | 85.60 | 86.30 | 83.52 | 4.21 |
| Twibot22 | burst | without_policy_contrast | 91.90 | 87.59 | 79.03 | 86.17 | 6.55 |
| Twibot22 | burst | two_pass_erm | 87.68 | 83.12 | 84.86 | 85.22 | 2.30 |
| Twibot22 | burst | without_flatness | 88.90 | 87.10 | 93.79 | 89.93 | 3.46 |
| Twibot22 | burst | erm_mlp | 75.86 | 71.25 | 90.69 | 79.27 | 10.16 |
| Twibot22 | burst | hgb_statistics | 55.45 | 55.45 | 55.45 | 55.45 | 0.00 |
| Twibot22 | burst | botdmm_timing | 55.15 | 55.23 | 55.23 | 55.20 | 0.05 |
| Twibot22 | cyclic | flat_policy | 97.97 | 98.49 | 97.87 | 98.11 | 0.33 |
| Twibot22 | cyclic | without_policy_contrast | 97.94 | 98.05 | 95.36 | 97.12 | 1.53 |
| Twibot22 | cyclic | two_pass_erm | 98.75 | 98.84 | 98.75 | 98.78 | 0.05 |
| Twibot22 | cyclic | without_flatness | 97.52 | 98.93 | 98.21 | 98.22 | 0.71 |
| Twibot22 | cyclic | erm_mlp | 97.79 | 98.31 | 98.13 | 98.07 | 0.27 |
| Twibot22 | cyclic | hgb_statistics | 99.82 | 99.82 | 99.82 | 99.82 | 0.00 |
| Twibot22 | cyclic | botdmm_timing | 99.82 | 99.47 | 99.82 | 99.70 | 0.21 |
| Twibot22 | uniform | flat_policy | 96.85 | 97.21 | 96.74 | 96.94 | 0.25 |
| Twibot22 | uniform | without_policy_contrast | 96.94 | 95.76 | 95.56 | 96.09 | 0.74 |
| Twibot22 | uniform | two_pass_erm | 96.56 | 96.56 | 97.55 | 96.89 | 0.57 |
| Twibot22 | uniform | without_flatness | 95.40 | 95.67 | 95.57 | 95.54 | 0.14 |
| Twibot22 | uniform | erm_mlp | 95.40 | 96.58 | 96.30 | 96.09 | 0.62 |
| Twibot22 | uniform | hgb_statistics | 99.55 | 99.55 | 99.55 | 99.55 | 0.00 |
| Twibot22 | uniform | botdmm_timing | 99.37 | 99.46 | 99.37 | 99.40 | 0.05 |
| Quadbot | burst | flat_policy | 95.62 | 93.25 | 93.40 | 94.09 | 1.33 |
| Quadbot | burst | without_policy_contrast | 83.34 | 95.47 | 94.36 | 91.06 | 6.70 |
| Quadbot | burst | two_pass_erm | 88.86 | 91.66 | 91.76 | 90.76 | 1.65 |
| Quadbot | burst | without_flatness | 90.03 | 94.51 | 83.97 | 89.50 | 5.29 |
| Quadbot | burst | erm_mlp | 90.78 | 95.40 | 85.30 | 90.49 | 5.05 |
| Quadbot | burst | hgb_statistics | 65.78 | 65.74 | 66.00 | 65.84 | 0.14 |
| Quadbot | burst | botdmm_timing | 65.35 | 65.97 | 65.71 | 65.68 | 0.31 |
| Quadbot | cyclic | flat_policy | 96.72 | 97.11 | 97.26 | 97.03 | 0.28 |
| Quadbot | cyclic | without_policy_contrast | 96.50 | 97.07 | 96.42 | 96.67 | 0.35 |
| Quadbot | cyclic | two_pass_erm | 97.45 | 97.98 | 97.19 | 97.54 | 0.40 |
| Quadbot | cyclic | without_flatness | 96.22 | 97.46 | 96.16 | 96.62 | 0.74 |
| Quadbot | cyclic | erm_mlp | 96.99 | 97.85 | 96.57 | 97.14 | 0.65 |
| Quadbot | cyclic | hgb_statistics | 98.73 | 98.73 | 98.60 | 98.69 | 0.08 |
| Quadbot | cyclic | botdmm_timing | 98.80 | 99.20 | 98.93 | 98.98 | 0.20 |
| Quadbot | uniform | flat_policy | 95.88 | 97.01 | 97.68 | 96.86 | 0.91 |
| Quadbot | uniform | without_policy_contrast | 96.30 | 97.08 | 97.88 | 97.09 | 0.79 |
| Quadbot | uniform | two_pass_erm | 97.40 | 96.62 | 98.64 | 97.55 | 1.02 |
| Quadbot | uniform | without_flatness | 97.62 | 96.79 | 96.62 | 97.01 | 0.53 |
| Quadbot | uniform | erm_mlp | 97.00 | 97.42 | 96.42 | 96.94 | 0.50 |
| Quadbot | uniform | hgb_statistics | 98.30 | 98.03 | 98.71 | 98.35 | 0.34 |
| Quadbot | uniform | botdmm_timing | 98.85 | 98.72 | 98.99 | 98.85 | 0.13 |

Short names denote `burst_scheduler`, `cyclic_reposter` and
`uniform_scheduler`. All class-specific and axis-run scores,
selected epochs, optimizer-update/evaluation counts, elapsed times
and verified prediction provenance remain in the archived JSON:
`b8198f9bc11542fe1ba1ce01312316306f4d344daf5eb337dd7c24aa4abf006c`.

## Capacity and Compute

Every new neural condition and both objective removals have exactly
1,089,667 parameters in Twibot22 and 1,089,796 in Quadbot.
No extra graph context or prediction-time metadata is used.
HGB has global summary inputs; adapted BotDMM has extra unlabelled
feature-kNN context. It is not a faithful original reproduction.

SAM evaluates two gradients and makes one update per batch; two-pass
ERM evaluates two gradients and makes two updates. They share epoch
limits but need not stop at the same epoch or use equal total compute.
Mean main-fit training times are 5.54/4.37/3.33 seconds in Twibot22
and 5.83/5.96/3.56 in Quadbot for full/SAM-only/two-pass ERM.
These timings exclude deployment tests, fold preparation, checkpoint
replay, aggregation and hashing and are not publication-grade
hardware-independent speed benchmarks.

## Decision and CCF-B Boundary

Retain this fixed recipe as a promising four-class development candidate,
not a final paper model. Quadbot full mean is above both removals and all
controls, and exceeds ordinary MLP by 1.04 points. The full primary
scores are more stable in these three runs than the preceding cosine
candidate. Both seed/partition axes also favor full over ERM in 3/3.
This supports further confirmation of the four-class stability hypothesis.

The stronger two-pass ERM narrows the mean gain to 0.66 points.
It is positive in 3/3 pairs but does not meet a universal one-point
gain criterion. Contrast improves the Quadbot mean by 0.94 points,
yet only one of three paired runs favors retaining it. Flatness
improves the mean by 1.49 points and is positive in 2/3.
Both module effects are not uniformly established across training runs.

Twibot22 full mean improves over ordinary ERM by 1.36 points, but
SAM removal is better by 1.53 points in all three paired main runs,
and two-pass ERM is better by 0.71. Contrast removal is better by
0.21 in mean. Do not claim that both mechanisms improve both views.
Four-class main-text ablations do not authorize hiding transfer failures.

Do not search SAM radius, contrast weight, favorable seed/fold pairs
or selected epochs on these exposed scores. Recent bot-specific fair
controls, genuinely independent confirmation and a substantive
task-specific method contribution are still missing. No SOTA,
CCF-B acceptance or formal submission is established by this round.
Keep Word, GitHub main and the released tag unchanged. Only the
revision branch receives code and results; final alignment is
independently checked and recorded separately.
