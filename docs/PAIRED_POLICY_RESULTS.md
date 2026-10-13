# Paired Policy Objective Results

## Scope and Verification

Training used commit `a6d05ad2453d82cfce06b2eb86516d2b81ad1e1b`.
The 54 new fits and 54 checkpoint replays finished on October 13, 2026.
All 54 matching control fits were verified/reused. Local and remote
regression suites both passed 122 tests. The 10 downloaded evidence
files matched their remote completion receipt hashes.

The primary target was fixed before training: overall three-class and
four-class improvement, not an improvement in every individual class.
The primary metric pools the three excluded-policy prediction sets once
per profile/class within each seed, then reports mean and sample SD.
Seeds 42/43/44 also change the deterministic inner selection partition.

These are post-hoc original-training-only development results, not new
independent submission evidence. Every planned condition/seed is retained.
Policy and profile identifiers are fitting-objective supervision only,
not model inputs at prediction. No validation/test tensors, new accounts,
relabeling, split manipulation or augmentation were used in this screen.
The two views share profiles and are not independent real-world datasets.

## Primary Overall Macro-F1

Scores are percentages. All methods use the exact verified prediction
sets for each seed; this is not an average of rounded per-policy scores.

| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---:|---:|---:|---:|---:|
| Twibot22 | paired_policy | 90.28 | 96.65 | 91.27 | 92.73 | 3.43 |
| Twibot22 | without_policy_contrast | 89.68 | 96.08 | 93.24 | 93.00 | 3.21 |
| Twibot22 | without_pair_margin | 94.00 | 93.99 | 95.88 | 94.62 | 1.09 |
| Twibot22 | erm_mlp | 90.32 | 89.78 | 95.08 | 91.73 | 2.92 |
| Twibot22 | hgb_statistics | 88.33 | 88.33 | 88.33 | 88.33 | 0.00 |
| Twibot22 | botdmm_timing | 88.15 | 88.11 | 88.20 | 88.15 | 0.04 |
| Quadbot | paired_policy | 95.63 | 96.24 | 92.25 | 94.70 | 2.15 |
| Quadbot | without_policy_contrast | 91.64 | 94.39 | 91.85 | 92.62 | 1.53 |
| Quadbot | without_pair_margin | 94.71 | 96.26 | 92.58 | 94.52 | 1.85 |
| Quadbot | erm_mlp | 94.98 | 96.90 | 93.01 | 94.96 | 1.95 |
| Quadbot | hgb_statistics | 90.18 | 90.10 | 90.34 | 90.21 | 0.12 |
| Quadbot | botdmm_timing | 90.27 | 90.54 | 90.43 | 90.41 | 0.14 |

## Paired Overall Differences

Full minus comparator, in percentage points, using unrounded scores.
Positive seed counts are descriptive, not significance tests.

| View | Comparator | Mean Delta | Delta SD | Positive Seeds |
|---|---|---:|---:|---:|
| Twibot22 | without_policy_contrast | -0.27 | 1.47 | 2/3 |
| Twibot22 | without_pair_margin | -1.89 | 3.96 | 1/3 |
| Twibot22 | erm_mlp | +1.01 | 5.42 | 1/3 |
| Twibot22 | hgb_statistics | +4.41 | 3.43 | 3/3 |
| Twibot22 | botdmm_timing | +4.58 | 3.46 | 3/3 |
| Quadbot | without_policy_contrast | +2.08 | 1.80 | 3/3 |
| Quadbot | without_pair_margin | +0.19 | 0.65 | 1/3 |
| Quadbot | erm_mlp | -0.26 | 0.78 | 1/3 |
| Quadbot | hgb_statistics | +4.50 | 2.27 | 3/3 |
| Quadbot | botdmm_timing | +4.29 | 2.15 | 3/3 |

## All Per-Policy Conditions and Seeds

| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | burst | paired_policy | 75.90 | 95.88 | 80.57 | 84.11 | 10.46 |
| Twibot22 | burst | without_policy_contrast | 74.33 | 93.95 | 86.72 | 85.00 | 9.92 |
| Twibot22 | burst | without_pair_margin | 88.90 | 87.10 | 93.79 | 89.93 | 3.46 |
| Twibot22 | burst | erm_mlp | 75.86 | 71.25 | 90.69 | 79.27 | 10.16 |
| Twibot22 | burst | hgb_statistics | 55.45 | 55.45 | 55.45 | 55.45 | 0.00 |
| Twibot22 | burst | botdmm_timing | 55.15 | 55.23 | 55.23 | 55.20 | 0.05 |
| Twibot22 | cyclic | paired_policy | 98.22 | 98.76 | 97.68 | 98.22 | 0.54 |
| Twibot22 | cyclic | without_policy_contrast | 97.78 | 98.67 | 96.58 | 97.68 | 1.05 |
| Twibot22 | cyclic | without_pair_margin | 97.52 | 98.93 | 98.21 | 98.22 | 0.71 |
| Twibot22 | cyclic | erm_mlp | 97.79 | 98.31 | 98.13 | 98.07 | 0.27 |
| Twibot22 | cyclic | hgb_statistics | 99.82 | 99.82 | 99.82 | 99.82 | 0.00 |
| Twibot22 | cyclic | botdmm_timing | 99.82 | 99.47 | 99.82 | 99.70 | 0.21 |
| Twibot22 | uniform | paired_policy | 94.76 | 95.27 | 94.45 | 94.83 | 0.42 |
| Twibot22 | uniform | without_policy_contrast | 94.66 | 95.55 | 96.02 | 95.41 | 0.69 |
| Twibot22 | uniform | without_pair_margin | 95.40 | 95.67 | 95.57 | 95.54 | 0.14 |
| Twibot22 | uniform | erm_mlp | 95.40 | 96.58 | 96.30 | 96.09 | 0.62 |
| Twibot22 | uniform | hgb_statistics | 99.55 | 99.55 | 99.55 | 99.55 | 0.00 |
| Twibot22 | uniform | botdmm_timing | 99.37 | 99.46 | 99.37 | 99.40 | 0.05 |
| Quadbot | burst | paired_policy | 93.40 | 93.88 | 82.97 | 90.08 | 6.17 |
| Quadbot | burst | without_policy_contrast | 79.60 | 88.09 | 81.94 | 83.21 | 4.38 |
| Quadbot | burst | without_pair_margin | 90.03 | 94.51 | 83.97 | 89.50 | 5.29 |
| Quadbot | burst | erm_mlp | 90.78 | 95.40 | 85.30 | 90.49 | 5.05 |
| Quadbot | burst | hgb_statistics | 65.78 | 65.74 | 66.00 | 65.84 | 0.14 |
| Quadbot | burst | botdmm_timing | 65.35 | 65.97 | 65.71 | 65.68 | 0.31 |
| Quadbot | cyclic | paired_policy | 96.57 | 97.73 | 95.81 | 96.70 | 0.97 |
| Quadbot | cyclic | without_policy_contrast | 96.71 | 97.66 | 95.89 | 96.75 | 0.89 |
| Quadbot | cyclic | without_pair_margin | 96.22 | 97.46 | 96.16 | 96.62 | 0.74 |
| Quadbot | cyclic | erm_mlp | 96.99 | 97.85 | 96.57 | 97.14 | 0.65 |
| Quadbot | cyclic | hgb_statistics | 98.73 | 98.73 | 98.60 | 98.69 | 0.08 |
| Quadbot | cyclic | botdmm_timing | 98.80 | 99.20 | 98.93 | 98.98 | 0.20 |
| Quadbot | uniform | paired_policy | 96.87 | 97.06 | 96.76 | 96.90 | 0.15 |
| Quadbot | uniform | without_policy_contrast | 96.71 | 97.07 | 96.36 | 96.71 | 0.36 |
| Quadbot | uniform | without_pair_margin | 97.62 | 96.79 | 96.62 | 97.01 | 0.53 |
| Quadbot | uniform | erm_mlp | 97.00 | 97.42 | 96.42 | 96.94 | 0.50 |
| Quadbot | uniform | hgb_statistics | 98.30 | 98.03 | 98.71 | 98.35 | 0.34 |
| Quadbot | uniform | botdmm_timing | 98.85 | 98.72 | 98.99 | 98.85 | 0.13 |

Short policy names denote `burst_scheduler`, `cyclic_reposter` and
`uniform_scheduler`. Complete paired per-policy differences and per-class
F1 are retained in the archived development summary JSON, SHA-256:
`887cb800030b3d3d79d5c859606b8a25f3b88dee1a393cf1b8765b33f64fa3e4`.

## Decision

The full condition fails the declared overall replacement criterion.
It improves Twibot22 overall mean versus ERM by 1.01 points, but only
one of three paired seeds favors it; paired delta SD is 5.42 points.
It is 0.26 points below ERM in Quadbot. Mean gains of 4.58/4.29 points
versus adapted BotDMM do not establish SOTA while ERM is stronger in
Quadbot and a removal is stronger in Twibot22.

Twibot22 favors removing the pair margin by 1.89 points overall and
removing contrast by 0.27. Quadbot favors the full condition by 2.08
versus contrast removal but only 0.19 versus margin removal; the latter
is positive in only one of three seeds. The evidence does not support
claiming both objective components improve both views.

Contrast alone improves Twibot22 overall mean compared with ERM but
is not automatically a final architecture. Do not rename it to conceal
the rejected margin or change the claims without new justification.
A task-dependent margin effect is a development finding, not permission
to activate it using challenge labels or choose losses after scoring.

## Remaining Work and Alignment

Both proposed additions are fitting objectives, not new inference
architecture modules. Every condition has the exact matched MLP
inference parameter count: 1,089,667 for Twibot22 and 1,089,796 for
Quadbot. No extra kNN context is used. HGB uses global summaries;
adapted BotDMM has additional unlabelled feature-kNN context and is
not a faithful reproduction of the original published setting.

Reconsider the unsupported margin before another model round, rather
than tuning its coefficient on these exposed scores. A future second
mechanism needs an independent justification and predefined removals.
Do not re-build or enlarge the datasets as a reaction to this outcome.
Fair recent controls and independent evaluation remain gaps.

Dataset master, released results and working Word are unchanged.
The completion receipt found no related training jobs or GPU compute
processes. Code/evidence synchronization belongs to the revision branch;
GitHub main and the released tag are not changed by this study.
The failed gate does not authorize promoting this candidate into the
manuscript or starting another formal submission.
