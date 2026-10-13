# Cosine Readout and Policy Contrast Results

## Scope and Verification

The fixed recipe in `configs/cosine_policy_screen.json` was trained at
commit `9521269a303e6acff205e81ef9d51009354e73b6`. All 36 new fits and
36 selected-checkpoint replays completed on October 13, 2026 at
21:58:49 Asia/Shanghai. All 72 matching cached fits were verified.
The complete local and remote regression suites both passed 132 tests.
All 10 downloaded evidence files matched the remote receipt hashes.

Quadbot was declared the primary four-class task before fitting;
Twibot22 was declared the three-class transfer task. Report both views,
all conditions and all seeds. The primary score pools the three excluded
policy prediction sets once per profile/class within each seed. Seeds
42/43/44 also change the deterministic inner profile selection folds.
SD below is sample SD across those seed/fold pairs.

The candidate preserves the matched MLP encoder and seeded classifier
weights, replaces the ordinary linear head with a fixed-scale normalized
head, and retains cross-policy class contrast. It does not use the
previous pair-margin loss. The scale 16, contrast weight 0.1 and
temperature 0.2 were frozen before fitting and were not searched.
Prediction needs no policy/profile identifiers or paired accounts.

These are post-hoc original-training-only development outcomes. The
challenges have already been exposed by preceding development studies.
They are not independent validation or submission-ready test results.
The two views share profiles, not independent real-world collections.
Original validation/test tensors were not opened for model evaluation.
The dataset master, released results, prior evidence and working Word
manuscript were preserved.

## Primary Overall Macro-F1

Scores are percentages, computed from verified predictions rather than
averages of rounded per-policy scores.

| View | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---:|---:|---:|---:|---:|
| Twibot22 | cosine_policy | 92.89 | 97.43 | 90.38 | 93.57 | 3.57 |
| Twibot22 | without_policy_contrast | 90.39 | 97.31 | 94.69 | 94.13 | 3.49 |
| Twibot22 | without_cosine_head | 94.00 | 93.99 | 95.88 | 94.62 | 1.09 |
| Twibot22 | erm_mlp | 90.32 | 89.78 | 95.08 | 91.73 | 2.92 |
| Twibot22 | hgb_statistics | 88.33 | 88.33 | 88.33 | 88.33 | 0.00 |
| Twibot22 | botdmm_timing | 88.15 | 88.11 | 88.20 | 88.15 | 0.04 |
| Quadbot | cosine_policy | 94.66 | 93.77 | 97.33 | 95.25 | 1.85 |
| Quadbot | without_policy_contrast | 93.88 | 96.59 | 94.51 | 94.99 | 1.42 |
| Quadbot | without_cosine_head | 94.71 | 96.26 | 92.58 | 94.52 | 1.85 |
| Quadbot | erm_mlp | 94.98 | 96.90 | 93.01 | 94.96 | 1.95 |
| Quadbot | hgb_statistics | 90.18 | 90.10 | 90.34 | 90.21 | 0.12 |
| Quadbot | botdmm_timing | 90.27 | 90.54 | 90.43 | 90.41 | 0.14 |

The ordinary-head removal is the preceding paired-policy study's
`without_pair_margin`: matched MLP plus contrast and no pair margin.
Its original condition name, checkpoint and prediction provenance
are preserved. Reusing this mathematically equivalent removal is not
renaming the rejected paired-margin model as a successful final method.
ERM removes both proposed mechanisms.

## Paired Overall Differences

Full minus comparator, in percentage points, using unrounded scores.
Positive seed counts are descriptive, not significance tests.

| View | Comparator | Mean Delta | Delta SD | Positive Seeds |
|---|---|---:|---:|---:|
| Twibot22 | without_policy_contrast | -0.56 | 3.45 | 2/3 |
| Twibot22 | without_cosine_head | -1.06 | 4.47 | 1/3 |
| Twibot22 | erm_mlp | +1.84 | 6.20 | 2/3 |
| Twibot22 | hgb_statistics | +5.24 | 3.57 | 3/3 |
| Twibot22 | botdmm_timing | +5.41 | 3.61 | 3/3 |
| Quadbot | without_policy_contrast | +0.26 | 2.85 | 2/3 |
| Quadbot | without_cosine_head | +0.74 | 3.68 | 1/3 |
| Quadbot | erm_mlp | +0.29 | 3.76 | 1/3 |
| Quadbot | hgb_statistics | +5.05 | 1.73 | 3/3 |
| Quadbot | botdmm_timing | +4.84 | 1.87 | 3/3 |

## All Per-Policy Conditions

Short names denote `burst_scheduler`, `cyclic_reposter` and
`uniform_scheduler`. Macro-F1 is percent.

| View | Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | Seed SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | burst | cosine_policy | 81.29 | 95.96 | 71.05 | 82.77 | 12.52 |
| Twibot22 | burst | without_policy_contrast | 73.32 | 95.41 | 88.15 | 85.63 | 11.26 |
| Twibot22 | burst | without_cosine_head | 88.90 | 87.10 | 93.79 | 89.93 | 3.46 |
| Twibot22 | burst | erm_mlp | 75.86 | 71.25 | 90.69 | 79.27 | 10.16 |
| Twibot22 | burst | hgb_statistics | 55.45 | 55.45 | 55.45 | 55.45 | 0.00 |
| Twibot22 | burst | botdmm_timing | 55.15 | 55.23 | 55.23 | 55.20 | 0.05 |
| Twibot22 | cyclic | cosine_policy | 98.13 | 99.20 | 99.29 | 98.87 | 0.65 |
| Twibot22 | cyclic | without_policy_contrast | 98.22 | 98.49 | 98.84 | 98.52 | 0.31 |
| Twibot22 | cyclic | without_cosine_head | 97.52 | 98.93 | 98.21 | 98.22 | 0.71 |
| Twibot22 | cyclic | erm_mlp | 97.79 | 98.31 | 98.13 | 98.07 | 0.27 |
| Twibot22 | cyclic | hgb_statistics | 99.82 | 99.82 | 99.82 | 99.82 | 0.00 |
| Twibot22 | cyclic | botdmm_timing | 99.82 | 99.47 | 99.82 | 99.70 | 0.21 |
| Twibot22 | uniform | cosine_policy | 98.29 | 97.12 | 97.38 | 97.59 | 0.61 |
| Twibot22 | uniform | without_policy_contrast | 97.02 | 98.01 | 96.84 | 97.29 | 0.63 |
| Twibot22 | uniform | without_cosine_head | 95.40 | 95.67 | 95.57 | 95.54 | 0.14 |
| Twibot22 | uniform | erm_mlp | 95.40 | 96.58 | 96.30 | 96.09 | 0.62 |
| Twibot22 | uniform | hgb_statistics | 99.55 | 99.55 | 99.55 | 99.55 | 0.00 |
| Twibot22 | uniform | botdmm_timing | 99.37 | 99.46 | 99.37 | 99.40 | 0.05 |
| Quadbot | burst | cosine_policy | 88.57 | 85.99 | 96.21 | 90.26 | 5.32 |
| Quadbot | burst | without_policy_contrast | 87.01 | 93.46 | 88.72 | 89.73 | 3.34 |
| Quadbot | burst | without_cosine_head | 90.03 | 94.51 | 83.97 | 89.50 | 5.29 |
| Quadbot | burst | erm_mlp | 90.78 | 95.40 | 85.30 | 90.49 | 5.05 |
| Quadbot | burst | hgb_statistics | 65.78 | 65.74 | 66.00 | 65.84 | 0.14 |
| Quadbot | burst | botdmm_timing | 65.35 | 65.97 | 65.71 | 65.68 | 0.31 |
| Quadbot | cyclic | cosine_policy | 98.33 | 97.26 | 98.00 | 97.86 | 0.54 |
| Quadbot | cyclic | without_policy_contrast | 97.33 | 97.99 | 97.38 | 97.57 | 0.37 |
| Quadbot | cyclic | without_cosine_head | 96.22 | 97.46 | 96.16 | 96.62 | 0.74 |
| Quadbot | cyclic | erm_mlp | 96.99 | 97.85 | 96.57 | 97.14 | 0.65 |
| Quadbot | cyclic | hgb_statistics | 98.73 | 98.73 | 98.60 | 98.69 | 0.08 |
| Quadbot | cyclic | botdmm_timing | 98.80 | 99.20 | 98.93 | 98.98 | 0.20 |
| Quadbot | uniform | cosine_policy | 96.72 | 97.40 | 97.75 | 97.29 | 0.53 |
| Quadbot | uniform | without_policy_contrast | 96.87 | 98.24 | 96.97 | 97.36 | 0.76 |
| Quadbot | uniform | without_cosine_head | 97.62 | 96.79 | 96.62 | 97.01 | 0.53 |
| Quadbot | uniform | erm_mlp | 97.00 | 97.42 | 96.42 | 96.94 | 0.50 |
| Quadbot | uniform | hgb_statistics | 98.30 | 98.03 | 98.71 | 98.35 | 0.34 |
| Quadbot | uniform | botdmm_timing | 98.85 | 98.72 | 98.99 | 98.85 | 0.13 |

Full class-specific metrics, policy records, histories, per-seed pooled
scores and verified prediction hashes remain in the archived summary
JSON, SHA-256:
`ce2e3fc0ed244aab13d6850f6dc804f0896db66338063d2131b2b4a655b163dc`.

## Selection and Stability Diagnostic

Read-only inspection of the saved histories gives the following full
condition records on the excluded-burst folds. No checkpoint was
reselected using these challenge results.

| View | Seed | Best Epoch | Known Selection Macro-F1 | Worst Known Policy Macro-F1 | Burst Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Twibot22 | 42 | 6 | 98.64 | 98.38 | 81.29 |
| Twibot22 | 43 | 4 | 99.73 | 99.46 | 95.96 |
| Twibot22 | 44 | 6 | 99.19 | 98.36 | 71.05 |
| Quadbot | 42 | 2 | 98.78 | 98.76 | 88.57 |
| Quadbot | 43 | 2 | 98.99 | 98.40 | 85.99 |
| Quadbot | 44 | 5 | 97.53 | 97.51 | 96.21 |

High known-policy scores coexist with variable excluded-policy scores.
The seed changes initialization, minibatch order and inner profile
partition, so these records do not isolate an initialization effect.
Nor do they prove that early stopping causes the generalization gap.
Using exposed challenge scores to select a different epoch or scale
would not resolve the evidence problem.

## Decision

The frozen four-class mean gate is met: the full condition exceeds all
three controls and both individual removals in mean pooled Macro-F1.
However, the desired one-point gain over the stronger ERM control is
not met: the observed gain is 0.2905 points, with paired SD 3.7593.
Only one of three paired seed/fold runs favors the full model over ERM
or over cosine-head removal. This is weak evidence of a stable gain,
not a supported submission replacement.

The four-class ablation means favor retaining contrast by 0.2607 points
and cosine readout by 0.7367 points, but their paired SDs are 2.8486 and
3.6807 points. Do not claim statistically established module benefits.
The three-class transfer means favor removing contrast by 0.5648 points
and removing cosine readout by 1.0557 points. Restricting the main text
to four-class ablations must not hide this transfer limitation.

The head has 1,089,664 parameters in Twibot22 and 1,089,792 in Quadbot,
respectively 3 and 4 fewer than the ordinary MLP because class biases
are omitted. The two new conditions have the same parameter count.
HGB uses global summaries; adapted BotDMM additionally uses unlabelled
feature-kNN context and is not a faithful reproduction of its original
published setting. Exceeding that adapted control is not SOTA evidence.
The cosine classifier and supervised contrast are established techniques;
combining them does not establish a new general algorithm.

## Next Work and Alignment Boundary

Do not tune cosine scale or contrast weight on the already exposed
challenges. Keep this recipe and all previous studies immutable.
Another model hypothesis needs a specific training/representation
diagnostic and predefined controls. Prediction stability and the
known-policy/excluded-policy gap remain unresolved. A new evaluation
must also distinguish seed effects from partition effects where feasible.

Fair recent baselines and independent confirmation remain necessary
before a replacement paper can be claimed. The dataset is not rebuilt
or enlarged in response to this outcome. Word, GitHub main and the
released tag remain unchanged. Only the development revision branch
receives the new code and outcome record; synchronization is separately
verified. This study does not authorize another formal submission.
