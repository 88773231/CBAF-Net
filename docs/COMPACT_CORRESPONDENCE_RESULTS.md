# Compact Correspondence Results

## Verified Execution

The fixed matrix completed on October 14, 2026 at 23:52:16 Asia/Shanghai,
34 minutes 49 seconds after deployment began. It includes 30 new fits,
six exact-partition cached controls and 36 checkpoint replays, CPU only.
All 188 local regression tests and 11 targeted remote tests passed.
The subsequent local receipt audit recomputed all 36 prediction metrics
and verified checkpoint envelopes, fit/reuse identities and shared
initialization. No original data, manuscript or canonical/public code changed.
An authenticated remote check on October 15 at 00:02:43 found no active
analysis jobs.

These are known-policy inner-selection scores, not test/challenge scores.
All three training seeds use the same seed-42 fitting/selection partitions.
Only 185 distinct selection profiles occur across the three overlapping
folds; each profile occurs in two folds. Seeds and folds are not independent
real-world evaluations. No original validation/test or excluded-policy
forward/scoring was performed.

## All Planned Selections

Macro-F1 in percent; "excluded" identifies the fold, not the scored policy.
The scored rows belong only to the other two known policies.

| Training Seed | Excluded Policy | Compact Correspondence | Without Correspondence | Supervised Encoder | BotSSCL Mechanism |
|---|---|---:|---:|---:|---:|
| 42 | Burst | 98.78 | 98.16 | 97.76 | 98.16 |
| 42 | Cyclic | 99.18 | 98.57 | 98.78 | 99.18 |
| 42 | Uniform | 98.59 | 98.78 | 98.39 | 97.78 |
| 43 | Burst | 98.78 | 98.78 | 97.95 | 97.32 |
| 43 | Cyclic | 99.18 | 98.37 | 98.37 | 98.37 |
| 43 | Uniform | 98.78 | 98.99 | 97.76 | 98.17 |
| 44 | Burst | 98.58 | 98.15 | 98.16 | 97.57 |
| 44 | Cyclic | 97.74 | 98.78 | 98.57 | 98.37 |
| 44 | Uniform | 98.17 | 98.78 | 97.76 | 98.17 |

Rounded ties do not necessarily imply identical predictions or exact scores.
Use archived unrounded metrics for every decision.

| Condition | Seed 42 Fold Mean | Seed 43 Fold Mean | Seed 44 Fold Mean | Overall Mean | Seed SD |
|---|---:|---:|---:|---:|---:|
| Compact correspondence | 98.85 | 98.92 | 98.16 | 98.64 | 0.42 |
| Without correspondence | 98.50 | 98.71 | 98.57 | 98.60 | 0.11 |
| Supervised encoder | 98.31 | 98.03 | 98.16 | 98.17 | 0.14 |
| BotSSCL mechanism | 98.37 | 97.95 | 98.04 | 98.12 | 0.22 |

## Paired Interpretation

Compact minus removal has seed-mean differences of +0.345, +0.204 and
-0.408 percentage points, a +0.047-point mean. It is positive in two
of three seed means and five of nine fold/seed cells. The candidate's
seed variability is substantially larger than its average module gain.
Do not describe the correspondence module as consistently beneficial
or as a demonstrated one-point improvement.

Compact minus supervised encoder averages +0.476 points, with positive
seed means in two of three seeds. Compact minus BotSSCL mechanism
averages +0.521 points, positive in all three seed means.
The removal itself also exceeds both controls on average; these
comparisons do not isolate the correspondence module as the explanation
for the whole-model improvement.

The frozen development-continuation gate passes, but it is a weak
directional screening gate, not significance, final generalization,
novelty or submission confirmation. The old transition remains discarded.
Under the latest user instruction, stop extending architecture searches
and proceed to the bounded scope/evidence/manuscript work order in
`SUBMISSION_SPRINT.md`. Do not automatically promote a removal or
rename an established control into an original method.

## Capacity and Compute

The compact correspondence model has 637,966 inference/trainable
parameters; its removal has 613,380. Neither serializes zero-input
transition/discrepancy columns. The control adapter has 1,122,564
training parameters, including its training-only projection, and
1,106,052 inference parameters. It uses a different architecture and
no dropout. BotSSCL also has 30 additional pretraining epochs.
Input/supervised-budget matching is not exact capacity/compute matching.
The BotSSCL condition is a mechanism adaptation, not a faithful reproduction
of its original benchmark.

## Receipts and Remaining Boundaries

- Summary SHA-256:
  `969c9b0a2a8a29075b201e481676de229274aaba5c894c301ee1eff80f6b1db8`.
- Evidence archive SHA-256:
  `e44b55ea787f52942de107c5d0c5fd9be823b0c7f481e098a10e2344b640dfb4`.
- Source archive SHA-256:
  `779184e4d493fee7f1f089c8c014318b036341cb1439d096f0850e489067d5d4`.

The single working manuscript remains the older released fusion method,
with unchanged SHA-256
`af7f16a306aa06711de9bed422f00455c30fd67899cf347f4806aab665c3f685`.
Canonical remote/public revision remains
`a7605e2af6f2990ce80b15df8305aabb7b3dcaad`.
No private development code was pushed as a final public replacement.
Recent compatible comparisons, evaluation exposure history, generator
archive/source permissions and a defensible method contribution remain
to be resolved before revising claims and submitting.
