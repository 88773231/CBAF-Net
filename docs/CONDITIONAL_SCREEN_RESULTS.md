# Conditional Routing and Content Bridge Results

## Fixed Development Scope

This is the outcome of `configs/conditional_screen.json`, trained at commit
`95698df2285fbee0a2af1b9274d17332e29ccaeb`. The 72 new fits and 72 checkpoint
replays finished on October 13, 2026, at 20:33:48 Asia/Shanghai, about
4 minutes 16 seconds after launch. All 54 matching control fits were
verified and reused. All 111 local and 111 remote regression tests passed.

The 10 collected evidence files matched the completion receipt's SHA-256
hashes. The 372 previous factor-study files, 34 released result files,
original dataset master and working Word manuscript were preserved.
No original validation/test tensors were used as model inputs.

These are post-hoc original-training-only policy challenges. They are not
independent confirmation or submission-ready results. Each seed also
changes the deterministic inner fitting/selection profile partition.
The two views share profiles and are not independent datasets.
Macro-F1 is in percent; SD is sample SD across three seeds.

## All Conditions and Seeds

| View | Excluded Policy | Method | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---:|---:|---:|---:|---:|
| Twibot22 | burst | conditional_bridge | 56.69 | 81.98 | 55.11 | 64.59 | 15.08 |
| Twibot22 | burst | without_conditional_routing | 57.77 | 78.79 | 55.97 | 64.18 | 12.69 |
| Twibot22 | burst | without_content_bridge | 68.10 | 83.14 | 59.85 | 70.36 | 11.81 |
| Twibot22 | burst | without_both | 55.25 | 55.70 | 72.99 | 61.32 | 10.12 |
| Twibot22 | burst | erm_mlp | 75.86 | 71.25 | 90.69 | 79.27 | 10.16 |
| Twibot22 | burst | hgb_statistics | 55.45 | 55.45 | 55.45 | 55.45 | 0.00 |
| Twibot22 | burst | botdmm_timing | 55.15 | 55.23 | 55.23 | 55.20 | 0.05 |
| Twibot22 | cyclic | conditional_bridge | 97.07 | 98.40 | 98.93 | 98.14 | 0.96 |
| Twibot22 | cyclic | without_conditional_routing | 99.11 | 99.47 | 99.56 | 99.38 | 0.24 |
| Twibot22 | cyclic | without_content_bridge | 98.04 | 98.40 | 99.11 | 98.52 | 0.54 |
| Twibot22 | cyclic | without_both | 99.47 | 99.56 | 98.75 | 99.26 | 0.44 |
| Twibot22 | cyclic | erm_mlp | 97.79 | 98.31 | 98.13 | 98.07 | 0.27 |
| Twibot22 | cyclic | hgb_statistics | 99.82 | 99.82 | 99.82 | 99.82 | 0.00 |
| Twibot22 | cyclic | botdmm_timing | 99.82 | 99.47 | 99.82 | 99.70 | 0.21 |
| Twibot22 | uniform | conditional_bridge | 98.74 | 98.65 | 98.92 | 98.77 | 0.14 |
| Twibot22 | uniform | without_conditional_routing | 99.37 | 99.37 | 99.19 | 99.31 | 0.10 |
| Twibot22 | uniform | without_content_bridge | 97.93 | 98.74 | 98.11 | 98.26 | 0.43 |
| Twibot22 | uniform | without_both | 99.64 | 99.64 | 99.55 | 99.61 | 0.05 |
| Twibot22 | uniform | erm_mlp | 95.40 | 96.58 | 96.30 | 96.09 | 0.62 |
| Twibot22 | uniform | hgb_statistics | 99.55 | 99.55 | 99.55 | 99.55 | 0.00 |
| Twibot22 | uniform | botdmm_timing | 99.37 | 99.46 | 99.37 | 99.40 | 0.05 |
| Quadbot | burst | conditional_bridge | 65.97 | 74.98 | 75.98 | 72.31 | 5.52 |
| Quadbot | burst | without_conditional_routing | 65.76 | 74.10 | 69.22 | 69.69 | 4.19 |
| Quadbot | burst | without_content_bridge | 90.75 | 73.80 | 83.85 | 82.80 | 8.52 |
| Quadbot | burst | without_both | 65.85 | 67.05 | 68.09 | 67.00 | 1.12 |
| Quadbot | burst | erm_mlp | 90.78 | 95.40 | 85.30 | 90.49 | 5.05 |
| Quadbot | burst | hgb_statistics | 65.78 | 65.74 | 66.00 | 65.84 | 0.14 |
| Quadbot | burst | botdmm_timing | 65.35 | 65.97 | 65.71 | 65.68 | 0.31 |
| Quadbot | cyclic | conditional_bridge | 98.40 | 98.66 | 98.86 | 98.64 | 0.23 |
| Quadbot | cyclic | without_conditional_routing | 98.39 | 98.86 | 98.46 | 98.57 | 0.25 |
| Quadbot | cyclic | without_content_bridge | 97.87 | 98.12 | 97.86 | 97.95 | 0.15 |
| Quadbot | cyclic | without_both | 98.73 | 98.93 | 98.66 | 98.78 | 0.14 |
| Quadbot | cyclic | erm_mlp | 96.99 | 97.85 | 96.57 | 97.14 | 0.65 |
| Quadbot | cyclic | hgb_statistics | 98.73 | 98.73 | 98.60 | 98.69 | 0.08 |
| Quadbot | cyclic | botdmm_timing | 98.80 | 99.20 | 98.93 | 98.98 | 0.20 |
| Quadbot | uniform | conditional_bridge | 98.31 | 98.24 | 97.55 | 98.03 | 0.42 |
| Quadbot | uniform | without_conditional_routing | 98.24 | 98.92 | 98.51 | 98.55 | 0.34 |
| Quadbot | uniform | without_content_bridge | 97.82 | 97.57 | 98.24 | 97.88 | 0.34 |
| Quadbot | uniform | without_both | 99.12 | 98.92 | 98.71 | 98.92 | 0.20 |
| Quadbot | uniform | erm_mlp | 97.00 | 97.42 | 96.42 | 96.94 | 0.50 |
| Quadbot | uniform | hgb_statistics | 98.30 | 98.03 | 98.71 | 98.35 | 0.34 |
| Quadbot | uniform | botdmm_timing | 98.85 | 98.72 | 98.99 | 98.85 | 0.13 |

Short policy names denote `burst_scheduler`, `cyclic_reposter` and
`uniform_scheduler`. The archived JSON uses the full names.

## Paired Full Minus Comparator

Differences are percentage points. Positive seed counts are descriptive,
not significance tests. Calculations use unrounded scores.

| View | Policy | Comparator | Mean Delta | Delta SD | Positive Seeds |
|---|---|---|---:|---:|---:|
| Twibot22 | burst | without_conditional_routing | +0.42 | 2.41 | 1/3 |
| Twibot22 | burst | without_content_bridge | -5.77 | 5.20 | 0/3 |
| Twibot22 | burst | without_both | +3.28 | 22.14 | 2/3 |
| Twibot22 | burst | erm_mlp | -14.67 | 23.48 | 1/3 |
| Twibot22 | burst | hgb_statistics | +9.15 | 15.08 | 2/3 |
| Twibot22 | burst | botdmm_timing | +9.39 | 15.06 | 2/3 |
| Twibot22 | cyclic | without_conditional_routing | -1.24 | 0.73 | 0/3 |
| Twibot22 | cyclic | without_content_bridge | -0.38 | 0.52 | 0/3 |
| Twibot22 | cyclic | without_both | -1.12 | 1.29 | 1/3 |
| Twibot22 | cyclic | erm_mlp | +0.06 | 0.76 | 2/3 |
| Twibot22 | cyclic | hgb_statistics | -1.69 | 0.96 | 0/3 |
| Twibot22 | cyclic | botdmm_timing | -1.57 | 1.03 | 0/3 |
| Twibot22 | uniform | without_conditional_routing | -0.54 | 0.24 | 0/3 |
| Twibot22 | uniform | without_content_bridge | +0.51 | 0.52 | 2/3 |
| Twibot22 | uniform | without_both | -0.84 | 0.19 | 0/3 |
| Twibot22 | uniform | erm_mlp | +2.68 | 0.64 | 3/3 |
| Twibot22 | uniform | hgb_statistics | -0.78 | 0.14 | 0/3 |
| Twibot22 | uniform | botdmm_timing | -0.63 | 0.18 | 0/3 |
| Quadbot | burst | without_conditional_routing | +2.62 | 3.60 | 3/3 |
| Quadbot | burst | without_content_bridge | -10.49 | 13.18 | 1/3 |
| Quadbot | burst | without_both | +5.31 | 4.50 | 3/3 |
| Quadbot | burst | erm_mlp | -18.18 | 7.98 | 0/3 |
| Quadbot | burst | hgb_statistics | +6.47 | 5.45 | 3/3 |
| Quadbot | burst | botdmm_timing | +6.63 | 5.25 | 3/3 |
| Quadbot | cyclic | without_conditional_routing | +0.07 | 0.31 | 2/3 |
| Quadbot | cyclic | without_content_bridge | +0.69 | 0.27 | 3/3 |
| Quadbot | cyclic | without_both | -0.13 | 0.29 | 1/3 |
| Quadbot | cyclic | erm_mlp | +1.50 | 0.74 | 3/3 |
| Quadbot | cyclic | hgb_statistics | -0.05 | 0.30 | 1/3 |
| Quadbot | cyclic | botdmm_timing | -0.33 | 0.24 | 0/3 |
| Quadbot | uniform | without_conditional_routing | -0.52 | 0.54 | 1/3 |
| Quadbot | uniform | without_content_bridge | +0.15 | 0.74 | 2/3 |
| Quadbot | uniform | without_both | -0.89 | 0.25 | 0/3 |
| Quadbot | uniform | erm_mlp | +1.09 | 0.25 | 3/3 |
| Quadbot | uniform | hgb_statistics | -0.32 | 0.74 | 2/3 |
| Quadbot | uniform | botdmm_timing | -0.82 | 0.54 | 0/3 |

## Capacity

| Condition | Twibot22 Parameters | Quadbot Parameters |
|---|---:|---:|
| conditional_bridge | 1,307,780 | 1,307,909 |
| without_conditional_routing | 1,308,551 | 1,308,937 |
| without_content_bridge | 1,291,396 | 1,291,525 |
| without_both | 1,292,167 | 1,292,553 |
| erm_mlp | 1,089,667 | 1,089,796 |
| botdmm_timing | 1,423,111 | 1,423,304 |

HGB has no comparable neural parameter count. Bridge removal physically
omits its projection. Routing removal retains auxiliary heads but replaces
conditional leaves with a flat head. These are not capacity-matched
comparisons. HGB receives global summaries; adapted BotDMM also receives
unlabelled feature-kNN context and is not a faithful original reproduction.

## Diagnosed Boundary

For the full model in the excluded-burst folds:

| View | Seed | Best Epoch | Known-Policy Selection Macro-F1 | Burst Traditional Bot F1 |
|---|---:|---:|---:|---:|
| Twibot22 | 42 | 3 | 98.92 | 5.21 |
| Twibot22 | 43 | 3 | 99.73 | 67.36 |
| Twibot22 | 44 | 9 | 99.46 | 1.06 |
| Quadbot | 42 | 12 | 98.58 | 1.07 |
| Quadbot | 43 | 4 | 99.59 | 31.84 |
| Quadbot | 44 | 6 | 98.78 | 36.40 |

Full-model burst LLM Bot F1 is 97.89/98.79/98.14% in Twibot22 and
98.65/98.66/98.66% in Quadbot. Agent F1 in Quadbot is
98.14/99.07/97.98%. The largest observed weakness is the
Human/Traditional boundary, not failure to recognize every synthetic class.

High known-policy selection scores coexist with weak excluded-policy
performance. This is consistent with policy-sensitive generalization,
but does not identify which input features cause it. The content/behavior
groups overlap because event hashes contain text. A content bridge is not
an independent causal intervention on textual information.

## Decision and Next Work

The full candidate is rejected as a supported paper replacement.
The bridge hurts burst means in both views; removing it does not establish
a final successful method. The no-bridge burst means are still below MLP
by 8.90 points in Twibot22 and 7.69 points in Quadbot.

Conditional routing is not consistently beneficial: Twibot22 cyclic and
uniform means both favor its removal, as does Quadbot uniform. The full
model trails adapted BotDMM on cyclic and uniform means in both views.
Equal-weight policy means also trail MLP: 87.17% versus 91.14% for
Twibot22 and 89.66% versus 94.86% for Quadbot. These summaries do not
replace the complete per-policy tables above.

Do not search bridge weights, drop seeds, rename a removal as a successful
full model, or update manuscript claims from these development scores.
Keep datasets and labels unchanged. Before another training round, use
existing checkpoints for read-only, fitting-reference input-sensitivity
diagnostics on the affected boundary. Feature masking is a diagnostic
distribution change, not proof of causality or a new valid evaluation.
Do not deploy a masked-input variant based on challenge scores.

A subsequent training intervention must target the diagnosed dependency,
define its controls before fitting and leave all exposed challenges
explicitly marked as development. Recent fair controls and independent
evaluation remain open gaps. A universal one-point improvement is not
feasible on a condition already at 99.82%; at most 0.18 points remain.
The required claims and success criteria must respect that ceiling.

Original datasets, published results, Word manuscript and GitHub main/tag
remain unchanged. Revision-branch synchronization is verified separately;
this document alone is not evidence of a successful push or submission
readiness.
