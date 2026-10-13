# Timing-Adapted BotDMM Module Outcomes

All 36 fixed module-removal fits and 36 checkpoint replays completed on
October 13, 2026. Each replay reproduced saved probabilities within
absolute tolerance 1e-6. The prior 36 timing-control fits are preserved;
only their 18 full-timing fits across both views are used as controls.

These are post-hoc diagnostics on already inspected validation-policy
challenges, not untouched tests, a new final method or a full reproduction
of the original BotDMM paper. The two data views are nested.

## All Policy Outcomes

Macro-F1 values are percentages, with sample SD across seeds 42, 43 and 44.

| View | Held-out policy | Full timing backbone | Structure output masked | Orthogonal projection skipped |
|---|---|---:|---:|---:|
| Twibot22 | burst_scheduler | 55.31 +/- 0.15 | 55.31 +/- 0.08 | 55.35 +/- 0.09 |
| Twibot22 | cyclic_reposter | 100.00 +/- 0.00 | 99.78 +/- 0.38 | 99.89 +/- 0.19 |
| Twibot22 | uniform_scheduler | 99.87 +/- 0.23 | 99.87 +/- 0.23 | 99.87 +/- 0.23 |
| Quadbot | burst_scheduler | 65.56 +/- 0.43 | 65.20 +/- 0.19 | 65.44 +/- 0.43 |
| Quadbot | cyclic_reposter | 99.26 +/- 0.43 | 99.01 +/- 0.00 | 98.52 +/- 0.43 |
| Quadbot | uniform_scheduler | 99.40 +/- 0.30 | 99.20 +/- 0.63 | 99.30 +/- 0.18 |

## Paired Descriptive Differences

Signs below are removal minus full backbone, in percentage points.
Intervals resample paired profiles with the three training seeds fixed.
They are descriptive, not confirmatory confidence statements.

| View | Policy | Removal | Difference | Descriptive 95% interval |
|---|---|---|---:|---:|
| Twibot22 | burst_scheduler | structure output | -0.00 | [-0.19, +0.26] |
| Twibot22 | burst_scheduler | orthogonal projection | +0.03 | [-0.16, +0.26] |
| Twibot22 | cyclic_reposter | structure output | -0.22 | [-0.55, +0.00] |
| Twibot22 | cyclic_reposter | orthogonal projection | -0.11 | [-0.33, +0.00] |
| Twibot22 | uniform_scheduler | structure output | +0.00 | [-0.40, +0.40] |
| Twibot22 | uniform_scheduler | orthogonal projection | +0.00 | [-0.40, +0.40] |
| Quadbot | burst_scheduler | structure output | -0.36 | [-0.95, +0.24] |
| Quadbot | burst_scheduler | orthogonal projection | -0.12 | [-0.62, +0.29] |
| Quadbot | cyclic_reposter | structure output | -0.25 | [-0.83, +0.25] |
| Quadbot | cyclic_reposter | orthogonal projection | -0.74 | [-1.48, -0.08] |
| Quadbot | uniform_scheduler | structure output | -0.20 | [-0.70, +0.30] |
| Quadbot | uniform_scheduler | orthogonal projection | -0.10 | [-0.41, +0.20] |

The largest absolute mean change from masking structure output is 0.36
percentage points; from skipping orthogonal projection it is 0.74.
Neither control repairs the burst-policy failure. The diagnostic does
not deliver the requested one-percentage-point model improvement.

## Decision and Scope

Removal has a higher mean in 1 of 12 comparisons and a lower mean in 9.
Neither direction alone establishes a better final model. Retain every
policy and seed; do not select a module configuration by these exposed
outer challenge scores, hide better ablations or invent improvements.

The structure control computes the existing branch and then masks its
output. Inactive parameters remain serialized, and graph-context rows
still affect training BatchNorm. It is not fully graph-free or a
reduced-computation control. The orthogonal control retains separate
branch normalization while skipping mutual projection. Their joint
removal is algebraically redundant once structure output is zero.

Both removals retain the same timing inputs, seeded initialization,
parameter layout, fold-local feature-kNN context, optimizer and
inner-profile checkpoint selection. Feature-kNN is not social-edge data;
challenge inference uses unlabelled transductive context.

The earlier burst-policy failure remains relevant and is not cancelled
by high scores on rule-like policies. No uniform improvement, SOTA,
independent robustness or submission-readiness claim follows here.

Any future method must be selected using fitting/inner-selection evidence
and evaluated independently of these exposed challenges. The working
manuscript and old published results were not replaced by these diagnostics.
Raw checkpoints, account predictions and individual trajectories remain private.
